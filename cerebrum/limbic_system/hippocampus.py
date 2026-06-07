"""
layer0_model.py —— L0 生成模型 + ClusterNetwork
自由能原理智能体 — M1 单智能体生存

功能：
- 隐状态 z 的初始化
- 从 z 预测感知输入 s (线性解码)
- ClusterNetwork: 特征哈希 + 侧抑制回忆 + Hebb-like 学习
- 睡眠巩固周期

v6.1 新增:
- STDP 时序学习 (pre→post = LTP, post→pre = LTD)
- 保护信号 (CD47-SIRPα "别吃我")
- 沉默突触候选集群 (NMDA-only → AMPA 觉醒)
- PNN 结构锁定 (周围神经网络包裹)
"""

from typing import Optional
import numpy as np
from cns.data_types import (
    D, H, K, Cluster, CandidateCluster, Theta,
)


def _auto_mask(s: np.ndarray) -> np.ndarray:
    """非零维度 = 有效查询维度。全零 → 全维度。"""
    mask = np.abs(s) > 1e-6
    if not np.any(mask):
        return np.ones(len(s), dtype=bool)
    return mask


def _masked_cosine(a: np.ndarray, b: np.ndarray,
                   mask: np.ndarray) -> float:
    """只在 mask 维度上计算余弦相似度"""
    a_m, b_m = a[mask], b[mask]
    denom = np.linalg.norm(a_m) * np.linalg.norm(b_m) + 1e-8
    return float(np.dot(a_m, b_m) / denom)


# ============================================================
# 隐状态
# ============================================================

# ============================================================
# 生成模型: 集群预测 s_pred (回退: z → s_pred 线性解码)
# ============================================================

def predict_sensations(z: np.ndarray, theta: Theta) -> np.ndarray:
    """线性解码：从隐状态 z 预测感知输入 x_hat
    W: (D, H) 简化为恒等投影（取 z 的前 D 维，不足补零）
    """
    D_dim = D if D <= len(z) else len(z)
    s_pred = np.zeros(D)
    s_pred[:D_dim] = z[:D_dim]
    # 缩放: sigma_x 控制预测的"自信度"
    return s_pred


# ============================================================
# ClusterNetwork — Hebb 细胞集群记忆网络
# ============================================================

class ClusterNetwork:
    """基于 Hebb 学习的细胞集群记忆网络

    核心原则：
    - 集群形成 = 学习（新感知创建簇）
    - 集群激活 = 回忆（输入部分匹配 → 簇被激活）
    - 集群衰减 = 遗忘（未使用的簇逐渐衰减）
    - 无暴力扫描：哈希定位 O(1) + 桶内竞争 O(K·dim)
    """

    def __init__(self, theta: Theta, hash_offset: int = 0):
        self.clusters: list[Cluster] = []
        self.buckets: dict = {}  # {hash_key → list[Cluster]}
        self.theta = theta
        self.hash_offset = hash_offset  # 用于视觉/音频等非文本通道的哈希偏移
        self.learn_rate_modifier: float = 1.0  # v5.5: VTA RPE → 事件驱动学习率调制
        self._context_dim: int = 8            # v6.0: 情境向量维度 (身体+情感快照)
        self._context_vector: np.ndarray = np.zeros(8, dtype=np.float32)
        # v6.1: STDP 时序追踪
        self._last_activated_id: int = -1         # 上一个激活簇的 id()
        self._last_activated_step: int = -1       # 上一个激活的步数
        self._step_counter: int = 0               # 内部步数计数
        # v6.1: 沉默突触候选集群
        self._candidate_clusters: list[CandidateCluster] = []
        self._candidate_max: int = theta.candidate_max

    # ----- v6.1: 发育调制因子 -----
    def _glun2b_factor(self) -> float:
        """GluN2B 占比 → 学习率调制因子 [0.55, 1.0].

        高 GluN2B (婴儿期 ~0.9) → 因子 ~0.95 (高可塑性)
        低 GluN2B (成年期 ~0.1) → 因子 ~0.55 (低可塑性)
        """
        r = self.theta.glun2b_ratio
        return 0.5 + 0.5 * r

    def _glun2b_threshold_factor(self) -> float:
        """GluN2B → 匹配阈值调制 [0.73, 1.0].

        高 GluN2B → 低阈值 (更容易匹配，更宽的整合时间窗)
        """
        r = self.theta.glun2b_ratio
        return 0.7 + 0.3 * r

    # ----- v6.1: STDP 更新 -----
    def _stdp_update(self, pre_cluster: Cluster, post_cluster: Cluster):
        """STDP: pre→post = LTP 增强, 反向 = LTD 减弱.

        在 pre_cluster 的 stdp_links 中增强到 post_cluster 的权重。
        权重衰减与时间间隔成正比。
        """
        dt = self._step_counter - self._last_activated_step
        if dt > self.theta.stdp_window:
            return  # 超出时间窗口

        # 时间衰减: 越近越强
        time_factor = max(0.0, 1.0 - dt / self.theta.stdp_window)
        delta = self.theta.stdp_lr * time_factor

        post_id = id(post_cluster)
        old_w = pre_cluster.stdp_links.get(post_id, 0.0)
        new_w = np.clip(old_w + delta, -1.0, 1.0)
        pre_cluster.stdp_links[post_id] = float(new_w)

    # ----- 特征哈希 -----
    def hash_features(self, s: np.ndarray) -> np.ndarray:
        """特征哈希：将感知向量映射到固定维度
        使用 tanh 非线性，输出范围 [-1, 1]
        """
        return np.tanh(s + 1e-8)

    def _hash_to_bucket(self, h: np.ndarray) -> int:
        """前 8 维符号位 → 8-bit 桶索引。

        使用 hash_offset 可对非文本通道（视觉[64:], 音频[128:], etc.）
        进行有效哈希，避免所有零维度的桶碰撞。
        """
        offset = self.hash_offset
        bits = (h[offset:offset+8] > 0).astype(int)
        return int(sum(bits[i] << i for i in range(8)))

    # ----- 情境设置 (v6.0) -----
    def set_context(self, body_b: np.ndarray = None,
                    valence: float = 0.0, arousal: float = 0.0):
        """设置当前情境向量 — 用于编码特异性原则的提取匹配。

        Args:
            body_b: 身体状态 (前6维)
            valence: 当前效价 [-1, 1]
            arousal: 当前唤醒 [0, 1]
        """
        ctx = np.zeros(self._context_dim, dtype=np.float32)
        if body_b is not None:
            n = min(len(body_b), 6)
            ctx[:n] = body_b[:n]
        ctx[6] = valence
        ctx[7] = arousal
        self._context_vector = ctx

    # ----- 连续相似度 (M2) -----
    def best_similarity(self, s: np.ndarray) -> float:
        """返回与输入 s 的最高余弦相似度 [0, 1]，不设阈值

        用于 M2 连续认知价值：sim → 0 时信息增益最大。
        空网络返回 0.0（完全新颖）。
        """
        if not self.clusters:
            return 0.0
        h = self.hash_features(s)
        mask = _auto_mask(s)
        return float(max(
            _masked_cosine(h, c.centroid, mask) for c in self.clusters))

    # ----- 回忆 (recall) — 桶内扫描 + 掩码匹配 O(K) -----
    def recall(self, s: np.ndarray, mask: np.ndarray = None) -> Optional[Cluster]:
        """检索与输入 s 最匹配的簇。

        mask 自动从非零维度推导 → 纯文本查询只在 text[0:64] 比较，
        纯视觉只在 vision[64:128] 比较。跨通道检索零额外成本。

        v6.1: +STDP 时序追踪 + 保护信号 + GluN2B 调制阈值
        """
        if not self.clusters:
            return None

        if mask is None:
            mask = _auto_mask(s)

        h = self.hash_features(s)
        hash_key = self._hash_to_bucket(h)
        bucket = self.buckets.get(hash_key, self.clusters)

        best_sim = -1.0
        best_c = None
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim = sim
                best_c = c

        # ---- v6.0 编码特异性: 情境不匹配 → 提取变难 ----
        if best_c is not None and np.any(self._context_vector):
            if best_c.centroid.shape[0] > 72:
                encoded_ctx = best_c.centroid[64:72]
                if np.any(np.abs(encoded_ctx) > 0.01):
                    ctx_sim = _masked_cosine(
                        self._context_vector[:8],
                        encoded_ctx[:8],
                        np.ones(8, dtype=bool))
                    ctx_factor = 0.3 + 0.7 * max(0.0, ctx_sim)
                    best_sim *= ctx_factor

        # 自适应阈值: 部分查询 → 放宽要求
        # v6.1: GluN2B 调制 — 高 GluN2B → 低阈值 (更宽整合窗)
        active_ratio = float(np.sum(mask)) / max(len(mask), 1)
        base_threshold = self.theta.cluster_threshold * (0.4 + 0.6 * active_ratio)
        eff_threshold = base_threshold * self._glun2b_threshold_factor()

        # ---- v6.2: 亚阈值标签 ----
        # 接近匹配但未达阈值的簇也可能在后续被巩固 (STC 假说)
        if best_c is not None and best_sim >= eff_threshold * 0.7:
            best_c.tag = max(best_c.tag, best_sim * 0.3)
            best_c.tag_age = 0

        if best_c is not None and best_sim >= eff_threshold:
            # ---- v6.1: STDP 时序学习 ----
            prev_cluster = None
            if self._last_activated_id >= 0:
                for c in self.clusters:
                    if id(c) == self._last_activated_id:
                        prev_cluster = c
                        break
            if prev_cluster is not None and prev_cluster is not best_c:
                self._stdp_update(prev_cluster, best_c)
            self._last_activated_id = id(best_c)
            self._last_activated_step = self._step_counter

            # ---- v6.1: 保护信号 ----
            best_c.protection_score += 0.02 * best_c.activation

            # ---- v6.2: 突触标签 + 激活持续性 ----
            best_c.tag = max(best_c.tag, best_c.activation * 0.5)
            best_c.tag_age = 0
            best_c.activation_persistence = 1.0  # CaMKII 样峰值

            best_c.activation = min(1.0, best_c.activation + 0.1)
            best_c.age += 1
            return best_c
        return None

    # ----- Hebb 扩散 — Broca 区等价物 (v6.1: +STDP 偏置) -----
    def diffuse(self, start_cluster, steps: int = 3,
                top_k: int = 5, sim_threshold: float = 0.15) -> tuple:
        """沿 Hebb 边随机游走 — 返回路径上所有质心。

        v6.1: 混合余弦相似度 + STDP 权重。
        混合权重 = (1 - stdp_weight) * cos_sim + stdp_weight * stdp_signals

        Returns:
            (path_centroids, path_indices)
        """
        if self.n_clusters < 2 or steps < 1:
            c = start_cluster.centroid[:64].copy()
            return [c], [0]

        C = np.stack([c.centroid for c in self.clusters])
        start_idx = next(i for i, c in enumerate(self.clusters)
                        if c is start_cluster)

        path_centroids = [C[start_idx][:64].copy()]
        path_indices = [start_idx]

        current_idx = start_idx
        for _ in range(steps):
            cur_vec = C[current_idx]
            cur_cluster = self.clusters[current_idx]
            dot = C @ cur_vec
            norms = np.linalg.norm(C, axis=1)
            cos_sims = dot / (norms * norms[current_idx] + 1e-8)
            cos_sims[current_idx] = -1.0

            # v6.1: 混合 STDP 权重
            stdp_signals = np.zeros(len(self.clusters))
            if cur_cluster.stdp_links:
                for i, c in enumerate(self.clusters):
                    stdp_signals[i] = cur_cluster.stdp_links.get(id(c), 0.0)

            mixed_sims = ((1.0 - self.theta.stdp_weight) * cos_sims
                          + self.theta.stdp_weight * stdp_signals)
            mixed_sims[current_idx] = -1.0

            top_indices = np.argsort(mixed_sims)[-top_k:]
            weights = mixed_sims[top_indices]
            weights = np.clip(weights - sim_threshold, 0, None)
            if weights.sum() < 1e-8:
                break
            weights = weights / weights.sum()
            current_idx = int(np.random.choice(top_indices, p=weights))
            path_centroids.append(C[current_idx][:64].copy())
            path_indices.append(current_idx)

        return path_centroids, path_indices

    # ----- 相干漫游 — 保持在语义场内 -----
    def coherent_diffuse(self, start_cluster, steps=3,
                         top_k=8, sim_threshold=0.15,
                         coherence_threshold=0.28):
        """相干漫游: 每步候选必须同时接近前一步和话题锚点。"""
        if self.n_clusters < 2 or steps < 1:
            c = start_cluster.centroid[:64].copy()
            return [c], [0]

        C = np.stack([c.centroid for c in self.clusters])
        start_idx = next(i for i, c in enumerate(self.clusters) if c is start_cluster)
        anchor = C[start_idx].copy()

        path_centroids = [C[start_idx][:64].copy()]
        path_indices = [start_idx]
        current_idx = start_idx

        for _ in range(steps):
            cur_vec = C[current_idx]
            norm_c = np.linalg.norm(cur_vec)
            dot = C @ cur_vec
            norms = np.linalg.norm(C, axis=1)
            sims = dot / (norms * norm_c + 1e-8)
            sims[current_idx] = -1.0

            anchor_sims = (C @ anchor) / (norms * np.linalg.norm(anchor) + 1e-8)

            candidate_order = np.argsort(sims)[::-1]
            candidates, weights = [], []
            for ci in candidate_order[:top_k*2]:
                if sims[ci] < sim_threshold: continue
                if anchor_sims[ci] < coherence_threshold: continue
                candidates.append(ci); weights.append(sims[ci])
                if len(candidates) >= top_k: break

            if not candidates: break
            weights = np.array(weights) / sum(weights)
            current_idx = int(np.random.choice(candidates, p=weights))
            path_centroids.append(C[current_idx][:64].copy())
            path_indices.append(current_idx)

        return path_centroids, path_indices

    # ----- v6.1: 沉默突触候选匹配 -----
    def _find_candidate(self, h: np.ndarray) -> Optional[CandidateCluster]:
        """在候选集群中查找最佳匹配.

        Returns:
            最佳匹配的 CandidateCluster，若无则 None
        """
        if not self._candidate_clusters:
            return None
        best_sim = -1.0
        best_cc = None
        for cc in self._candidate_clusters:
            sim = _masked_cosine(h, cc.centroid, np.ones(len(h), dtype=bool))
            if sim > best_sim:
                best_sim = sim
                best_cc = cc
        if best_cc is not None:
            best_cc.max_similarity = max(best_cc.max_similarity, best_sim)
        return best_cc

    def _awaken_candidate(self, cc: CandidateCluster) -> Cluster:
        """沉默突触觉醒: CandidateCluster → 完整 Cluster (AMPA 插入).

        移除候选，创建新 Cluster，给予高初始 activation。
        """
        self._candidate_clusters = [
            x for x in self._candidate_clusters if x is not cc]
        if len(self.clusters) < K:
            c = Cluster(centroid=cc.centroid.copy())
            c.activation = 0.4  # 觉醒时高激活
            c.count = cc.exposure_count
            self.clusters.append(c)
            hash_key = self._hash_to_bucket(cc.centroid)
            self.buckets.setdefault(hash_key, []).append(c)
            return c
        else:
            # 容量满 → 替换最旧且保护最低的簇
            oldest = min(self.clusters,
                        key=lambda c: c.age - c.protection_score * 50)
            old_key = self._hash_to_bucket(oldest.centroid)
            if old_key in self.buckets:
                self.buckets[old_key] = [
                    x for x in self.buckets[old_key] if x is not oldest]
            oldest.centroid = cc.centroid.copy()
            oldest.count = cc.exposure_count
            oldest.age = 0
            oldest.activation = 0.4
            oldest.protection_score = 0.1
            oldest.pnn_level = 0.0
            oldest.stdp_links = {}
            hash_key = self._hash_to_bucket(cc.centroid)
            self.buckets.setdefault(hash_key, []).append(oldest)
            return oldest

    # ----- v6.2: 突触标签捕获 (STC 假说) -----
    def capture_tags(self, arousal: float = 0.0, F_body_delta: float = 0.0):
        """高唤醒 → 带标签的簇获得额外学习率 (STC §2.2.4).

        模拟: 全细胞合成 PRPs → 只有带标签的突触能捕获。
        高唤醒/高 F_body 偏差 = 细胞检测到"重要事件" → 触发 PRP 合成。

        Args:
            arousal: 当前唤醒度 [0, 1]
            F_body_delta: F_body 变化幅度 (正=变差)
        """
        event_strength = max(0.0, arousal + 0.5 * abs(F_body_delta))
        if event_strength < 0.3:
            return  # 事件不够强，不触发捕获

        capture_lr = self.theta.tag_capture_strength * event_strength
        n_captured = 0
        for c in self.clusters:
            if c.tag > 0.01 and c.tag_age < self.theta.tag_window:
                # 标签捕获: 向当前质心方向微调 (self-reinforcement)
                # 不依赖具体输入 — 标签是"这个簇重要"的信号
                c.activation = min(1.0, c.activation + capture_lr * c.tag * 0.2)
                c.tag *= 0.5  # 标签消耗 (捕获后减弱)
                n_captured += 1

    # ----- v6.2: 激活持续性调制 -----
    def _persistence_factor(self, cluster: Cluster) -> tuple[float, float]:
        """CaMKII 样持续性 → 阈值降低 + 学习率提升因子.

        Returns:
            (threshold_factor [0.7, 1.0], lr_factor [1.0, 1.5])
        """
        p = cluster.activation_persistence
        if p < 0.01:
            return (1.0, 1.0)
        threshold_mod = 1.0 - self.theta.persistence_threshold_boost * p
        lr_mod = 1.0 + self.theta.persistence_lr_boost * p
        return (float(threshold_mod), float(lr_mod))

    # ----- 学习 (learn) -----
    def learn(self, s: np.ndarray) -> Cluster:
        """学习新感知模式：匹配到则更新，否则创建新簇

        策略：
        1. 若 recall 匹配 → 以 learn_rate 向输入更新 centroid
        2. 若未匹配且亚阈值 → v6.1 沉默突触: 候选集群追踪
        3. 若未匹配且簇数 < K → 创建新簇
        4. 若未匹配且簇数 >= K → 替换最旧/最弱簇

        v6.1: +STDP +保护信号 +GluN2B调制 +PNN锁定 +沉默突触
        v6.2: +突触标签捕获 +激活持续性调制

        Returns:
            被创建或更新的 Cluster
        """
        h = self.hash_features(s)
        existing = self.recall(s)

        if existing is not None:
            # ---- 更新已有簇 ----
            old_key = self._hash_to_bucket(existing.centroid)
            # v5.5: VTA RPE 调制 × v6.1: GluN2B 发育调制
            lr = (self.theta.learn_rate_l0
                  * self.learn_rate_modifier
                  * self._glun2b_factor())
            # 激活调制: 更强的匹配 → 更强的学习 (Hebb 机制)
            activation_factor = 0.3 + 0.7 * existing.activation  # [0.3, 1.0]

            # ---- v6.2: 激活持续性调制 (CaMKII 窗口) ----
            _, persistence_lr_factor = self._persistence_factor(existing)
            activation_factor *= persistence_lr_factor

            # ---- v6.1: PNN 结构锁定 ----
            pnn_resistance = existing.pnn_level * 0.8  # PNN 减少最多 80% 学习率
            effective_lr = lr * activation_factor * (1.0 - pnn_resistance)

            existing.centroid = (1 - effective_lr) * existing.centroid + effective_lr * h
            existing.count += 1
            # ---- v6.1: PNN 累积 (用得越多越固化) ----
            existing.pnn_level = min(1.0, existing.pnn_level
                                     + self.theta.pnn_formation_rate * existing.count)
            # ---- v6.1: 保护信号 ----
            existing.protection_score += 0.01

            # 如果 centroid 变化导致桶改变，迁移
            new_key = self._hash_to_bucket(existing.centroid)
            if new_key != old_key and old_key in self.buckets:
                self.buckets[old_key] = [
                    c for c in self.buckets[old_key] if c is not existing]
                self.buckets.setdefault(new_key, []).append(existing)

            # ---- v6.0 倒摄干扰 ----
            ri_bucket = self.buckets.get(new_key, [])
            for other_c in ri_bucket:
                if other_c is not existing:
                    sim_to_new = _masked_cosine(h, other_c.centroid,
                                              np.ones(len(h), dtype=bool))
                    if sim_to_new > 0.75:
                        other_c.activation *= 0.97

            # ---- v6.1: STDP 时序追踪 ----
            self._last_activated_id = id(existing)
            self._last_activated_step = self._step_counter

            return existing

        # ---- 无匹配 → 检查沉默突触候选 (v6.1) ----
        # 计算最佳相似度 (即使低于阈值)
        best_sim = 0.0
        best_cc = None
        if self._candidate_clusters:
            best_cc = self._find_candidate(h)
            best_sim = best_cc.max_similarity if best_cc else 0.0

        # 实际最佳相似度 (从 recall 失败中推断——我们需重新扫描)
        # 为了效率: 仅在无匹配时扫描 bucket 找亚阈值最佳匹配
        hash_key = self._hash_to_bucket(h)
        bucket = self.buckets.get(hash_key, self.clusters)
        full_mask = np.ones(len(h), dtype=bool)
        nearest_sim = 0.0
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, full_mask)
            if sim > nearest_sim:
                nearest_sim = sim

        # 沉默突触范围: [threshold-0.2, threshold)
        silence_low = self.theta.cluster_threshold - 0.2
        silence_high = self.theta.cluster_threshold

        if silence_low <= nearest_sim < silence_high:
            # 亚阈值匹配 → 沉默突触候选
            if best_cc is not None and best_cc.max_similarity > nearest_sim * 0.9:
                # 更新已有候选
                best_cc.exposure_count += 1
                best_cc.centroid = (0.7 * best_cc.centroid + 0.3 * h)
                best_cc.max_similarity = max(best_cc.max_similarity, nearest_sim)
                best_cc.age = 0
            else:
                # 新建候选
                cc = CandidateCluster(centroid=h.copy())
                cc.max_similarity = nearest_sim
                self._candidate_clusters.append(cc)
                # 超限清理
                if len(self._candidate_clusters) > self._candidate_max:
                    self._candidate_clusters.sort(key=lambda x: x.age, reverse=True)
                    self._candidate_clusters = self._candidate_clusters[
                        :self._candidate_max]

            # 检查觉醒条件
            if best_cc and (best_cc.exposure_count >= 3
                           or best_cc.max_similarity > self.theta.cluster_threshold - 0.05):
                awakened = self._awaken_candidate(best_cc)
                # STDP: 从上一个活跃簇到觉醒簇
                prev_cluster = None
                if self._last_activated_id >= 0:
                    for c in self.clusters:
                        if id(c) == self._last_activated_id and c is not awakened:
                            prev_cluster = c
                            break
                if prev_cluster is not None:
                    self._stdp_update(prev_cluster, awakened)
                self._last_activated_id = id(awakened)
                self._last_activated_step = self._step_counter
                return awakened

            # 候选存在但未觉醒 → 不创建完整 Cluster, 返回 None-like 行为
            # 需要一个占位返回 — 返回最佳候选的信息包装为特殊 Cluster
            # (实际返回 None 会导致上游 learn 认为无学习，所以创建临时 marker)
            self._step_counter += 1
            return None  # 沉默——不形成完整记忆

        # ---- 无匹配 + 不在亚阈值范围 → 创建新簇 (新颖性检测) ----
        # ---- v6.0 前摄干扰 ----
        bucket = self.buckets.get(hash_key, [])
        pi_n_similar = sum(1 for c in bucket
                          if _masked_cosine(h, c.centroid,
                                          np.ones(len(h), dtype=bool)) > 0.65)
        pi_factor = 1.0 / (1.0 + pi_n_similar * 0.12)
        centroid_new = h.copy() * pi_factor
        newly_created = None
        if len(self.clusters) < K:
            c = Cluster(centroid=centroid_new)
            self.clusters.append(c)
            hash_key = self._hash_to_bucket(centroid_new)
            self.buckets.setdefault(hash_key, []).append(c)
            newly_created = c
        else:
            # 容量满 → v6.1: 替换最旧且保护最低的簇
            oldest = min(self.clusters,
                        key=lambda c: c.age - c.protection_score * 50)
            old_key = self._hash_to_bucket(oldest.centroid)
            if old_key in self.buckets:
                self.buckets[old_key] = [
                    c for c in self.buckets[old_key] if c is not oldest]
            oldest.centroid = centroid_new
            oldest.count = 1
            oldest.age = 0
            oldest.activation = 0.0
            oldest.protection_score = 0.0
            oldest.pnn_level = 0.0
            oldest.stdp_links = {}
            hash_key = self._hash_to_bucket(centroid_new)
            self.buckets.setdefault(hash_key, []).append(oldest)
            newly_created = oldest

        # ---- v6.0 倒摄干扰 ----
        ri_bucket = self.buckets.get(hash_key, [])
        for other_c in ri_bucket:
            if other_c is not newly_created:
                sim_to_new = _masked_cosine(h, other_c.centroid,
                                          np.ones(len(h), dtype=bool))
                if sim_to_new > 0.75:
                    other_c.activation *= 0.97

        # ---- v6.1: STDP 时序追踪 (新簇也参与 STDP) ----
        prev_cluster = None
        if self._last_activated_id >= 0:
            for c in self.clusters:
                if id(c) == self._last_activated_id and c is not newly_created:
                    prev_cluster = c
                    break
        if prev_cluster is not None:
            self._stdp_update(prev_cluster, newly_created)
        self._last_activated_id = id(newly_created)
        self._last_activated_step = self._step_counter

        return newly_created

    # ----- 桶迁移 -----
    def _migrate_bucket(self, cluster: Cluster, old_centroid: np.ndarray):
        """如果簇的 centroid 改变导致桶变化，迁移到新桶"""
        old_key = self._hash_to_bucket(old_centroid)
        new_key = self._hash_to_bucket(cluster.centroid)
        if new_key != old_key and old_key in self.buckets:
            self.buckets[old_key] = [
                c for c in self.buckets[old_key] if c is not cluster]
            self.buckets.setdefault(new_key, []).append(cluster)

    # ----- 衰减 -----
    def decay(self):
        """衰减所有簇的激活值 + v6.1: 保护信号衰减 + 候选集群老化.

        v6.2: 突触标签衰减 + 激活持续性衰减.
        未激活的簇随时间 activation → 0（自然遗忘）。
        保护信号缓慢衰减——长期不用的簇逐渐失去保护。
        """
        for c in self.clusters:
            # ---- v6.2: 巩固锁调制 decay ----
            lock_divisor = 1.0 + c.consolidation_count * self.theta.consolidation_lock_factor
            effective_decay = self.theta.decay_rate / min(lock_divisor,
                                                          self.theta.consolidation_lock_max)
            c.activation *= (1 - effective_decay)
            c.age += 1
            # v6.1: 保护信号缓慢衰减
            c.protection_score *= self.theta.protection_decay
            # ---- v6.2: 突触标签衰减 ----
            if c.tag > 0.001:
                c.tag *= (1.0 - self.theta.tag_decay_rate)
                c.tag_age += 1
                if c.tag_age > self.theta.tag_window:
                    c.tag = 0.0  # 过期标签清除
            # ---- v6.2: 激活持续性衰减 ----
            c.activation_persistence *= (1.0 - self.theta.persistence_decay_rate)
            if c.activation_persistence < 0.001:
                c.activation_persistence = 0.0  # 归零防止浮点积累

        # v6.1: 候选集群老化
        for cc in self._candidate_clusters:
            cc.age += 1
        # 移除老化候选 (age > 20 且暴露次数 < 2)
        self._candidate_clusters = [
            cc for cc in self._candidate_clusters
            if not (cc.age > 20 and cc.exposure_count < 2)
        ]

        # v6.1: 步数计数
        self._step_counter += 1

    # ----- 统计 -----
    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def total_activation(self) -> float:
        return float(sum(c.activation for c in self.clusters))

    def get_top_clusters(self, n: int = 5) -> list[Cluster]:
        """返回激活度最高的 n 个簇"""
        sorted_clusters = sorted(self.clusters, key=lambda c: c.activation, reverse=True)
        return sorted_clusters[:n]

    # ----- v6.1: PNN 消化 (情感事件触发) -----
    def digest_pnn(self, arousal: float, novelty: float):
        """高唤醒+高新颖性 → 临时降低活跃簇的 PNN (模拟 MMP-9 降解).

        软骨素酶 ABC 等价物: 去甲肾上腺素 → MMP-9 激活 → PNN 降解。
        在 agent.step() 中由高情感事件驱动调用。

        Args:
            arousal: 当前唤醒度 [0, 1]
            novelty: 新颖性信号 [0, 1]
        """
        digest_strength = 0.05 * arousal * novelty
        if digest_strength <= 0.001:
            return
        for c in self.clusters:
            if c.activation > 0.1:  # 只消化活跃簇的 PNN
                c.pnn_level = max(0.0, c.pnn_level - digest_strength)

    # ----- v6.1: 统计 ----
    @property
    def n_candidates(self) -> int:
        return len(self._candidate_clusters)

    @property
    def mean_pnn(self) -> float:
        if not self.clusters:
            return 0.0
        return float(sum(c.pnn_level for c in self.clusters) / len(self.clusters))

    @property
    def mean_protection(self) -> float:
        if not self.clusters:
            return 0.0
        return float(sum(c.protection_score for c in self.clusters) / len(self.clusters))

    @property
    def n_stdp_links(self) -> int:
        return sum(len(c.stdp_links) for c in self.clusters)


# ============================================================
# 引导与睡眠
# ============================================================

def bootstrap_clusters(samples: list[np.ndarray], theta: Theta) -> ClusterNetwork:
    """用种子样本预训练簇网络 —— 冷启动

    Args:
        samples: 初始感知样本列表
        theta: 参数配置

    Returns:
        已预训练的 ClusterNetwork
    """
    net = ClusterNetwork(theta)
    for s in samples:
        net.learn(s)
    return net


def sleep_cycle(net: ClusterNetwork, theta: Theta) -> int:
    """睡眠巩固周期

    模拟记忆巩固：
    1. 衰减所有簇激活值
    2. 移除极度不活跃的簇（activation ≈ 0）
    3. 保留强激活簇（长期记忆）

    Returns:
        被移除的簇数量
    """
    net.decay()
    n_before = len(net.clusters)
    removed = [c for c in net.clusters if c.activation <= 0.01]
    net.clusters = [c for c in net.clusters if c.activation > 0.01]
    # 同步清理桶
    for c in removed:
        key = net._hash_to_bucket(c.centroid)
        if key in net.buckets:
            net.buckets[key] = [
                x for x in net.buckets[key] if x is not c]
    n_removed = n_before - len(net.clusters)
    return n_removed


# ============================================================
# 睡眠回放巩固 (Sleep Replay Consolidation)
# ============================================================

def sleep_replay(net: ClusterNetwork, theta: Theta,
                 replay_lr: float = 0.04,
                 n_replay_cycles: int = 1,
                 cross_link_strength: float = 0.005,
                 replay_noise: float = 0.006,
                 min_activation_for_replay: float = 0.03) -> dict:
    """睡眠回放巩固 —— 部分模式重放 + 模式分离

    神经科学基础：
    - **Sharp-wave ripples** (海马): 用部分线索重放，锻炼模式补全
    - **Pattern separation** (齿状回): 推离过于相似的记忆，增加可辨别性
    - **Slow-wave sleep** (皮层): 衰减 + 清理弱记忆

    算法：
    1. 选择近期活跃簇 (activation > threshold) 作为重放候选
    2. 对每个候选簇，用部分维度 (纯视觉/纯文本/随机子集) 触发 recall
       → 锻炼跨模态模式补全能力 (这正是 V→T 检索需要的)
    3. 模式分离: 找到过于相似的簇 (cos>0.85)，在差异维度上推离
       → 防止表示坍缩，增加簇间可辨别性
    4. 衰减 + 移除死簇

    这替代了旧的"跨簇拉近"方式——拉近相似簇会降低可辨别性，恶化检索。
    正确的睡眠巩固应该: 内部一致化 (模式补全) + 外部区分化 (模式分离)。

    Args:
        net: ClusterNetwork 实例
        theta: 参数配置
        replay_lr: 未使用 (保留兼容)
        n_replay_cycles: 重放轮数
        cross_link_strength: 模式分离强度 (旧名保留兼容)
        replay_noise: 未使用 (保留兼容)
        min_activation_for_replay: 重放候选的最低激活值

    Returns:
        stats dict:
            n_replayed:     总重放次数 (partial recall 调用数)
            n_linked:       模式分离对数
            n_removed:      移除的簇数
            replay_boost:   总激活增量
            n_clusters:     巩固后簇数
            n_candidates:   重放候选数
            mean_activation: 平均激活值
    """
    if net.n_clusters == 0:
        return {'n_replayed': 0, 'n_linked': 0, 'n_removed': 0,
                'replay_boost': 0.0, 'n_clusters': 0,
                'n_candidates': 0, 'mean_activation': 0.0}

    # ---- Step 1: 选择重放候选 ----
    # 激活值高于阈值的簇参与重放 (近期使用过的记忆)
    mean_act = sum(c.activation for c in net.clusters) / net.n_clusters
    replay_threshold = max(min_activation_for_replay, mean_act * 0.4)

    candidates = [c for c in net.clusters if c.activation > replay_threshold]

    if not candidates:
        # Fallback: 至少重放 top 20% 最活跃簇
        sorted_clusters = sorted(net.clusters, key=lambda c: c.activation,
                                 reverse=True)
        n_top = max(3, net.n_clusters // 5)
        candidates = sorted_clusters[:n_top]

    # ---- Step 2: 海马重放 + 模式分离 ----
    n_replayed = 0
    total_boost = 0.0
    n_separated = 0

    for cycle in range(n_replay_cycles):
        cycle_decay = 0.8 ** cycle

        # --- 2a. 部分模式重放 (Partial Pattern Replay) ---
        # 用部分维度 (纯视觉 or 纯文本) 触发 recall，
        # 锻炼跨模态模式补全能力。不直接修改 centroid，
        # 让 Hebb recall 的 activation 增强自然选择内部一致的簇。
        for c in candidates:
            salience = 0.3 + 0.7 * c.activation
            n_replays = int(1 + salience * 1.5)  # 1-2 次/簇/轮

            centroid = c.centroid
            for ri in range(n_replays):
                # 交替: 纯视觉查询、纯文本查询、随机子集
                partial = np.zeros_like(centroid)
                mode_idx = (n_replayed + ri) % 3
                if mode_idx == 0:
                    # Visual → Text: 只保留视觉通道 [64:192]
                    partial[64:192] = centroid[64:192]
                elif mode_idx == 1:
                    # Text → Visual: 只保留文本通道 [0:64]
                    partial[0:64] = centroid[0:64]
                else:
                    # 随机 60% 维度 (模拟不完整记忆)
                    keep = np.random.random(len(centroid)) > 0.4
                    partial[keep] = centroid[keep]

                # Recall 用部分输入 → 激活匹配簇 → Hebb 强化
                # 不加噪声、不直接改 centroid；
                # recall 的 activation boost 自然选择跨模态一致的簇
                net.recall(partial)
                n_replayed += 1

            # 温和的激活增强 (模拟记忆巩固，远小于 recall 的 0.1)
            boost = 0.005 * salience * cycle_decay
            c.activation = min(1.0, c.activation + boost)
            total_boost += boost

        # --- 2b. 模式分离 (Pattern Separation) ---
        # 找到过于相似的簇 (cos > 0.85)，在差异最大的维度上推离
        # 模拟齿状回功能: 增加簇间可辨别性，防止表示坍缩
        C = np.stack([c.centroid for c in net.clusters])
        cid_to_idx = {id(c): i for i, c in enumerate(net.clusters)}
        candidate_indices = [cid_to_idx[id(c)] for c in candidates]

        # 跟踪已处理的配对 (避免重复推离)
        separated_pairs = set()

        for ci in candidate_indices:
            cur = C[ci]
            cur_norm = np.linalg.norm(cur)
            dot = C @ cur
            norms = np.linalg.norm(C, axis=1)
            sims = dot / (norms * cur_norm + 1e-8)
            sims[ci] = -1.0

            # 只关注最相似的邻居
            best_j = int(np.argmax(sims))
            best_sim = float(sims[best_j])

            # 过于相似 → 推离 (模式分离)
            if best_sim > 0.85:
                pair_key = tuple(sorted([ci, best_j]))
                if pair_key in separated_pairs:
                    continue
                separated_pairs.add(pair_key)

                # 找到差异最大的维度 (这些是区分的依据)
                diff = np.abs(cur - C[best_j])
                top_diff_dims = np.argsort(diff)[-32:]  # top 10% dims

                # 在差异维度上温和推离 (分离强度)
                sep_strength = cross_link_strength * (best_sim - 0.85) * cycle_decay
                push = sep_strength * 0.5

                old_ci = net.clusters[ci].centroid.copy()
                old_best = net.clusters[best_j].centroid.copy()

                # 互相推离 (只在差异最大的维度)
                net.clusters[ci].centroid[top_diff_dims] += push * diff[top_diff_dims]
                net.clusters[best_j].centroid[top_diff_dims] -= push * diff[top_diff_dims]

                net._migrate_bucket(net.clusters[ci], old_ci)
                net._migrate_bucket(net.clusters[best_j], old_best)
                n_separated += 1

                # ---- v6.1: PNN 在参与模式分离的簇上累积 ----
                net.clusters[ci].pnn_level = min(1.0,
                    net.clusters[ci].pnn_level + 0.002)
                net.clusters[best_j].pnn_level = min(1.0,
                    net.clusters[best_j].pnn_level + 0.002)

        # 更新质心矩阵
        if cycle < n_replay_cycles - 1:
            C = np.stack([c.centroid for c in net.clusters])

    # ---- Step 3: 衰减 (自然遗忘) ----
    net.decay()

    # ---- Step 4: v6.1 保护感知修剪 ----
    n_before = net.n_clusters
    removed = []
    for c in net.clusters:
        # 保护信号调制修剪阈值: 高频使用的簇更难被删
        protected_threshold = 0.01 / (1.0 + c.protection_score * 5.0)
        if c.activation <= protected_threshold:
            removed.append(c)
    removed_ids = {id(c) for c in removed}
    net.clusters = [c for c in net.clusters if id(c) not in removed_ids]
    # 同步桶
    for c in removed:
        key = net._hash_to_bucket(c.centroid)
        if key in net.buckets:
            net.buckets[key] = [x for x in net.buckets[key] if x is not c]
    n_removed = n_before - net.n_clusters

    # ---- v6.2: 巩固锁定 (PKMζ 类比) ----
    # 存活的簇 ← 递增 consolidation_count (模拟多轮睡眠巩固)
    for c in net.clusters:
        if c.consolidation_count < theta.consolidation_lock_max:
            c.consolidation_count += 1
    n_locked = sum(1 for c in net.clusters if c.consolidation_count > 0)
    mean_lock = (sum(c.consolidation_count for c in net.clusters)
                 / max(net.n_clusters, 1))

    # v6.1: 候选集群统计
    n_candidates_alive = len(net._candidate_clusters)

    return {
        'n_replayed': n_replayed,
        'n_linked': n_separated,   # 模式分离对数 (兼容旧字段名)
        'n_removed': n_removed,
        'replay_boost': float(total_boost),
        'n_clusters': net.n_clusters,
        'n_candidates': len(candidates),
        'mean_activation': float(
            sum(c.activation for c in net.clusters) / max(net.n_clusters, 1)),
        # v6.1: 新统计
        'n_silent_candidates': n_candidates_alive,
        'mean_pnn': float(sum(c.pnn_level for c in net.clusters)
                         / max(net.n_clusters, 1)),
        'mean_protection': float(sum(c.protection_score for c in net.clusters)
                                / max(net.n_clusters, 1)),
        # v6.2: 巩固锁统计
        'n_locked_clusters': n_locked,
        'mean_consolidation_lock': float(mean_lock),
    }


# ============================================================
# v6.3: NREM/REM 双相睡眠巩固
# ============================================================

def sleep_consolidation_nrem(net: ClusterNetwork, theta: Theta,
                              semantic_memory=None,
                              n_replay_cycles: int = 2,
                              downscale_rate: float = 0.03) -> dict:
    """NREM 慢波睡眠巩固 — 三波耦合回放 + 突触尺度缩小 + 类淋巴清除.

    神经科学基础 (Diekelmann & Born 2010; Tononi & Cirelli 2014):
      → 皮层慢振荡 (~0.75 Hz): 调度全局 UP/DOWN 状态
      → 丘脑纺锤波 (12-15 Hz): 嵌套在慢振荡, 为可塑性创造时间窗
      → 海马尖波涟漪 (SWR, ~100-200 Hz): 记忆痕迹压缩重放
      → 突触尺度缩小: 等比降低所有突触权重, 保留相对强度
      → 类淋巴清除: 深度睡眠中清除代谢废物 (低质量连接)

    NREM 的独特功能:
      - 陈述性记忆巩固 (海马 → 皮层转移)
      - 突触稳态恢复 (释放新一轮学习容量)
      - 代谢废物清除 (β-淀粉样蛋白/tau清除)

    Args:
        net: ClusterNetwork
        theta: 参数配置
        semantic_memory: 语义记忆 (可选, 用于海马→皮层转移)
        n_replay_cycles: 回放轮数 (NREM 应 > 普通 sleep_replay)
        downscale_rate: 突触尺度缩小率 [0.01, 0.10]

    Returns:
        stats dict with nrem-specific metrics
    """
    if net.n_clusters == 0:
        return {'n_replayed': 0, 'n_downscaled': 0, 'n_cleared': 0,
                'n_transferred': 0, 'mean_downscale': 0.0,
                'n_clusters_after': 0, 'nrem_phase': 'NREM'}

    # ---- Step 1: 选择重放候选 (高激活 + 近期活跃) ----
    mean_act = sum(c.activation for c in net.clusters) / net.n_clusters
    replay_threshold = max(0.02, mean_act * 0.3)
    candidates = [c for c in net.clusters if c.activation > replay_threshold]

    if not candidates:
        sorted_clusters = sorted(net.clusters, key=lambda c: c.activation,
                                 reverse=True)
        n_top = max(5, net.n_clusters // 5)
        candidates = sorted_clusters[:n_top]

    # ---- Step 2: 三波耦合重放 ----
    # 模拟: 慢振荡调度 → 纺锤波窗口 → SWR 压缩重放
    n_replayed = 0

    for cycle in range(n_replay_cycles):
        cycle_decay = 0.85 ** cycle  # 逐轮衰减强度

        for c in candidates:
            salience = 0.3 + 0.7 * c.activation
            # NREM 回放: 更结构化, 使用跨模态部分提示
            n_replays = int(1 + salience * 1.5)  # 1-2 次/簇/轮

            centroid = c.centroid
            for ri in range(n_replays):
                partial = np.zeros_like(centroid)
                mode_idx = (n_replayed + ri) % 4
                if mode_idx == 0:
                    # Text→Visual: 只保留文本 [0:64]
                    partial[0:64] = centroid[0:64]
                elif mode_idx == 1:
                    # Visual→Text: 只保留视觉 [64:372]
                    partial[64:372] = centroid[64:372]
                elif mode_idx == 2:
                    # Audio: 只保留听觉 [372:468]
                    partial[372:468] = centroid[372:468]
                else:
                    # 随机 50% 维度
                    keep = np.random.random(len(centroid)) > 0.5
                    partial[keep] = centroid[keep]

                net.recall(partial)
                n_replayed += 1

            # 温和激活增强 (远小于清醒时的 0.1)
            boost = 0.003 * salience * cycle_decay
            c.activation = min(1.0, c.activation + boost)

    # ---- Step 3: 突触尺度缩小 (Synaptic Downscaling) ----
    # 等比降低所有簇的 activation, 保留相对强度
    # v6.2 保护信号 + 巩固锁 → 调制缩小率
    n_downscaled = 0
    total_downscale = 0.0
    for c in net.clusters:
        # 保护信号降低缩小率: 高保护 → 小缩小
        protection_mod = 1.0 / (1.0 + c.protection_score * 3.0)
        # 巩固锁降低缩小率: 高锁 → 小缩小
        lock_mod = 1.0 / (1.0 + c.consolidation_count *
                         theta.consolidation_lock_factor * 0.5)
        # 综合缩小率
        effective_downscale = downscale_rate * protection_mod * lock_mod

        old_act = c.activation
        c.activation *= (1.0 - effective_downscale)
        # 也温和缩小 tag (不超过 activation 下降)
        c.tag *= (1.0 - effective_downscale * 0.5)
        # 持续性衰减 (NREM 中持续性的衰减加速)
        c.activation_persistence *= (1.0 - effective_downscale * 2.0)
        if c.activation_persistence < 0.001:
            c.activation_persistence = 0.0

        if old_act - c.activation > 0.0001:
            n_downscaled += 1
            total_downscale += (old_act - c.activation)

    mean_downscale = total_downscale / max(n_downscaled, 1)

    # ---- Step 4: 海马→皮层转移 ----
    n_transferred = 0
    if semantic_memory is not None and hasattr(semantic_memory, 'consolidate_from_episodic'):
        try:
            transfer_stats = semantic_memory.consolidate_from_episodic(
                episodic_net=net,
                n_top=min(30, net.n_clusters // 3),
                min_activation=0.05,
            )
            n_transferred = transfer_stats.get('n_processed', 0)
        except Exception:
            pass

    # ---- Step 5: 类淋巴清除 (glymphatic clearance) ----
    # 深度 NREM (N3) 中脑间质空间增大 ~60% → CSF 流入增加
    # 清除 activation 极低且无保护的"代谢废物"
    glymphatic_threshold = theta.glymphatic_clear_rate
    n_cleared = 0
    n_before_clear = net.n_clusters
    to_remove = []
    for c in net.clusters:
        # 清除条件: activation < 阈值 AND 保护低
        if (c.activation < glymphatic_threshold and
            c.protection_score < 0.1 and
            c.pnn_level < 0.2):
            to_remove.append(c)

    if to_remove:
        removed_ids = {id(c) for c in to_remove}
        net.clusters = [c for c in net.clusters if id(c) not in removed_ids]
        # 清理桶
        for c in to_remove:
            key = net._hash_to_bucket(c.centroid)
            if key in net.buckets:
                net.buckets[key] = [x for x in net.buckets[key] if x is not c]
        n_cleared = n_before_clear - net.n_clusters

    # ---- Step 6: NREM 巩固锁递增 ----
    for c in net.clusters:
        if c.consolidation_count < theta.consolidation_lock_max:
            c.consolidation_count += 1

    return {
        'n_replayed': n_replayed,
        'n_downscaled': n_downscaled,
        'n_cleared': n_cleared,
        'n_transferred': n_transferred,
        'mean_downscale': float(mean_downscale),
        'n_clusters_after': net.n_clusters,
        'n_locked': sum(1 for c in net.clusters if c.consolidation_count > 0),
        'mean_consolidation_lock': float(
            sum(c.consolidation_count for c in net.clusters)
            / max(net.n_clusters, 1)),
        'nrem_phase': 'NREM',
    }


def sleep_consolidation_rem(net: ClusterNetwork, theta: Theta,
                             amygdala=None, striatum=None,
                             emotional_processing_strength: float = 0.3) -> dict:
    """REM 睡眠巩固 — 情绪去刺痛 + 跨域联想 + 程序性记忆整合.

    神经科学基础 (Walker & van der Helm 2009; Hobson 2009):
      → 去甲肾上腺素 ≈ 0: 独特的低应激神经化学环境
      → 乙酰胆碱 ↑↑: 促进皮层可塑性和联想
      → 杏仁核环路重新激活: 保留记忆内容, 消解情绪"刺痛"
      → 前额叶执行控制放松: 允许远程记忆的自由组合
      → 程序性记忆巩固: 运动技能、习惯的 REM 依赖强化

    REM 的独特功能:
      - 情绪记忆去刺痛 (PTSD 相关)
      - 跨域创造性联想 (远距离记忆的自由组合)
      - 程序性记忆巩固 (运动/技能)
      - 抽象规律提取 (洞察)

    Args:
        net: ClusterNetwork
        theta: 参数配置
        amygdala: 杏仁核 (可选, 用于情绪簇识别与去刺痛)
        striatum: 纹状体 (可选, 用于程序性记忆巩固)
        emotional_processing_strength: 情绪去刺痛强度 [0, 1]

    Returns:
        stats dict with rem-specific metrics
    """
    if net.n_clusters == 0:
        return {'n_emotional_processed': 0, 'n_cross_linked': 0,
                'n_habit_boosted': 0, 'emotional_shift_mean': 0.0,
                'n_clusters_after': 0, 'rem_phase': 'REM'}

    # ---- Step 1: 低 NE 环境 (模拟 REM 独特神经化学) ----
    # 临时降低学习率 (NE→0 → 可塑性模式切换)
    original_lr_mod = net.learn_rate_modifier
    net.learn_rate_modifier = 0.3  # 低 NE → 降低新学习, 促进内部分析

    # ---- Step 2: 情绪去刺痛 (Emotional Depotentiation) ----
    # 找到高情感标记的簇 (通过 centroid 的情感通道推断)
    # centroid[64:72] 包含身体+情感快照 (v6.0 情境向量)
    # 或通过 amygdala 的词→情感映射识别
    n_emotional_processed = 0
    total_emotional_shift = 0.0

    for c in net.clusters:
        if c.activation < 0.02:
            continue

        # 检测情感强度: centroid 的情感段能量
        centroid = c.centroid
        if len(centroid) > 72:
            emotional_segment = centroid[64:72]
            emotional_intensity = float(np.linalg.norm(emotional_segment))
        else:
            emotional_intensity = 0.0

        # 高情感强度的簇 → 温和衰减(去刺痛)
        if emotional_intensity > 0.3:
            # 保留记忆内容 (centroid 方向不变)
            # 但降低情感关联强度
            decay_factor = 1.0 - (emotional_processing_strength
                                 * 0.05 * emotional_intensity)
            if len(centroid) > 72:
                old_emotion = centroid[64:72].copy()
                c.centroid[64:72] *= decay_factor
                shift = float(np.linalg.norm(old_emotion - c.centroid[64:72]))
            else:
                shift = 0.0

            # 降低情感簇的 activation (让它们不那么"灼热")
            c.activation *= (1.0 - emotional_processing_strength * 0.03)

            if shift > 0.0001:
                n_emotional_processed += 1
                total_emotional_shift += shift

    # ---- Step 3: 跨域创造性联想 (Random Cross-Linking) ----
    # REM 中前额叶执行控制暂时放松 → 远距离记忆自由组合
    # 从不同桶中随机选择簇对 → 温和拉近 (促进创造性的新颖关联)
    n_cross_linked = 0
    if net.n_clusters >= 4:
        # 选择中等活跃的簇 (不是最高也不是最低)
        sorted_by_act = sorted(net.clusters, key=lambda c: c.activation,
                              reverse=True)
        mid_start = max(2, net.n_clusters // 5)
        mid_end = max(mid_start + 1, 4 * net.n_clusters // 5)
        mid_clusters = sorted_by_act[mid_start:mid_end]

        if len(mid_clusters) >= 2:
            n_pairs = min(8, len(mid_clusters) // 2)
            for _ in range(n_pairs):
                i, j = np.random.choice(len(mid_clusters), size=2, replace=False)
                c_a, c_b = mid_clusters[i], mid_clusters[j]
                sim = _masked_cosine(c_a.centroid, c_b.centroid,
                                    np.ones(len(c_a.centroid), dtype=bool))

                # 只对远距离簇对 (低相似度) 做温和拉近
                if 0.1 < sim < 0.4:
                    # 温和的质心拉近 (创造关联但不合并)
                    link_strength = 0.002 * (0.4 - sim)
                    old_a = c_a.centroid.copy()
                    old_b = c_b.centroid.copy()

                    c_a.centroid = ((1.0 - link_strength) * c_a.centroid
                                   + link_strength * c_b.centroid)
                    c_b.centroid = ((1.0 - link_strength) * c_b.centroid
                                   + link_strength * c_a.centroid)

                    net._migrate_bucket(c_a, old_a)
                    net._migrate_bucket(c_b, old_b)
                    n_cross_linked += 1

    # ---- Step 4: 程序性记忆巩固 ----
    n_habit_boosted = 0
    if striatum is not None and hasattr(striatum, 'boost_habits'):
        try:
            n_habit_boosted = striatum.boost_habits(boost=0.01)
        except Exception:
            pass

    # ---- Step 5: 恢复学习率 ----
    net.learn_rate_modifier = original_lr_mod

    emotional_shift_mean = (total_emotional_shift /
                           max(n_emotional_processed, 1))

    return {
        'n_emotional_processed': n_emotional_processed,
        'n_cross_linked': n_cross_linked,
        'n_habit_boosted': n_habit_boosted,
        'emotional_shift_mean': float(emotional_shift_mean),
        'n_clusters_after': net.n_clusters,
        'rem_phase': 'REM',
    }


def dual_phase_sleep(net: ClusterNetwork, theta: Theta,
                     semantic_memory=None, amygdala=None, striatum=None,
                     nrem_duration_ratio: float = 0.65,
                     sleep_duration_steps: int = 30,
                     force: bool = False) -> dict:
    """v6.3: NREM/REM 双相睡眠完整周期.

    编排 NREM (慢波睡眠) 和 REM (快速眼动睡眠) 两个阶段,
    模拟真实睡眠结构: 前半夜 NREM 主导, 后半夜 REM 主导.

    NREM 阶段:
      - 三波耦合记忆重放 (海马 SWR + 丘脑纺锤波 + 皮层慢振荡)
      - 突触尺度缩小 (等比降低, 保留相对强度)
      - 海马→皮层记忆转移
      - 类淋巴废物清除

    REM 阶段:
      - 低 NE 环境 (去甲肾上腺素 ≈ 0)
      - 情绪记忆去刺痛
      - 跨域创造性联想
      - 程序性记忆巩固

    Args:
        net: ClusterNetwork
        theta: 参数配置
        semantic_memory: 语义记忆 (可选)
        amygdala: 杏仁核 (可选)
        striatum: 纹状体 (可选)
        nrem_duration_ratio: NREM 占睡眠比例 [0, 1]
        sleep_duration_steps: 总睡眠步数
        force: 即使网络弱小也强制执行

    Returns:
        combined_stats dict with all metrics
    """
    if net.n_clusters == 0:
        return {'nrem': {}, 'rem': {}, 'combined': {
            'total_replayed': 0, 'total_downscaled': 0,
            'total_cleared': 0, 'total_emotional': 0,
            'total_cross_linked': 0, 'clusters_before': 0,
            'clusters_after': 0, 'dual_phase_complete': False,
        }}

    clusters_before = net.n_clusters

    # ---- Phase 1: NREM 慢波睡眠 (前半夜主导) ----
    nrem_steps = max(5, int(sleep_duration_steps * nrem_duration_ratio))
    nrem_cycles = max(1, nrem_steps // 15)  # 每 ~15 步一个回放子周期

    nrem_stats = {'n_replayed': 0, 'n_downscaled': 0, 'n_cleared': 0,
                  'n_transferred': 0}
    for _ in range(nrem_cycles):
        stats = sleep_consolidation_nrem(
            net, theta, semantic_memory=semantic_memory,
            n_replay_cycles=1,
            downscale_rate=theta.synaptic_downscale_rate,
        )
        for k in nrem_stats:
            nrem_stats[k] += stats.get(k, 0)

    # ---- Phase 2: REM 睡眠 (后半夜主导) ----
    rem_steps = max(3, sleep_duration_steps - nrem_steps)
    rem_cycles = max(1, rem_steps // 10)  # 每 ~10 步一个 REM 子周期

    rem_stats = {'n_emotional_processed': 0, 'n_cross_linked': 0,
                 'n_habit_boosted': 0}
    for _ in range(rem_cycles):
        stats = sleep_consolidation_rem(
            net, theta, amygdala=amygdala, striatum=striatum,
            emotional_processing_strength=theta.rem_emotional_processing,
        )
        for k in rem_stats:
            rem_stats[k] += stats.get(k, 0)

    # ---- 综合统计 ----
    combined = {
        'total_replayed': nrem_stats['n_replayed'],
        'total_downscaled': nrem_stats['n_downscaled'],
        'total_cleared': nrem_stats['n_cleared'],
        'total_transferred': nrem_stats['n_transferred'],
        'total_emotional': rem_stats['n_emotional_processed'],
        'total_cross_linked': rem_stats['n_cross_linked'],
        'total_habit_boosted': rem_stats['n_habit_boosted'],
        'clusters_before': clusters_before,
        'clusters_after': net.n_clusters,
        'dual_phase_complete': True,
        'nrem_steps': nrem_steps,
        'rem_steps': rem_steps,
    }

    return {
        'nrem': nrem_stats,
        'rem': rem_stats,
        'combined': combined,
    }
