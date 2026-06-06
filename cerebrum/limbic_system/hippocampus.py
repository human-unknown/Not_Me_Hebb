"""
layer0_model.py —— L0 生成模型 + ClusterNetwork
自由能原理智能体 — M1 单智能体生存

功能：
- 隐状态 z 的初始化
- 从 z 预测感知输入 s (线性解码)
- ClusterNetwork: 特征哈希 + 侧抑制回忆 + Hebb-like 学习
- 睡眠巩固周期
"""

from typing import Optional
import numpy as np
from cns.data_types import (
    D, H, K, Cluster, Theta,
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
        # 当前身体/情感状态 vs 编码时的状态
        if best_c is not None and np.any(self._context_vector):
            # 从质心中提取身体快照段 (centroid[64:72] 通常存身体状态)
            if best_c.centroid.shape[0] > 72:
                encoded_ctx = best_c.centroid[64:72]
                if np.any(np.abs(encoded_ctx) > 0.01):
                    ctx_sim = _masked_cosine(
                        self._context_vector[:8],
                        encoded_ctx[:8],
                        np.ones(8, dtype=bool))
                    # 情境不匹配 → 降低有效相似度
                    # ctx_sim in [-1, 1] → mapped to [0.3, 1.0]
                    ctx_factor = 0.3 + 0.7 * max(0.0, ctx_sim)
                    best_sim *= ctx_factor

        # 自适应阈值: 部分查询 → 放宽要求
        active_ratio = float(np.sum(mask)) / max(len(mask), 1)
        eff_threshold = self.theta.cluster_threshold * (0.4 + 0.6 * active_ratio)
        if best_c is not None and best_sim >= eff_threshold:
            best_c.activation = min(1.0, best_c.activation + 0.1)
            best_c.age += 1
            return best_c
        return None

    # ----- Hebb 扩散 — Broca 区等价物 -----
    def diffuse(self, start_cluster, steps: int = 3,
                top_k: int = 5, sim_threshold: float = 0.15) -> tuple:
        """沿 Hebb 边随机游走 — 返回路径上所有质心。

        Returns:
            (path_centroids, path_indices)
            path_centroids: [start[:64], step1[:64], ..., end[:64]]
            path_indices:   [start_idx, step1_idx, ..., end_idx]
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
            dot = C @ cur_vec
            norms = np.linalg.norm(C, axis=1)
            sims = dot / (norms * norms[current_idx] + 1e-8)
            sims[current_idx] = -1.0

            top_indices = np.argsort(sims)[-top_k:]
            weights = sims[top_indices]
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

    # ----- 学习 (learn) -----
    def learn(self, s: np.ndarray) -> Cluster:
        """学习新感知模式：匹配到则更新，否则创建新簇

        策略：
        1. 若 recall 匹配 → 以 learn_rate 向输入更新 centroid
        2. 若未匹配且簇数 < K → 创建新簇
        3. 若未匹配且簇数 >= K → 替换 age 最大的最旧簇

        Returns:
            被创建或更新的 Cluster
        """
        h = self.hash_features(s)
        existing = self.recall(s)

        if existing is not None:
            # Hebb 学习: Δw = lr * activation * (input - w)
            # 匹配越强 (activation 越高) → 学习率越大 → 更强的 Hebb 强化
            # 这比纯 EMA 更接近生物 Hebb 规则: fire together → wire together
            old_key = self._hash_to_bucket(existing.centroid)
            lr = self.theta.learn_rate_l0 * self.learn_rate_modifier  # v5.5: VTA RPE 调制
            # 激活调制: 更强的匹配 → 更强的学习 (Hebb 机制)
            activation_factor = 0.3 + 0.7 * existing.activation  # [0.3, 1.0]
            effective_lr = lr * activation_factor

            existing.centroid = (1 - effective_lr) * existing.centroid + effective_lr * h
            existing.count += 1
            # 如果 centroid 变化导致桶改变，迁移
            new_key = self._hash_to_bucket(existing.centroid)
            if new_key != old_key and old_key in self.buckets:
                self.buckets[old_key] = [
                    c for c in self.buckets[old_key] if c is not existing]
                self.buckets.setdefault(new_key, []).append(existing)

            # ---- v6.0 倒摄干扰: 新学习轻微衰减桶内高相似旧簇 ----
            ri_bucket = self.buckets.get(new_key, [])
            for other_c in ri_bucket:
                if other_c is not existing:
                    sim_to_new = _masked_cosine(h, other_c.centroid,
                                              np.ones(len(h), dtype=bool))
                    if sim_to_new > 0.75:
                        other_c.activation *= 0.97  # 被"覆盖"的旧记忆衰减3%

            return existing

        # 无匹配 → 创建新簇（新颖性检测）
        # ---- v6.0 前摄干扰: 桶内相似簇过多 → 新簇质心被"稀释" ----
        hash_key = self._hash_to_bucket(h)
        bucket = self.buckets.get(hash_key, [])
        pi_n_similar = sum(1 for c in bucket
                          if _masked_cosine(h, c.centroid,
                                          np.ones(len(h), dtype=bool)) > 0.65)
        # 相似簇越多 → 前摄干扰越强 → 新记忆形成越弱
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
            # 容量满 → 替换最旧簇
            oldest = min(self.clusters, key=lambda c: c.age)
            # 从旧桶移除 (用 id 比较避免 numpy array == 歧义)
            old_key = self._hash_to_bucket(oldest.centroid)
            if old_key in self.buckets:
                self.buckets[old_key] = [
                    c for c in self.buckets[old_key] if c is not oldest]
            oldest.centroid = centroid_new
            oldest.count = 1
            oldest.age = 0
            oldest.activation = 0.0
            # 入新桶
            hash_key = self._hash_to_bucket(centroid_new)
            self.buckets.setdefault(hash_key, []).append(oldest)
            newly_created = oldest

        # ---- v6.0 倒摄干扰: 新簇衰减桶内高相似旧簇 ----
        ri_bucket = self.buckets.get(hash_key, [])
        for other_c in ri_bucket:
            if other_c is not newly_created:
                sim_to_new = _masked_cosine(h, other_c.centroid,
                                          np.ones(len(h), dtype=bool))
                if sim_to_new > 0.75:
                    other_c.activation *= 0.97

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
        """衰减所有簇的激活值
        未激活的簇随时间 activation → 0（自然遗忘）
        """
        for c in self.clusters:
            c.activation *= (1 - self.theta.decay_rate)
            c.age += 1

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

        # 更新质心矩阵
        if cycle < n_replay_cycles - 1:
            C = np.stack([c.centroid for c in net.clusters])

    # ---- Step 3: 衰减 (自然遗忘) ----
    net.decay()

    # ---- Step 4: 清理死簇 ----
    n_before = net.n_clusters
    removed = [c for c in net.clusters if c.activation <= 0.01]
    net.clusters = [c for c in net.clusters if c.activation > 0.01]
    # 同步桶
    for c in removed:
        key = net._hash_to_bucket(c.centroid)
        if key in net.buckets:
            net.buckets[key] = [x for x in net.buckets[key] if x is not c]
    n_removed = n_before - net.n_clusters

    return {
        'n_replayed': n_replayed,
        'n_linked': n_separated,   # 模式分离对数 (兼容旧字段名)
        'n_removed': n_removed,
        'replay_boost': float(total_boost),
        'n_clusters': net.n_clusters,
        'n_candidates': len(candidates),
        'mean_activation': float(
            sum(c.activation for c in net.clusters) / max(net.n_clusters, 1)),
    }
