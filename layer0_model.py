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
from data_types import (
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

        # 自适应阈值: 部分查询 → 放宽要求
        active_ratio = float(np.sum(mask)) / max(len(mask), 1)
        eff_threshold = self.theta.cluster_threshold * (0.4 + 0.6 * active_ratio)
        if best_c is not None and best_sim >= eff_threshold:
            best_c.activation += 0.1
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
            lr = self.theta.learn_rate_l0
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
            return existing

        # 无匹配 → 创建新簇（新颖性检测）
        if len(self.clusters) < K:
            c = Cluster(centroid=h.copy())
            self.clusters.append(c)
            hash_key = self._hash_to_bucket(h)
            self.buckets.setdefault(hash_key, []).append(c)
            return c

        # 容量满 → 替换最旧簇
        oldest = min(self.clusters, key=lambda c: c.age)
        # 从旧桶移除 (用 id 比较避免 numpy array == 歧义)
        old_key = self._hash_to_bucket(oldest.centroid)
        if old_key in self.buckets:
            self.buckets[old_key] = [
                c for c in self.buckets[old_key] if c is not oldest]
        oldest.centroid = h.copy()
        oldest.count = 1
        oldest.age = 0
        oldest.activation = 0.0
        # 入新桶
        new_key = self._hash_to_bucket(h)
        self.buckets.setdefault(new_key, []).append(oldest)
        return oldest

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
