"""
it_cortex.py — 下颞皮层 (Inferotemporal Cortex, IT) [v5.0]

对应脑区: BA20, BA21
所属层级: 大脑 → 颞叶 → IT 皮层

v5.0 职责:
  - 腹侧通路终端 — Hebb 物体类别学习
  - 位置/大小不变性 — 全局池化
  - 反馈预测到 V4 — 物体→特征预期 (闭合律基础)
  - 简洁律: F_cognitive 自然偏好少数激活簇

参考:
  - 中枢神经系统视觉通路.md §5.2 (IT 皮层)
  - Logothetis & Sheinberg (1996). Visual object recognition.
  - Tanaka (1996). Inferotemporal cortex and object vision.
"""

import numpy as np
from typing import Optional


class ITCortex:
    """IT 皮层 — Hebb 物体表征 + 自上而下预测 (v5.0).

    输入: V4 汇合表征 (convergence)
    输出: 物体簇激活 + 向 V4 的预测

    机制:
      1. Hebb 簇: 统计共现形成的物体类别表征
      2. 匹配: 余弦相似度 × 激活度 → 最佳簇
      3. 学习: 匹配→更新 (LTP), 无匹配→创建新簇
      4. 衰减: 未使用的簇逐渐失活 (LTD)
      5. 反馈: IT 预测激活簇的质心 → V4 (闭合律)
    """

    def __init__(self, input_dim: int = 96, max_clusters: int = 64,
                 cluster_dim: int = 32):
        self.input_dim = input_dim
        self.max_clusters = max_clusters
        self.cluster_dim = cluster_dim

        # ---- Hebb 物体簇 ----
        self.n_clusters: int = 0
        self.clusters: list = []  # list of dict: {centroid, activation, count}

        # ---- 簇匹配阈值 ----
        self.threshold: float = 0.6

        # ---- 学习率 + 衰减 ----
        self.lr: float = 0.01
        self.decay: float = 0.001

        # ---- 预测误差 ----
        self.PE: Optional[np.ndarray] = None

    def feedforward(self, v4_output: dict) -> dict:
        """V4 汇合 → IT 物体簇激活.

        Args:
            v4_output: V4.feedforward() 输出, 使用 'convergence' 键

        Returns:
            dict with 'object_code' (compact cluster activation),
                     'dominant_cluster' (int or None),
                     'activation' (float)
        """
        convergence = v4_output.get('convergence',
                                     np.zeros(self.input_dim, dtype=np.float32))
        x = self._pad_or_trunc(convergence, self.input_dim)

        # Hebb 簇匹配
        best_idx, best_sim, activations = self._match(x)

        # 紧凑物体编码: top-k 簇激活
        object_code = np.zeros(self.cluster_dim, dtype=np.float32)
        if activations and len(activations) > 0:
            top_k = min(self.cluster_dim, len(activations))
            sorted_act = sorted(activations, key=lambda a: a[1], reverse=True)
            for i, (cidx, act) in enumerate(sorted_act[:top_k]):
                if cidx < len(self.clusters):
                    object_code[i] = act * float(np.mean(np.abs(
                        self.clusters[cidx]['centroid'])))

        return {
            'object_code': object_code,
            'dominant_cluster': best_idx,
            'activation': best_sim,
        }

    def predict_to_V4(self, current_output: dict) -> np.ndarray:
        """IT → V4: 物体预测 — "如果这是 X, V4 应该看到 Y".

        这是闭合律的关键: IT 预测激活簇的质心 → V4.
        如果 V4 看到的特征与质心匹配, PE 降低 → "闭合感".
        """
        dominant = current_output.get('dominant_cluster')
        if (dominant is not None and
            dominant < len(self.clusters) and
            len(self.clusters[dominant]['centroid']) > 0):
            # 预测 = 最佳匹配簇的质心 (该物体应有的特征)
            prediction = self.clusters[dominant]['centroid'].copy()
            padded = np.zeros(self.input_dim, dtype=np.float32)
            padded[:len(prediction)] = prediction
            return padded
        return np.zeros(self.input_dim, dtype=np.float32)

    def learn(self, v4_convergence: np.ndarray):
        """Hebb 学习: 创建新簇或更新现有簇.

        LTP 类比: 匹配→更新质心 (一起放电 → 连接增强)
        LTD 类比: 弱簇衰减 (不常用 → 连接削弱)
        """
        x = self._pad_or_trunc(v4_convergence, self.input_dim)

        best_idx, best_sim, _ = self._match(x)

        if best_sim >= self.threshold and best_idx is not None:
            # 更新现有簇 (LTP)
            c = self.clusters[best_idx]
            centroid_len = len(c['centroid'])
            c['centroid'] += self.lr * (x[:centroid_len] - c['centroid'])
            c['activation'] = 0.9 * c['activation'] + 0.1 * best_sim
            c['count'] += 1
        elif self.n_clusters < self.max_clusters:
            # 创建新簇
            self.clusters.append({
                'centroid': x[:self.cluster_dim].copy().astype(np.float32),
                'activation': best_sim,
                'count': 1,
            })
            self.n_clusters += 1

        # 簇衰减 (LTD): 所有簇缓慢衰减
        for c in self.clusters:
            c['activation'] *= (1.0 - self.decay)

    def compute_prediction_error(self, current_output: dict) -> np.ndarray:
        """IT 预测误差: 当前物体编码与最佳簇质心的差异.

        高 PE → "不认识的物体" → 可能触发学习 (新簇) 或探索.
        """
        object_code = current_output.get('object_code',
                        np.zeros(self.cluster_dim, dtype=np.float32))
        dominant = current_output.get('dominant_cluster')
        if dominant is not None and dominant < len(self.clusters):
            centroid = self.clusters[dominant]['centroid']
            clen = min(len(centroid), len(object_code))
            self.PE = np.abs(object_code[:clen] - centroid[:clen])
        else:
            self.PE = np.abs(object_code)  # 无匹配 → 高 PE
        return self.PE

    def _match(self, x: np.ndarray) -> tuple:
        """簇匹配: 余弦相似度 × 激活度.

        Returns:
            (best_idx, best_sim, [(cluster_idx, combined_score), ...])
        """
        if self.n_clusters == 0:
            return None, 0.0, []
        best_idx = None
        best_sim = -1.0
        activations = []
        for i, c in enumerate(self.clusters):
            cen = c['centroid']
            clen = min(len(cen), len(x))
            if clen == 0:
                continue
            sim = float(np.dot(x[:clen], cen[:clen]) /
                       (np.linalg.norm(x[:clen]) * np.linalg.norm(cen[:clen]) + 1e-8))
            combined = sim * (0.5 + 0.5 * c['activation'])
            activations.append((i, combined))
            if combined > best_sim:
                best_sim = combined
                best_idx = i
        return best_idx, best_sim, activations

    def _pad_or_trunc(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        if len(vec) >= target_len:
            return vec[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(vec)] = vec
        return out
