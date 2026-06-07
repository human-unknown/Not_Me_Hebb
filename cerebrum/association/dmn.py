"""
self_model.py —— 自我模型 (Default Mode Network / vmPFC 等价物)

Agent 对"我是谁"的 Hebb 表征。

核心原则:
- 每次说话都是一次"自我体验" → 存储为 Hebb 集群
- 相似体验聚类 → 形成稳定的"人格"锚点
- 生成回应时，自我锚点调制查询 → 保持一致性

脑区映射:
- Default Mode Network: 自我参照加工
- vmPFC:            基于价值的自我表征
- Posterior Cingulate: 自我相关信息的整合

神经学病症对应:
- 人格解体 (Depersonalization): 无自我模型 → 回应风格漂移
- 解离性身份障碍 (DID): 多个不连通的自我集群
"""

import numpy as np
from cns.data_types import D, Theta, Cluster
from cerebrum.limbic_system.hippocampus import ClusterNetwork, _masked_cosine


class SelfModel:
    """Agent 的自我模型 —— "我是谁"的 Hebb 表征。

    存储格式 (330-dim 集群):
      centroid[:64]   = response_vec     — 我说了什么
      centroid[64:72] = emotion_vec      — [valence, arousal, self_valence, self_arousal, coherence, 0,0,0]
      centroid[72:80] = body_snapshot    — [b0(社交), b1(安全), b2(能量), b3(新颖), b4, b5, b6, b7]
      centroid[80:144]= comprehension    — 我理解了什么
      centroid[144:208]= dialogue_snap   — 对话上下文快照 (压缩)
      centroid[208:]  = zeros            — padding
    """

    def __init__(self, max_clusters: int = 512):
        theta = Theta()
        theta.cluster_threshold = 0.55  # 中等阈值: 相似的自我体验合并
        theta.learn_rate_l0 = 0.08
        self.net = ClusterNetwork(theta)
        self.max_clusters = max_clusters

        # 自我锚点: 当前"我是什么样的人"的 EMA 表征
        self.anchor: np.ndarray = np.zeros(64, dtype=np.float32)
        self.anchor_emotion: np.ndarray = np.zeros(8, dtype=np.float32)
        self.anchor_body: np.ndarray = np.zeros(8, dtype=np.float32)

        # 追踪
        self.n_experiences: int = 0
        self.anchor_alpha: float = 0.15  # 锚点更新速率

    def add_experience(self,
                       response_vec: np.ndarray,       # (64,) 回应语义
                       valence: float,
                       arousal: float,
                       self_valence_ema: float,
                       self_arousal_ema: float,
                       self_coherence: float,
                       body_state,                      # BodyVector
                       comprehension_vec: np.ndarray,   # (64,) 理解内容
                       dialogue_ctx_vec: np.ndarray,    # (64,) 对话上下文
                       ):
        """存储一次自我体验。

        每次 Agent 对外说话 → 形成一条"我是谁"的证据。
        高频相似的体验聚合成人格锚点。
        """
        # ---- 构建体验向量 ----
        exp = np.zeros(D, dtype=np.float32)

        # 回应语义 [0:64]
        exp[:64] = response_vec[:64].astype(np.float32)

        # 情感快照 [64:72]
        exp[64] = valence
        exp[65] = arousal
        exp[66] = self_valence_ema
        exp[67] = self_arousal_ema
        exp[68] = self_coherence

        # 身体快照 [72:80]
        b = body_state.b
        for i in range(min(8, len(b))):
            exp[72 + i] = b[i]

        # 理解内容 [80:144]
        exp[80:144] = comprehension_vec[:64].astype(np.float32)

        # 对话上下文快照 [144:208]
        exp[144:208] = dialogue_ctx_vec[:64].astype(np.float32)

        # ---- 学习/创建集群 ----
        if self.net.n_clusters < self.max_clusters:
            # 容量内: 尝试 Hebb 学习 (相似体验合并)
            self.net.learn(exp)
        else:
            # 容量满: 替换最旧 + 最弱的集群
            oldest = min(self.net.clusters, key=lambda c: c.activation + c.age * 0.001)
            h = self.net.hash_features(exp)
            old_key = self.net._hash_to_bucket(oldest.centroid)
            if old_key in self.net.buckets:
                self.net.buckets[old_key] = [
                    c for c in self.net.buckets[old_key] if c is not oldest]
            oldest.centroid = h.copy()
            oldest.count = 1
            oldest.activation = 0.5
            new_key = self.net._hash_to_bucket(h)
            self.net.buckets.setdefault(new_key, []).append(oldest)

        self.n_experiences += 1

        # ---- 更新自我锚点 (EMA) ----
        # v3: 显著性 ∝ |F_body 变化| — 自由能波动越大 → 记忆越强
        # 情感极值 (|valence|/arousal) 是 F_body 波动的代理
        # 手写组合被替换: significance = f(F_body_change_proxy)
        F_body_change_proxy = abs(valence) * 0.5 + arousal * 0.5
        significance = 0.2 + F_body_change_proxy * 0.8  # [0.2, 1.0]
        effective_alpha = self.anchor_alpha * significance

        self.anchor = ((1 - effective_alpha) * self.anchor
                       + effective_alpha * response_vec[:64].astype(np.float32))
        self.anchor_emotion = ((1 - effective_alpha) * self.anchor_emotion
                               + effective_alpha * exp[64:72])
        self.anchor_body = ((1 - effective_alpha) * self.anchor_body
                            + effective_alpha * exp[72:80])

    def recall_self(self, context_vec: np.ndarray) -> np.ndarray | None:
        """在当前上下文中回忆'我是谁'。

        Args:
            context_vec: 当前上下文向量 (64-dim, 文本语义空间)

        Returns:
            最相关的自我体验的回应语义 (64,) 或 None
        """
        if self.net.n_clusters == 0:
            return None

        q = np.zeros(D, dtype=np.float32)
        q[:64] = context_vec[:64].astype(np.float32)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True

        c = self.net.recall(q, mask=mask)
        if c is not None:
            return c.centroid[:64].copy()
        return None

    def get_self_anchor(self) -> np.ndarray:
        """获取当前自我锚点。

        这是"我认为我是谁"的紧凑表征，用于:
        1. 调制回应生成 (保持人格一致性)
        2. 评估自我一致性 (我说的话符合我吗？)
        """
        return self.anchor.copy()

    def get_personality_summary(self) -> dict:
        """返回可解释的人格摘要"""
        if self.n_experiences == 0:
            return {'n_experiences': 0, 'dominant_valence': 0.0,
                    'dominant_arousal': 0.0, 'stability': 1.0}

        # 主导情感
        dom_valence = float(self.anchor_emotion[0])
        dom_arousal = float(self.anchor_emotion[1])

        # 稳定性: 集群越多 → 人格越稳定
        stability = min(1.0, self.net.n_clusters / 50.0)

        # 主导身体需求
        dom_body_need = float(np.argmax(
            self.anchor_body - np.array([0.7, 0.7, 0.0, 0.0, 0.3, 0.3, 0.3, 0.5])))

        return {
            'n_experiences': self.n_experiences,
            'n_clusters': self.net.n_clusters,
            'dominant_valence': dom_valence,
            'dominant_arousal': dom_arousal,
            'stability': stability,
            'dominant_body_need': dom_body_need,
        }

    # ---- v6.4: 自发回忆与走神链 ----

    def spontaneous_recall(self, agent_net=None) -> np.ndarray | None:
        """随机回忆一个自我体验 (v6.4)。

        模拟 DMN 在静息状态下的自发活动:
          从自身网络中随机选一个集群 → 返回其情感/体验质心

        Args:
            agent_net: Agent 的海马网络 (可选, 混合回忆用)

        Returns:
            回忆的质心向量 (D,) 或 None
        """
        import random

        # 优先从自身网络回忆 (自我相关记忆)
        if self.net.n_clusters > 0:
            top_n = min(10, self.net.n_clusters)
            top_self = sorted(self.net.clusters,
                            key=lambda c: c.activation, reverse=True)[:top_n]
            c = random.choice(top_self)
            # 重新激活该集群 (模拟回忆增强)
            c.activation = min(1.0, c.activation + 0.02)
            return c.centroid.copy()

        # 回退: 从 Agent 海马网络回忆
        if agent_net is not None and agent_net.n_clusters > 0:
            top_n = min(10, agent_net.n_clusters)
            top_agent = sorted(agent_net.clusters,
                             key=lambda c: c.activation, reverse=True)[:top_n]
            c = random.choice(top_agent)
            c.activation = min(1.0, c.activation + 0.02)
            return c.centroid.copy()

        return None

    def mind_wander_chain(self, agent_net=None,
                          chain_length: int = 3) -> list[np.ndarray]:
        """自由联想链 — 连续的自发回忆形成思维流 (v6.4)。

        每次回忆作为下一个查询的种子 → 模拟"一个想法引出另一个想法"。

        Args:
            agent_net: Agent 海马网络
            chain_length: 联想链长度

        Returns:
            回忆质心列表 [centroid, ...]
        """
        chain = []
        current_seed = None

        for _ in range(chain_length):
            if current_seed is not None and len(current_seed) >= 64:
                # 用上一个回忆作为查询种子
                recalled = self.net.recall(current_seed[:64])
                if recalled is not None:
                    chain.append(recalled.centroid.copy())
                    current_seed = recalled.centroid
                    continue

            # 无法链式 → 随机跳跃
            recalled = self.spontaneous_recall(agent_net=agent_net)
            if recalled is not None:
                chain.append(recalled)
                current_seed = recalled
            else:
                break

        return chain

    def get_state_for_save(self) -> dict:
        """可序列化状态 (用于持久化) — v6.4。"""
        return {
            'n_experiences': self.n_experiences,
            'anchor_alpha': self.anchor_alpha,
            'anchor': self.anchor[:64].tolist(),
            'anchor_emotion': self.anchor_emotion.tolist(),
            'anchor_body': self.anchor_body.tolist(),
        }

    def restore_from_save(self, data: dict):
        """从持久化数据恢复 — v6.4。"""
        if not data:
            return
        self.n_experiences = data.get('n_experiences', 0)
        self.anchor_alpha = data.get('anchor_alpha', 0.15)
        anchor = data.get('anchor', [])
        if len(anchor) >= 64:
            self.anchor[:64] = np.array(anchor[:64], dtype=np.float32)
        anchor_emotion = data.get('anchor_emotion', [])
        if len(anchor_emotion) >= 8:
            self.anchor_emotion = np.array(anchor_emotion[:8], dtype=np.float32)
        anchor_body = data.get('anchor_body', [])
        if len(anchor_body) >= 8:
            self.anchor_body = np.array(anchor_body[:8], dtype=np.float32)
