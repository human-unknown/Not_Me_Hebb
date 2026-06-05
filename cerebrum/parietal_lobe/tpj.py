"""
tpj.py — 颞顶联合区 (Temporoparietal Junction) [v5.6 实现: 心理理论与语用]

对应脑区: BA39 (角回) + BA40 (缘上回) + 颞上沟后部(pSTS)
所属层级: 大脑 → 顶叶 → TPJ

脑区标记: TPJ (temporoparietal junction) — 社会认知的核心枢纽

功能职责 (参考: 语言与大脑 §3.5 双语/聋人失语症, §4 偏侧化, §5.2 脑成像):
  - 心理理论 (Theory of Mind): 推断他人的信念、意图、情绪状态
  - 自我/他人区分: 区分自己的心理状态和他人的心理状态
  - 视角采择 (Perspective Taking): 从他人视角看待同一情境
  - 语用理解: 理解言外之意 (反讽、间接请求、隐喻)
  - 社会注意力: 共同注意 (joint attention)、注视线索

语言与TPJ的关系:
  - 语用语言处理: 理解"你真的太聪明了" (是表扬还是讽刺?) 取决于说话人身份+情境
  - 跨语言/跨文化: 多语者的TPJ参与语言切换和语用适应
  - 与Broca/Wernicke协作: TPJ提供社会语用上下文, 调制语言理解和产出

在 NotMe 中的集成:
  - 接收: Wernicke 理解向量 + 对话上下文 + 社会信任度
  - 输出: 语用丰富化的理解向量 (literal → enriched)
  - 协作: TPJ → Wernicke (语用调制理解) → AF → Broca (语用得体的回应)

训练来源:
  语料中约30%的句子有角色标记 "角色名：台词"
  这为无监督学习角色模型和意图推断提供了天然数据.

当前状态 (v5.6 初始实现):
  - 简化版心理理论: 基于说话人模型的意图推断
  - 视角采择: 从社会上下文中推导对方视角
  - 语用调制: 将字面意义 + 社会意图 → 丰富化理解

参考:
  - Saxe, R., & Kanwisher, N. (2003). People thinking about thinking people:
    The role of the temporo-parietal junction in "theory of mind".
    NeuroImage.
  - Frith, C. D., & Frith, U. (2006). The neural basis of mentalizing.
    Neuron.
  - Sperber, D., & Wilson, D. (1986). Relevance: Communication and
    Cognition. (关联理论 — 语用理解的理论基础)
"""

import numpy as np
from typing import Optional

from cns.data_types import D, Theta, Cluster
from cerebrum.limbic_system.hippocampus import ClusterNetwork, _masked_cosine


class TPJ:
    """颞顶联合区 — 心理理论与语用语言理解.

    三层功能:
      1. 心理理论 (ToM): 推断说话人的意图和信念
      2. 视角采择: 从对方视角看当前情境
      3. 语用丰富化: 字面意义 + 社会意图 → 真正的含义
    """

    def __init__(self):
        """初始化 TPJ."""
        # 说话人模型: {(speaker_name_hash → speaker_vector_64, trust, familiarity)}
        self.speaker_models: dict[int, tuple[np.ndarray, float, float]] = {}

        # 意图推断网络: 话语→意图的 Hebb 映射
        # centroid[:64]  = 话语向量 (what was said)
        # centroid[64:128] = 意图向量 (what was meant)
        # centroid[128:192] = 情境上下文 (when/where/to whom)
        intent_theta = Theta()
        intent_theta.cluster_threshold = 0.18
        intent_theta.learn_rate_l0 = 0.04
        self.intent_net = ClusterNetwork(intent_theta)
        self._n_intents: int = 0

        # 追踪
        self.n_inferences: int = 0

        # 自我模型: 用于自我/他人区分
        self.self_signature: np.ndarray = np.zeros(64, dtype=np.float32)

    # ================================================================
    # 说话人模型
    # ================================================================

    def register_speaker(self, speaker_name: str,
                        initial_trust: float = 0.5):
        """注册一个新说话人 (从语料中学习).

        Args:
            speaker_name: 说话人名称 (如 "樱", "我")
            initial_trust: 初始信任度
        """
        name_hash = hash(speaker_name) % (2**31)
        if name_hash not in self.speaker_models:
            # (speaker_vector, trust, familiarity)
            self.speaker_models[name_hash] = (
                np.zeros(64, dtype=np.float32),
                initial_trust,
                0.0,  # familiarity starts at 0
            )

    def learn_from_utterance(self, speaker_name: str,
                            utterance_vec: np.ndarray,
                            response_vec: np.ndarray = None):
        """从说话人的一句话中学习其说话人模型.

        通过 Hebb 累积, 同一说话人的话语向量逐渐形成
        该人物的"语言指纹" — 说话风格、用词偏好、情感模式.

        Args:
            speaker_name: 说话人名称
            utterance_vec: 话语语义向量 (64,)
            response_vec: 回应的语义向量 (64,), 可选
        """
        name_hash = hash(speaker_name) % (2**31)
        if name_hash not in self.speaker_models:
            self.register_speaker(speaker_name)

        speaker_vec, trust, familiarity = self.speaker_models[name_hash]

        # EMA 更新: 说话人向量 = 其所有话语的加权平均
        alpha = 0.1  # 更新率
        speaker_vec = (1.0 - alpha) * speaker_vec + alpha * utterance_vec[:64]
        familiarity = min(1.0, familiarity + 0.01)  # 越听越熟悉

        # 信任更新: 如果提供了回应 (对话继续), 信任微升
        if response_vec is not None:
            trust = min(1.0, trust + 0.005)

        self.speaker_models[name_hash] = (
            speaker_vec.astype(np.float32), trust, familiarity)

    def get_speaker_model(self, speaker_name: str) -> dict:
        """获取说话人模型.

        Args:
            speaker_name: 说话人名称

        Returns:
            {'speaker_vec': (64,), 'trust': float, 'familiarity': float}
            或 None (未知说话人)
        """
        name_hash = hash(speaker_name) % (2**31)
        if name_hash not in self.speaker_models:
            return None

        vec, trust, fam = self.speaker_models[name_hash]
        return {
            'speaker_vec': vec.copy(),
            'trust': float(trust),
            'familiarity': float(fam),
        }

    # ================================================================
    # 心理理论: 意图推断
    # ================================================================

    def infer_speaker_intent(self,
                            utterance_vec: np.ndarray,
                            speaker_name: str = None,
                            context_vec: np.ndarray = None,
                            ) -> tuple[np.ndarray, dict]:
        """推断说话人的真实意图.

        字面意义 (what was SAID) → 加上说话人模型 + 情境 → 真实意图 (what was MEANT).

        例如:
          SAID: "你真聪明" (literal: positive)
          Speaker: 竞争对手, 低信任
          Context: 我刚犯了一个明显错误
          → MEANT: 讽刺 (sarcasm)

        这个过程通过 Hebb 学习: 话语+说话人+情境 → 意图,
        从语料中角色对话的统计模式中涌现.

        Args:
            utterance_vec: 话语语义向量 (64,)
            speaker_name: 说话人名称 (可选)
            context_vec: 情境上下文向量 (64,), 可选

        Returns:
            (intent_vec, inference)
            - intent_vec: 推断的意图向量 (64,)
            - inference: 推断的元数据
        """
        self.n_inferences += 1

        # ---- Step 1: 说话人调制 ----
        speaker_bias = np.zeros(64, dtype=np.float32)
        trust_level = 0.5
        familiarity = 0.0

        if speaker_name is not None:
            speaker_info = self.get_speaker_model(speaker_name)
            if speaker_info is not None:
                speaker_bias = speaker_info['speaker_vec']
                trust_level = speaker_info['trust']
                familiarity = speaker_info['familiarity']

        # ---- Step 2: 字面 vs 情境一致性 ----
        literal_context_sim = 0.5  # neutral
        if context_vec is not None and np.linalg.norm(context_vec) > 1e-8:
            denom = (np.linalg.norm(utterance_vec[:64])
                     * np.linalg.norm(context_vec[:64]) + 1e-8)
            literal_context_sim = float(np.dot(
                utterance_vec[:64], context_vec[:64]) / denom)

        # ---- Step 3: Hebb 检索意图 ----
        intent_vec = utterance_vec[:64].copy()  # 默认: 字面 = 意图

        if self.intent_net.n_clusters > 0:
            q = np.zeros(D, dtype=np.float32)
            q[:64] = utterance_vec[:64].astype(np.float32)
            if context_vec is not None:
                q[128:192] = context_vec[:64].astype(np.float32)

            h = self.intent_net.hash_features(q)
            mask = np.zeros(D, dtype=bool)
            mask[:64] = True

            hash_key = self.intent_net._hash_to_bucket(h)
            bucket = self.intent_net.buckets.get(hash_key, [])

            if not bucket:
                bucket = self.intent_net.clusters[:min(500,
                    len(self.intent_net.clusters))]

            best_sim = -1.0
            best_c = None
            for c in bucket:
                sim = _masked_cosine(h, c.centroid, mask)
                if sim > best_sim:
                    best_sim = sim
                    best_c = c

            if (best_c is not None and
                best_sim >= self.intent_net.theta.cluster_threshold):
                intent_vec = best_c.centroid[64:128].copy().astype(np.float32)

        # ---- Step 4: 信任调制 ----
        # 高信任 → 意图接近字面 (take at face value)
        # 低信任 → 意图偏离字面 (skeptical interpretation)
        literal_weight = 0.3 + 0.7 * trust_level  # [0.3, 1.0]
        skeptic_weight = 1.0 - literal_weight

        # 怀疑成分: 偏离字面, 偏向说话人模型的"典型意图"
        intent_vec = (literal_weight * utterance_vec[:64]
                     + skeptic_weight * speaker_bias).astype(np.float32)

        # ---- Step 5: 反讽检测 ----
        # 字面正面 + 低信任 + 情境负面 → 可能是反讽
        sarcasm_score = 0.0
        if (literal_context_sim < 0.2  # 字面与情境不匹配
            and trust_level < 0.5        # 低信任
            and familiarity > 0.1):      # 足够熟悉才会反讽
            sarcasm_score = float(np.clip(
                (0.5 - trust_level) * (0.3 - literal_context_sim) * 5.0,
                0.0, 1.0))
            # 反讽: 翻转意图的情感方向
            if sarcasm_score > 0.5:
                intent_vec = -intent_vec * 0.5 + utterance_vec[:64] * 0.5

        inference = {
            'literal_context_similarity': float(literal_context_sim),
            'trust_level': float(trust_level),
            'familiarity': float(familiarity),
            'sarcasm_score': float(sarcasm_score),
            'literal_weight': float(literal_weight),
            'has_speaker_model': speaker_name is not None,
        }

        return intent_vec.astype(np.float32), inference

    # ================================================================
    # 视角采择
    # ================================================================

    def perspective_take(self,
                        situation_vec: np.ndarray,
                        other_model: dict = None
                        ) -> np.ndarray:
        """从他人视角看同一情境.

        "如果我是他/她, 我会看到/听到/理解什么?"

        Args:
            situation_vec: 当前情境的语义向量
            other_model: 他人的说话人模型 (来自 get_speaker_model())

        Returns:
            (64,) 从他人视角的情境向量
        """
        if other_model is None:
            return situation_vec[:64].copy().astype(np.float32)

        # 视角采择 = 情境 + 他人的典型关注偏向
        speaker_vec = other_model.get('speaker_vec',
                                      np.zeros(64, dtype=np.float32))

        # 他人视角: 情境被说话人模型调制
        # 不同人关注同一情境的不同方面
        other_view = (0.6 * situation_vec[:64]
                     + 0.4 * speaker_vec).astype(np.float32)

        # 归一化
        norm = np.linalg.norm(other_view)
        if norm > 1e-8:
            other_view = other_view / norm

        return other_view

    # ================================================================
    # 语用丰富化
    # ================================================================

    def pragmatic_enrichment(self,
                            literal_comprehension: np.ndarray,
                            speaker_intent: np.ndarray,
                            social_context=None,
                            ) -> np.ndarray:
        """语用丰富化: 字面理解 + 说话人意图 → 真正的理解.

        组合: 字面语义 (Wernicke) + 意图推断 (TPJ) + 社会情境.
        这是 Wernicke 理解向量的"升级版" — 更接近人类实际理解的内容.

        权重分配 (精度驱动):
          - 信任高 → 字面权重高 (对方说的话可信)
          - 熟悉度高 → 意图权重高 (我更了解对方的"言外之意")
          - 无说话人模型 → 字面权重 0.7 (保守: 更多依赖字面)

        Args:
            literal_comprehension: Wernicke字面理解 (64,)
            speaker_intent: TPJ意图推断 (64,)
            social_context: SocialContext实例 (可选, 用于信任/熟悉度)

        Returns:
            (64,) 语用丰富化的理解向量
        """
        # 默认权重
        literal_w = 0.6
        intent_w = 0.4

        # 从社会上下文调整权重
        if social_context is not None:
            trust = getattr(social_context, 'trust_level', 0.5)
            n_interactions = getattr(social_context, 'n_interactions', 0)
            familiarity = min(1.0, n_interactions / 20.0)

            # 信任 + 熟悉 → 更多依赖字面
            # 低信任 + 熟悉 → 更多依赖意图推断
            literal_w = 0.3 + trust * 0.4
            intent_w = 1.0 - literal_w

        # 加权组合
        enriched = (literal_w * literal_comprehension[:64]
                   + intent_w * speaker_intent[:64]).astype(np.float32)

        # 保持原始规范
        orig_norm = np.linalg.norm(literal_comprehension[:64])
        enriched_norm = np.linalg.norm(enriched)
        if enriched_norm > 1e-8 and orig_norm > 1e-8:
            enriched = enriched / enriched_norm * orig_norm

        return enriched

    # ================================================================
    # 训练: 从语料学习意图推断
    # ================================================================

    def train_from_corpus(self, sentences: list[str],
                         n_samples: int = 3000,
                         verbose: bool = True):
        """从角色对话语料训练意图推断 Hebb 网络.

        语料格式: "角色名：台词" (约30%的句子有此格式)
        训练:
          1. 提取角色名 → 建立说话人模型
          2. 每句话 = (话语, 说话人, 上下文) → 学习意图映射
          3. 同一角色的连续台词 → 角色一致性学习

        Args:
            sentences: 句子列表 (含角色标记)
            n_samples: 采样数
            verbose: 是否打印进度
        """
        import jieba

        if verbose:
            print(f"  TPJ: training from {min(n_samples, len(sentences))} "
                  f"sentences...")

        # 提取含角色标记的句子
        import re
        char_pattern = re.compile(r'^([一-鿿\w/·]{1,6})[：:]\s*(.+)')

        char_sentences: dict[str, list[str]] = {}  # {角色名: [台词, ...]}
        all_scored = []

        for sent in sentences[:n_samples]:
            m = char_pattern.match(sent)
            if m:
                char_name = m.group(1)
                content = m.group(2)
                if len(content) >= 2:
                    if char_name not in char_sentences:
                        char_sentences[char_name] = []
                        self.register_speaker(char_name)
                    char_sentences[char_name].append(content)
                    all_scored.append((char_name, content, len(content)))

        if verbose:
            n_chars = len(char_sentences)
            n_lines = len(all_scored)
            print(f"  TPJ: {n_chars} characters, {n_lines} labeled lines")

        # 为每个角色建立说话人模型
        for char_name, lines in char_sentences.items():
            if len(lines) >= 3:
                # 用角色台词学习意图推断
                for i in range(len(lines) - 1):
                    utterance = lines[i]
                    response = lines[i + 1]  # 下一句作为"意图的结果"

                    # 简化学习: 在同一角色连续对话中,
                    # 前一句→后一句 映射 近似 "话语→意图"
                    self.learn_intent(
                        utterance_vec=self._encode_text_proxy(utterance),
                        intent_vec=self._encode_text_proxy(response),
                        context_vec=None,
                        speaker_name=char_name,
                    )

                # 更新说话人模型
                line_vecs = [self._encode_text_proxy(l) for l in lines[-10:]]
                avg_vec = np.mean(line_vecs, axis=0)
                self.learn_from_utterance(char_name, avg_vec)

        self._n_intents = self.intent_net.n_clusters

        if verbose:
            print(f"  TPJ: {self._n_intents} intent clusters, "
                  f"{len(char_sentences)} speaker models")

    def learn_intent(self, utterance_vec: np.ndarray,
                    intent_vec: np.ndarray,
                    context_vec: np.ndarray = None,
                    speaker_name: str = None):
        """学习单条话语→意图映射.

        Args:
            utterance_vec: 话语语义向量 (64,)
            intent_vec: 真实意图向量 (64,) — 从后续对话或角色行为推断
            context_vec: 情境上下文 (64,) 可选
            speaker_name: 说话人名称 可选
        """
        pattern = np.zeros(D, dtype=np.float32)
        pattern[:64] = utterance_vec[:64].astype(np.float32)
        pattern[64:128] = intent_vec[:64].astype(np.float32)
        if context_vec is not None:
            pattern[128:192] = context_vec[:64].astype(np.float32)

        orig_lr = self.intent_net.theta.learn_rate_l0
        self.intent_net.theta.learn_rate_l0 = 0.06
        self.intent_net.learn(pattern)
        self.intent_net.theta.learn_rate_l0 = orig_lr

        # 同时更新说话人模型
        if speaker_name is not None:
            self.learn_from_utterance(speaker_name, utterance_vec)

    # ================================================================
    # 工具
    # ================================================================

    @staticmethod
    def _encode_text_proxy(text: str) -> np.ndarray:
        """文本编码代理 (无 TextEnvironment 时使用).

        使用简单词向量平均作为语义代理.
        """
        import jieba
        words = [w for w in jieba.lcut(text) if len(w.strip()) >= 1]
        if not words:
            return np.zeros(64, dtype=np.float32)

        # 用词的hash做简单向量化
        vec = np.zeros(64, dtype=np.float32)
        for i, w in enumerate(words[:10]):
            h = hash(w) % (2**31)
            for j in range(8):
                vec[(i * 8 + j) % 64] += ((h >> (j * 8)) & 0xFF) / 255.0

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec.astype(np.float32)

    # ================================================================
    # 诊断
    # ================================================================

    def get_state(self) -> dict:
        """返回当前状态 (供 dashboard 使用)."""
        return {
            'n_speaker_models': len(self.speaker_models),
            'n_intent_clusters': self.intent_net.n_clusters,
            'n_inferences': self.n_inferences,
        }

    def __repr__(self) -> str:
        return (f"TPJ(speakers={len(self.speaker_models)}, "
                f"intents={self.intent_net.n_clusters})")
