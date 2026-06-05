"""
sentiment.py —— 中文情感强度检测 + Hebb 情感词汇学习 (v2: 无手标词典)

核心原则 (无现象学):
- 不预设哪些词是"正面"或"负面"
- 情感效价 = Agent 自身 F_body 变化 (内感受), 不是文本标签
- 情感唤醒 = 文本客观特征: 标点密度、重复模式、长度、强调标记
- HebbEmotionalLexicon: 从互动中学习词→情感关联 (fire together, wire together)

流:
1. 人类输入 → analyze_sentiment() → arousal/intensity (仅客观特征)
2. 人类输入 → EmotionalLexicon.predict() → learned_emotional_bias (初始 0)
3. Agent 处理输入后 → F_body 变化 ∆F
4. ∆F → EmotionalLexicon.learn(input_words, ∆F) → 关联被 Hebb 强化
5. 下一轮, 含相似词的输入 → lexicon 预测出更准确的情感偏差
"""

import re
import numpy as np
from typing import Optional


# ============================================================
# 客观语言特征 (不含情感标签)
# ============================================================

# 强调标点 — 客观存在, 不含情感解读
EMPHASIS_PUNCTUATION = {
    '！', '!', '？', '?', '~', '…', '...', '。。。',
    '❗', '❓', '‼', '⁉',
}

# 重复模式检测阈值
REPEAT_THRESHOLD = 3  # 同一字符连续出现 ≥3 次 = 强调

# v3: 无手选唤醒词 — 唤醒关联从 HebbEmotionalLexicon 中学习
# 每个词的 learned_arousal 从 F_body 波动中涌现 (高唤醒时刻 → 共现词被 Hebb 强化)
# 冷启动时所有词 arousal = 0.0 (中性), 随互动逐渐学习


def _count_punctuation_intensity(text: str) -> float:
    """计算标点强度 [0, 1] — 纯客观统计"""
    if not text:
        return 0.0
    count = sum(1 for c in text if c in EMPHASIS_PUNCTUATION)
    # 每10个字1个强调标点 = 正常, >3 = 高强度
    char_len = max(len(text), 1)
    ratio = count / char_len
    return float(np.clip(ratio * 8.0, 0.0, 1.0))


def _detect_repetition_intensity(text: str) -> float:
    """检测字符重复模式强度 [0, 1] — 纯客观统计
    例如: "哈哈哈" → 高, "!!!" → 高
    """
    if len(text) < 2:
        return 0.0
    max_run = 1
    current_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    if max_run >= REPEAT_THRESHOLD:
        return float(np.clip((max_run - 2) / 6.0, 0.0, 1.0))
    return 0.0


def _learned_word_arousal(text: str,
                         lexicon: 'HebbEmotionalLexicon' = None) -> float:
    """从 Hebb 情感词汇网络预测输入文本的词级唤醒度 [0, 1]。

    不依赖手选词集——每个词的唤醒关联从 Agent 的 F_body 波动中学习:
    - 高唤醒时刻 (|F_body| 大) 出现的词 → learned_arousal 高
    - 低唤醒时刻出现的词 → learned_arousal 低
    - 未见过的词 → 0.0 (中性)

    冷启动时所有词都为中性, 随互动逐渐分化。
    """
    if lexicon is None or not lexicon.memory:
        return 0.0

    import jieba
    words = [w for w in jieba.lcut(text) if len(w.strip()) >= 1]
    if not words:
        return 0.0

    total_weight = 0.0
    weighted_arousal = 0.0
    for w in words:
        if w in lexicon.memory:
            learned_a = float(lexicon.memory[w][3])  # learned_arousal
            count = int(lexicon.memory[w][1])
            weight = np.log1p(count)
            weighted_arousal += weight * learned_a
            total_weight += weight

    if total_weight < 1e-8:
        return 0.0
    return float(np.clip(weighted_arousal / total_weight, 0.0, 1.0))


def _length_intensity(text: str) -> float:
    """消息长度反映的强度 — 极短或极长 = 高强度"""
    length = len(text)
    if length < 5:
        return 0.6  # 非常短 → 可能强烈 (如 "滚!", "爱你")
    elif length < 15:
        return 0.3
    elif length > 200:
        return 0.5  # 非常长 → 可能激动
    else:
        return 0.1


# ============================================================
# Hebb 情感词汇网络 — 从 F_body 变化中学习
# ============================================================

class HebbEmotionalLexicon:
    """Hebb 情感词汇网络 — Agent 从自身内感受中学习词的"情感效果"。

    不预设"喜欢=正面"——而是:
    1. 听到词 → 测量 F_body 变化 ∆F
    2. ∆F < 0 (自由能下降) → 身体变好 → 这个词是"好词"
    3. ∆F > 0 (自由能上升) → 身体变差 → 这个词是"坏词"
    4. Hebb 学习: 词 ↔ F_body 效应 共激活 → 关联强化

    这是纯粹的 Hebb + FEP 情感学习, 零手标。
    """

    def __init__(self, dim: int = 64, learning_rate: float = 0.05):
        self.dim = dim
        self.lr = learning_rate

        # 词 → 累积情感效应的 Hebb 记忆
        # 结构: {word_hash: np.array([cumulative_delta_F, count, learned_valence, learned_arousal])}
        self.memory: dict[str, np.ndarray] = {}

        # 词 → 向量 (由 TextEnvironment 填充 — 懒加载)
        self._word_vecs: dict[str, np.ndarray] = {}
        self._word_vec_source = None  # 外部设置的词向量源

    def set_word_vec_source(self, broca_or_word_speaker):
        """设置词向量来源 (Broca 或 WordSpeaker 实例)"""
        self._word_vec_source = broca_or_word_speaker

    def _get_word_vec(self, word: str) -> Optional[np.ndarray]:
        """获取词的向量表示 (从已加载的 Broca/WordSpeaker)"""
        if word in self._word_vecs:
            return self._word_vecs[word]

        if self._word_vec_source is not None:
            if hasattr(self._word_vec_source, '_word_to_vec'):
                v = self._word_vec_source._word_to_vec(word)
                if v is not None:
                    self._word_vecs[word] = v[:64].copy()
                    return self._word_vecs[word]
            # fallback: hash-based deterministic vector
        return None

    def predict_valence(self, words: list[str]) -> float:
        """从已学习的词汇记忆中预测情感效价 [-1, 1]。
        返回 0.0 = 未知/中性 (初始状态)。
        """
        if not words or not self.memory:
            return 0.0

        total_weight = 0.0
        weighted_valence = 0.0

        for w in words:
            if w in self.memory:
                # 记忆的 learned_valence (EMA of ∆F)
                learned_v = float(self.memory[w][2])
                count = int(self.memory[w][1])
                weight = np.log1p(count)  # 更多样本 → 更高权重
                weighted_valence += weight * learned_v
                total_weight += weight

        if total_weight < 1e-8:
            return 0.0

        return float(np.clip(weighted_valence / total_weight, -1.0, 1.0))

    def predict_arousal(self, words: list[str]) -> float:
        """从已学习的词汇记忆中预测唤醒度 [0, 1]。
        返回 0.0 = 未知/中性 (初始状态)。
        """
        if not words or not self.memory:
            return 0.0

        total_weight = 0.0
        weighted_arousal = 0.0
        for w in words:
            if w in self.memory:
                learned_a = float(self.memory[w][3])
                count = int(self.memory[w][1])
                weight = np.log1p(count)
                weighted_arousal += weight * learned_a
                total_weight += weight

        if total_weight < 1e-8:
            return 0.0
        return float(np.clip(weighted_arousal / total_weight, 0.0, 1.0))

    def learn_from_feedback(self, words: list[str], delta_F: float,
                           arousal: float = 0.5):
        """从 F_body 变化中学习: 这些词 → 这个身体效果。

        Args:
            words: 输入消息中的词列表
            delta_F: F_body 的变化 (负 = 变好, 正 = 变差)
            arousal: 当前唤醒度 (高唤醒 → 更强的学习)
        """
        if not words:
            return

        # 高唤醒强化学习 (杏仁核调制)
        effective_lr = self.lr * (0.5 + arousal)

        for w in words:
            if len(w) < 1:
                continue

            if w not in self.memory:
                # 初始化: [cumulative_delta_F, count, learned_valence, learned_arousal]
                self.memory[w] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            entry = self.memory[w]
            # EMA 更新 learned_valence
            entry[2] = (1.0 - effective_lr) * entry[2] + effective_lr * float(delta_F)
            entry[1] += 1.0  # count
            entry[3] = (1.0 - effective_lr) * entry[3] + effective_lr * float(arousal)

    def get_stats(self) -> dict:
        """返回词汇网络统计"""
        if not self.memory:
            return {'n_words': 0, 'top_positive': [], 'top_negative': []}

        items = [(w, float(e[2]), int(e[1]))
                 for w, e in self.memory.items()]
        items.sort(key=lambda x: x[1])
        top_neg = [(w, v) for w, v, _ in items[:5]]
        top_pos = [(w, v) for w, v, _ in items[-5:][::-1]]

        return {
            'n_words': len(self.memory),
            'top_positive': top_pos,
            'top_negative': top_neg,
            'most_experienced': sorted(items, key=lambda x: -x[2])[:5],
        }


# ============================================================
# 全局单例 (跨轮次持久)
# ============================================================

_emotional_lexicon: Optional[HebbEmotionalLexicon] = None


def get_emotional_lexicon() -> HebbEmotionalLexicon:
    """获取全局情感词汇网络 (单例)"""
    global _emotional_lexicon
    if _emotional_lexicon is None:
        _emotional_lexicon = HebbEmotionalLexicon()
    return _emotional_lexicon


# ============================================================
# 情感分析 (v2: 客观特征 + Hebb 学习)
# ============================================================

def analyze_sentiment(text: str,
                      lexicon: HebbEmotionalLexicon = None
                      ) -> dict:
    """分析文本的情感强度和唤醒度 (v2: 无手标词典)

    valence 来源:
      - 已学习的 Hebb 情感词汇 → 从 F_body 历史中涌现
      - 默认 0.0 (中性, 未学习)

    arousal 来源:
      - 客观语言特征: 标点密度 + 重复模式 + 语气词 + 长度
      - 纯统计, 不含情感标签

    Returns:
        dict with valence, arousal, intensity, and learned_bias flag
    """
    text = text.strip()
    if not text:
        return {
            'valence': 0.0, 'arousal': 0.0,
            'intensity': 0.0, 'learned': False,
        }

    # ---- 唤醒度: 客观特征 + Hebb 学习 ----
    punct_intensity = _count_punctuation_intensity(text)
    repeat_intensity = _detect_repetition_intensity(text)
    # v3: 词级唤醒从 Hebb 记忆学习, 不用手选词集
    learned_arousal = _learned_word_arousal(text, lexicon)
    len_intensity = _length_intensity(text)

    # 综合唤醒度 [0, 1] — 各通道等权平均 (v3: 无手设权重)
    # Hebb 学习通道的权重随记忆增长自然提升:
    #   learned_arousal = 0.0 时 → 仅客观特征贡献
    #   learned_arousal > 0 时 → Hebb 记忆开始驱动唤醒
    n_active_channels = sum(1 for v in [punct_intensity, repeat_intensity,
                                         learned_arousal, len_intensity]
                            if v > 0.01)
    n_active_channels = max(1, n_active_channels)
    arousal = float(np.clip(
        (punct_intensity + repeat_intensity + learned_arousal + len_intensity)
        / n_active_channels,
        0.0, 1.0
    ))

    # 综合强度 — 同等权平均
    intensity = float(np.clip(
        (punct_intensity + repeat_intensity + learned_arousal + len_intensity)
        / max(1, sum(1 for v in [punct_intensity, repeat_intensity,
                                  learned_arousal, len_intensity]
                     if v > 0.005)),
        0.0, 1.0
    ))

    # ---- 效价: Hebb 学习的情感词汇 ----
    if lexicon is not None and lexicon.memory:
        import jieba
        words = [w for w in jieba.lcut(text) if len(w.strip()) >= 1]
        learned_valence = lexicon.predict_valence(words)
        learned = abs(learned_valence) > 0.05
    else:
        learned_valence = 0.0
        learned = False

    return {
        'valence': round(learned_valence, 4),
        'arousal': round(arousal, 4),
        'intensity': round(intensity, 4),
        'learned': learned,
        'arousal_components': {
            'punctuation': round(punct_intensity, 3),
            'repetition': round(repeat_intensity, 3),
            'learned_arousal': round(learned_arousal, 3),
            'length': round(len_intensity, 3),
        },
    }


def sentiment_to_social_signal(sentiment: dict) -> np.ndarray:
    """将情感分析结果编码为 8 维社会感觉信号 (v2)

    填入 s[80:88] — 与 layer1_free_energy.compute_F_social 兼容:
      [80]: arousal_intensity   → 唤醒/强度 (来自客观特征)
      [81]: arousal             → 唤醒度 (同 [80], 兼容旧代码读取 signal[1])
      [82]: learned_valence_scaled → 已学习效价, 缩放至 [0,1]
      [83]: pos_count_norm      → 保留字段 (v2 中来自 learned 强度)
      [84]: intensity           → 情感强度
      [85]: is_human_active     → 人类是否刚输入 (1.0)
      [86]: valence_raw         → 原始效价 [-1,1] (来自 Hebb 记忆)
      [87]: reserved            → 留空
    """
    signal = np.zeros(8, dtype=np.float32)

    # 与旧版兼容的布局
    signal[0] = (sentiment['valence'] + 1.0) / 2.0   # 缩放至 [0, 1]
    signal[1] = sentiment['arousal']                  # [0, 1] 唤醒度
    signal[2] = sentiment['intensity']                # 情感强度 (替代 pos_count_norm)
    signal[3] = 1.0 - sentiment['intensity']          # 低强度 (替代 neg_count_norm)
    signal[4] = sentiment['intensity'] * 0.5          # 缩放的强度
    signal[5] = 1.0                                   # human_active flag
    signal[6] = sentiment['valence']                  # raw valence [-1, 1]
    signal[7] = 0.0                                   # reserved

    return signal
