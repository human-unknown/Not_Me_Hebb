"""
phrase_structure.py — 短语结构网络 (Phrase Structure Network) [v5.6 新增]

对应脑区: Broca区 BA44 (岛盖部, 层级句法结构构建)
所属层级: 大脑 → 额叶 → 短语结构网络

脑区标记: BA44 (pars opercularis) — Broca区的核心句法处理部分

功能职责 (参考: 语言与大脑 §3.1 Broca失语症, §5.2 句法加工):
  - 短语边界检测: 从语料统计中发现短语的自然边界
  - 短语聚类: 相似内部结构的短语 → 短语类型涌现
  - 递归嵌入: 短语可以嵌套 (递归性 — 人类语言的标志特征)
  - 句法约束: 为 Broca 的 speak_from_state() 提供柔性句法偏向

核心机制:
  1. 边界检测: 转移概率骤降 → 短语边界
     p(w_{i+1} | w_i) 显著低于上下文平均值 → boundary
  2. 短语聚类: 边界间序列 → 平均词向量 → Hebb聚类 → 短语类型涌现
  3. 结构生成: speak_from_state() 时, 候选词由 trigram + 短语连贯性双约束

设计原则 (符合项目原则):
  - 零手写规则: 所有短语类型从语料统计中涌现, 不预设 NP/VP/PP 等类别
  - Hebb学习: 短语模式存储为 ClusterNetwork 集群
  - 柔性约束: 短语结构是偏向(bias), 不是硬规则 —
    高温度时短语约束弱, 低温度时强

神经学病症对应:
  - Broca失语症 (语法缺失/agrammatism):
    短语结构网络受损 → 只能产出"电报式言语"
    内容词保留, 功能词缺失, 无层级句法结构
  - 在模型中: phrase_strength 降低 → 回退到纯 trigram 模式

参考:
  - Friederici, A. D. (2011). The brain basis of language processing:
    from structure to function. Physiological Reviews.
  - Hagoort, P. (2013). MUC (Memory, Unification, Control) and beyond.
    Frontiers in Psychology.
  - Chomsky, N. (1957). Syntactic Structures.
    (递归性与层级结构是核心洞察, 但实现走统计涌现路线, 非符号规则)
"""

import numpy as np
import os
from typing import Optional

from cns.data_types import D, Theta, Cluster
from cerebrum.limbic_system.hippocampus import ClusterNetwork, _masked_cosine


class PhraseStructureNetwork:
    """短语结构网络 — 从语料统计中涌现的层级句法.

    不是 Chomsky 式的符号语法, 而是 Hebb 统计学习:
      1. 检测转移概率骤降 → 短语边界点
      2. 聚类边界间的词序列 → 短语类型涌现
      3. 生成时用短语连贯性柔性约束词选择
    """

    def __init__(self, cache_dir: str = None):
        """初始化短语结构网络.

        Args:
            cache_dir: 缓存目录
        """
        # 短语模式网络: centroid存储短语签名
        # - centroid[:64]   = 短语的语义合成向量 (组成词的平均)
        # - centroid[64:128] = 短语类型签名 (类似短语的centroid聚类)
        # - centroid[128:192] = 短语内部过渡模式 (简化)
        phrase_theta = Theta()
        phrase_theta.cluster_threshold = 0.20
        phrase_theta.learn_rate_l0 = 0.04
        self.phrase_net = ClusterNetwork(phrase_theta)
        self._n_phrases: int = 0

        # 转移概率矩阵 (从语料学习)
        # {(word_i, word_j): transition_probability}
        self.bigram_probs: dict[tuple[str, str], float] = {}
        self.unigram_counts: dict[str, int] = {}
        self._total_bigrams: int = 0

        # 边界阈值: 转移概率低于此值 → 短语边界
        self.boundary_threshold: float = 0.02

        # 短语类型统计
        self.phrase_types: dict[int, list[str]] = {}  # {type_id: [phrase_str, ...]}

        # 缓存
        if cache_dir is None:
            base = os.path.dirname(__file__)
            cache_dir = os.path.join(base, '.cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 是否已训练
        self._trained: bool = False

    # ================================================================
    # 训练: 从语料学习转移概率和短语边界
    # ================================================================

    def train_from_corpus(self, sentences: list[str],
                         min_bigram_count: int = 2,
                         n_samples: int = None,
                         verbose: bool = True):
        """从句子语料学习 bigram 转移概率和短语边界.

        Args:
            sentences: 句子列表
            min_bigram_count: bigram 最小出现次数 (过滤噪声)
            n_samples: 采样句子数 (None=全部)
            verbose: 是否打印进度
        """
        import jieba

        if n_samples is not None and n_samples < len(sentences):
            stride = max(1, len(sentences) // n_samples)
            sample_indices = list(range(0, len(sentences), stride))[:n_samples]
            target_sents = [sentences[i] for i in sample_indices]
        else:
            target_sents = sentences

        if verbose:
            print(f"  PhraseStructure: training on {len(target_sents)} sentences...")

        # ---- Step 1: 统计 bigram 频率 ----
        bigram_counts: dict[tuple[str, str], int] = {}
        self.unigram_counts = {}

        for sent in target_sents:
            words = [w for w in jieba.lcut(sent) if len(w.strip()) >= 1]
            if len(words) < 2:
                continue

            for w in words:
                self.unigram_counts[w] = self.unigram_counts.get(w, 0) + 1

            for i in range(len(words) - 1):
                key = (words[i], words[i + 1])
                bigram_counts[key] = bigram_counts.get(key, 0) + 1

        # ---- Step 2: 转换为概率 ----
        self._total_bigrams = sum(bigram_counts.values())
        total_unigrams = sum(self.unigram_counts.values())

        # 转移概率: P(w2 | w1) = count(w1,w2) / count(w1)
        self.bigram_probs = {}
        for (w1, w2), count in bigram_counts.items():
            if count >= min_bigram_count:
                w1_count = self.unigram_counts.get(w1, 1)
                self.bigram_probs[(w1, w2)] = count / w1_count

        # ---- Step 3: 自适应边界阈值 ----
        if self.bigram_probs:
            all_probs = list(self.bigram_probs.values())
            # 边界阈值 = 下四分位数 (最低的 25% 转移概率)
            sorted_probs = sorted(all_probs)
            q1_idx = max(0, len(sorted_probs) // 4)
            self.boundary_threshold = sorted_probs[q1_idx]
        else:
            self.boundary_threshold = 0.02

        # ---- Step 4: 提取短语并聚类 ----
        phrases_extracted = 0
        for sent in target_sents[:min(10000, len(target_sents))]:
            words = [w for w in jieba.lcut(sent) if len(w.strip()) >= 1]
            boundaries = self.detect_boundaries(words)

            # 在边界之间提取短语 (2-5词长)
            prev_b = 0
            for b_idx in boundaries:
                if b_idx - prev_b >= 2 and b_idx - prev_b <= 5:
                    phrase_words = words[prev_b:b_idx]
                    self._add_phrase(phrase_words)
                    phrases_extracted += 1
                prev_b = b_idx

            # 句子末尾
            if len(words) - prev_b >= 2 and len(words) - prev_b <= 5:
                phrase_words = words[prev_b:]
                self._add_phrase(phrase_words)
                phrases_extracted += 1

        self._trained = True

        if verbose:
            print(f"  PhraseStructure: {len(self.bigram_probs)} bigrams, "
                  f"threshold={self.boundary_threshold:.4f}, "
                  f"{self.phrase_net.n_clusters} phrase clusters "
                  f"({phrases_extracted} phrases)")

    # ================================================================
    # 短语边界检测
    # ================================================================

    def detect_boundaries(self, word_sequence: list[str]) -> list[int]:
        """检测词序列中的短语边界.

        原理: 转移概率骤降 → 短语边界.
        不需要任何语法知识, 纯粹从 bigram 统计中涌现.

        例如: "我/喜欢/吃/苹果"
          P(喜欢|我)=0.04, P(吃|喜欢)=0.08, P(苹果|吃)=0.12
          阈值为 0.02, 无边界的自然流 → 整个是一个短语

          而: "我/的/猫/喜欢/鱼"
          P(的|我)=0.45, P(猫|的)=0.85 → 高概率短语内
          P(喜欢|猫)=0.01 → < 边界阈值 → 边界!
          → "我的猫" 是一个 NP, "喜欢鱼" 是 VP

        Args:
            word_sequence: 词列表

        Returns:
            边界位置列表 (边界 = 下一个词是新短语的开始)
        """
        if not self._trained or len(word_sequence) < 2:
            return []

        boundaries = []
        for i in range(len(word_sequence) - 1):
            key = (word_sequence[i], word_sequence[i + 1])
            prob = self.bigram_probs.get(key, 0.0)

            # 未见过的大 → 极低概率 → 很可能是边界
            if prob < self.boundary_threshold:
                # 但标点总是边界 (jieba已分好)
                boundaries.append(i + 1)

        return boundaries

    def boundary_probability(self, w1: str, w2: str) -> float:
        """返回 w1 和 w2 之间是短语边界的概率.

        Args:
            w1: 前一个词
            w2: 后一个词

        Returns:
            p_boundary [0,1], 越高越可能是边界
        """
        if not self._trained:
            return 0.2  # 中性

        prob = self.bigram_probs.get((w1, w2), 0.0)

        # 转移概率越低 → 边界概率越高
        if prob < self.boundary_threshold:
            return 1.0 - prob / max(self.boundary_threshold, 1e-8)
        else:
            # 转移概率高 → 边界概率低
            return max(0.0, 0.5 - prob)

    # ================================================================
    # 短语生成约束
    # ================================================================

    def phrase_coherence(self, current_phrase_words: list[str],
                        candidate_word: str) -> float:
        """评估候选词是否与当前正在构建的短语连贯.

        高连贯性 = 候选词与短语中已有词有高转移概率.
        低连贯性 = 可能是短语边界.

        Args:
            current_phrase_words: 当前短语中已有的词
            candidate_word: 候选的下一个词

        Returns:
            coherence [0,1], 越高越连贯
        """
        if not self._trained or not current_phrase_words:
            return 0.5  # 中性

        last_word = current_phrase_words[-1]
        key = (last_word, candidate_word)
        prob = self.bigram_probs.get(key, 0.0)

        # 与全局平均比较
        avg_prob = self.boundary_threshold * 2  # 估计的平均概率
        coherence = np.clip(prob / max(avg_prob, 1e-8), 0.0, 1.0)

        return float(coherence)

    def phrase_closure_probability(self, current_phrase_words: list[str]
                                  ) -> float:
        """当前短语应该结束的概率.

        原则: 短语不能无限延长.
          - 短语越长 → 结束概率越高
          - 最后一个词是高频"短语结束词" → 结束概率高

        Args:
            current_phrase_words: 当前短语中的词

        Returns:
            p_closure [0,1], 越高越应该结束当前短语
        """
        if not current_phrase_words:
            return 0.0

        n = len(current_phrase_words)

        # 长度因素: 2-3词是最常见的短语长度
        length_factor = min(1.0, max(0.0, (n - 3) / 3.0))

        # 转移概率因素: 最后一个词之后出现任何词的概率都低 → 可能是结束
        last_word = current_phrase_words[-1]
        if last_word in self.unigram_counts:
            # 检查这个词作为 bigram 首词的频率
            n_as_first = sum(
                1 for (w1, w2) in self.bigram_probs if w1 == last_word)
            n_total = self.unigram_counts.get(last_word, 1)
            # 作为首词的比例低 → 更多是结束词
            first_ratio = n_as_first / n_total if n_total > 0 else 0.0
            trailing_factor = 1.0 - min(1.0, first_ratio * 5.0)
        else:
            trailing_factor = 0.5

        closure = float(np.clip(
            0.4 * length_factor + 0.6 * trailing_factor, 0.0, 1.0))

        return closure

    def modulate_candidates(self, candidates: list[tuple[str, float, np.ndarray]],
                           current_phrase_words: list[str],
                           phrase_strength: float = 0.3
                           ) -> list[tuple[str, float, np.ndarray]]:
        """用短语结构约束调制候选词得分.

        这是 speak_from_state() 的插件:
          候选词 = trigram_score × (1 + phrase_coherence × phrase_strength)

        Args:
            candidates: [(word, score, vec), ...]
            current_phrase_words: 当前短语中已有的词
            phrase_strength: 短语约束强度 [0,1]
                           0 = 纯 trigram (无短语约束)
                           1 = 强短语约束 (Broca 区正常功能)
                           低值模拟 Broca 失语症的语法缺失

        Returns:
            调制后的候选列表, 得分已调整
        """
        if not self._trained or not current_phrase_words:
            return candidates

        modulated = []
        for word, score, vec in candidates:
            coherence = self.phrase_coherence(current_phrase_words, word)
            closure = self.phrase_closure_probability(current_phrase_words)

            # 短语连贯性奖励: 连贯 → 得分增加
            # 但如果短语应该结束 → 连贯性过高的词反而被轻微惩罚
            if closure > 0.6:
                # 短语应该结束了 → 开启新短语的词获得奖励
                new_phrase_bonus = (1.0 - coherence) * closure
                adjusted = score * (1.0 + new_phrase_bonus * phrase_strength)
            else:
                # 短语中 → 连贯的词获得奖励
                adjusted = score * (1.0 + coherence * phrase_strength)

            modulated.append((word, adjusted, vec))

        # 重新排序
        modulated.sort(key=lambda x: x[1], reverse=True)
        return modulated

    # ================================================================
    # 内部方法
    # ================================================================

    def _add_phrase(self, words: list[str]):
        """将词序列作为短语添加到网络中.

        centroid[:64]   = 词向量的加权平均 (短语语义)
        centroid[64:128] = 短语类型签名 (从词序和边界特征派生)
        """
        if len(words) < 2:
            return

        # 短语语义 = 词的平均 (需要用词向量, 但这里只存占位)
        # 实际词向量由 broca._word_to_vec() 提供
        # 此处存储的是短语结构信息, 不是语义
        phrase_vec = np.zeros(D, dtype=np.float32)

        # 前64维: 用词的长度和位置编码短语结构
        for i, w in enumerate(words):
            # 简单编码: 词的长度和位置
            hash_val = hash(w) % 1000 / 1000.0
            idx = min(i * 8, 56)
            phrase_vec[idx] = hash_val
            phrase_vec[idx + 1] = len(w) / 10.0  # 词长归一化

        # 中间64维: 短语类型的分布式编码
        # 基于词数和内部转移模式
        phrase_vec[64] = len(words) / 10.0  # 短语长度
        for i in range(len(words) - 1):
            key = (words[i], words[i + 1])
            if key in self.bigram_probs:
                prob = self.bigram_probs[key]
                idx = 65 + min(i * 4, 55)
                phrase_vec[idx] = prob

        h = self.phrase_net.hash_features(phrase_vec)

        # 构建集群
        c = Cluster(centroid=h.copy())
        c.count = 1
        c.activation = 0.15  # 低初始激活

        # 检查是否与已有短语类型重复
        existing = self.phrase_net.recall(phrase_vec)
        if existing is not None:
            existing.count += 1
            existing.activation = min(1.0, existing.activation + 0.05)
        else:
            self.phrase_net.clusters.append(c)
            hash_key = self.phrase_net._hash_to_bucket(h)
            self.phrase_net.buckets.setdefault(hash_key, []).append(c)
            self._n_phrases += 1

    # ================================================================
    # 诊断
    # ================================================================

    def get_boundary_examples(self, sentence: str, n_examples: int = 5
                             ) -> list[dict]:
        """对给定句子检测短语边界并返回示例."""
        import jieba
        words = [w for w in jieba.lcut(sentence) if len(w.strip()) >= 1]
        boundaries = self.detect_boundaries(words)

        examples = []
        prev = 0
        for b in boundaries:
            phrase = ''.join(words[prev:b])
            examples.append({
                'phrase': phrase,
                'words': words[prev:b],
                'boundary_at': b,
                'p_boundary': float(self.boundary_probability(
                    words[b-1] if b > 0 else '',
                    words[b] if b < len(words) else '')),
            })
            prev = b
            if len(examples) >= n_examples:
                break

        # 最后一段
        if prev < len(words):
            phrase = ''.join(words[prev:])
            examples.append({
                'phrase': phrase,
                'words': words[prev:],
                'boundary_at': len(words),
                'p_boundary': 1.0,  # 句子结束 = 绝对边界
            })

        return examples

    @property
    def n_phrase_types(self) -> int:
        return self.phrase_net.n_clusters

    def get_state(self) -> dict:
        """返回当前状态 (供 dashboard 使用)."""
        return {
            'n_phrase_types': self.n_phrase_types,
            'n_bigrams': len(self.bigram_probs),
            'boundary_threshold': float(self.boundary_threshold),
            'trained': self._trained,
            'n_phrases': self._n_phrases,
        }

    def __repr__(self) -> str:
        return (f"PhraseStructureNetwork(types={self.n_phrase_types}, "
                f"bigrams={len(self.bigram_probs)}, "
                f"trained={self._trained})")
