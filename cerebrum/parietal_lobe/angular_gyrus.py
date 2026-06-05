"""
angular_gyrus.py — 角回 (Angular Gyrus) [v5.6 新增]

对应脑区: BA39 (角回, Angular Gyrus)
所属层级: 大脑 → 顶叶 → 角回

脑区标记: BA39 (angular gyrus) — 顶下小叶的一部分

功能职责 (参考: 语言与大脑 §3.3 Wernicke-Geschwind模型, §5.2 语义系统):
  - 视觉符号→语音转换: 看到文字 → 激活对应的语音表征
  - 阅读通路的关键中继: 视觉皮层(BA17) → 角回(BA39) → Wernicke区(BA22)
  - 多模态语义整合: 跨模态概念激活 (视觉/听觉/触觉→统一的语义表征)
  - 数学/符号处理: 数字和抽象符号的理解

临床对应 (角回损伤):
  - 失读症伴失写症 (Alexia with Agraphia):
    能说话和理解口语, 但不能读或写
  - Gerstmann 综合征 (角回优势半球损伤):
    失写+失算+手指失认+左右失定向
  - 在模型中: angular_gyrus 受损 → 视觉通路无法激活语音表征
    → 阅读通道失效, 只能依赖听觉/语义代理

双路径阅读模型 (Dual-Route Reading Model):
  1. 腹侧通路 (lexical): 熟悉词 → 整体识别 → 直接到语义 (快速)
  2. 背侧通路 (sublexical): 不熟悉词 → 字→音转换规则 → 语音→语义 (慢)
  角回在腹侧通路中起关键作用.

在 NotMe 中的集成:
  - 快速路径 (current): text → MiniLM → s[0:64] (语义代理, 效率优先)
  - 脑路径 (new):    text → 字符视觉 → 角回 → 语音向量 → Wernicke (生物合理)
  两条路径共存 (双路径模型), 结果可以合并提高理解鲁棒性.

与现有模块的关系:
  - 接收: IT皮层 (物体识别) 输出的字符视觉特征
  - 输出: 语音向量 → Wernicke区 → 理解
  - 协作: 与弓状束协作 — 角回→Wernicke→AF→Broca = 朗读通路

参考:
  - Dehaene, S., & Cohen, L. (2011). The unique role of the visual word
    form area in reading. Trends in Cognitive Sciences.
  - Pugh, K. R., et al. (2000). The angular gyrus in developmental dyslexia.
  - Geschwind, N. (1965). Disconnexion syndromes in animals and man. Brain.
"""

import numpy as np
import os
from typing import Optional

from cns.data_types import D, Theta, Cluster
from cerebrum.limbic_system.hippocampus import ClusterNetwork, _masked_cosine


class AngularGyrus:
    """角回 — 视觉文字符号→语音表征的 Hebb 映射.

    学习的映射:
      (字符视觉模式, 上下文) → 语音词向量 (64-dim, 音频频谱空间)

    训练方式:
      对词表中的每个汉字/词:
        1. 生成字符的简化视觉表示 (位图/笔画编码)
        2. 配对词的音频频谱向量 (来自 word_spectrum_dataset)
        3. Hebb 学习: 视觉模式 ↔ 语音向量
    """

    def __init__(self, cache_dir: str = None):
        """初始化角回.

        Args:
            cache_dir: 缓存目录
        """
        # Hebb 网络: 视觉字形 → 语音
        # centroid[:64]   = 字符视觉特征 (简化位图编码)
        # centroid[64:128] = 语音词向量 (音频频谱空间)
        # centroid[128:192] = 语义上下文 (来自周围词的语义)
        ag_theta = Theta()
        ag_theta.cluster_threshold = 0.15  # 低阈值: 允许变体 (同一字的不同字体)
        ag_theta.learn_rate_l0 = 0.05
        self.grapheme_to_phoneme = ClusterNetwork(ag_theta)
        self._n_associations: int = 0

        # 词→语音缓存 (快速查找, 用于已训练的词)
        self._word_to_phoneme: dict[str, np.ndarray] = {}

        # 缓存
        if cache_dir is None:
            base = os.path.dirname(__file__)
            cache_dir = os.path.join(base, '.cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 已训练标志
        self._trained: bool = False

        # 双路径权重: 脑路径 vs 快速路径
        self.brain_path_weight: float = 0.3  # 脑路径初始权重 (低, 因为简单视觉编码)

    # ================================================================
    # 字符视觉编码
    # ================================================================

    @staticmethod
    def encode_character_visual(char: str) -> np.ndarray:
        """将单个汉字编码为简化视觉特征向量.

        使用字符的 Unicode 码点 + 笔画数代理 + 结构特征,
        生成一个简化的 "视觉字形" 表示.
        这不是真实的位图渲染, 而是分布式特征编码.

        Args:
            char: 单个汉字

        Returns:
            (64,) float32 视觉字形特征向量
        """
        if not char:
            return np.zeros(64, dtype=np.float32)

        vec = np.zeros(64, dtype=np.float32)

        # 1. Unicode 码点编码 (CJK统一汉字: U+4E00-U+9FFF)
        cp = ord(char)
        # 归一化到 [0,1] 的 CJK 基本区
        cjk_start, cjk_end = 0x4E00, 0x9FFF
        if cjk_start <= cp <= cjk_end:
            vec[0] = (cp - cjk_start) / (cjk_end - cjk_start)
            vec[1] = 1.0  # is_cjk 标记
        elif 0x3400 <= cp <= 0x4DBF:  # CJK扩展A
            vec[0] = (cp - 0x3400) / (0x4DBF - 0x3400) * 0.5
            vec[1] = 0.8
        else:
            vec[0] = cp / 0xFFFF  # 非CJK字符
            vec[1] = 0.0

        # 2. 笔画数代理 (粗略: 基于码点间距)
        # CJK字符按部首/笔画排列, 相邻码点常有相似结构
        vec[2] = (cp % 256) / 256.0   # 低位字节 → 笔画复杂度代理
        vec[3] = (cp % 16) / 16.0     # 更细的粒度

        # 3. 部首/偏旁分布特征
        # 用码点各字节作为散列特征
        for i, byte_val in enumerate(cp.to_bytes(4, 'big')):
            if i < 4:
                vec[4 + i * 4] = (byte_val & 0xF0) / 256.0   # 高半字节
                vec[5 + i * 4] = (byte_val & 0x0F) / 16.0    # 低半字节
                vec[6 + i * 4] = (byte_val % 7) / 7.0        # 模特征
                vec[7 + i * 4] = (byte_val % 13) / 13.0      # 模特征2

        # 4. 笔画结构特征 (基于 Unicode block 的粗略代理)
        # CJK基本区按部首排列, 位置粗略反映结构复杂度
        if cjk_start <= cp <= cjk_end:
            relative_pos = (cp - cjk_start) / (cjk_end - cjk_start)
            vec[20] = relative_pos
            vec[21] = np.sin(relative_pos * np.pi * 8) * 0.5 + 0.5
            vec[22] = np.cos(relative_pos * np.pi * 13) * 0.5 + 0.5
            vec[23] = 1.0 - abs(relative_pos - 0.5) * 2.0  # 中间最高

        # 5. 多元特征散列
        for i in range(24, 64):
            vec[i] = ((cp * (i + 1)) % 256) / 256.0

        return vec.astype(np.float32)

    @staticmethod
    def encode_text_visual(text: str) -> np.ndarray:
        """将文本字符串编码为视觉字形特征序列的聚合向量.

        Args:
            text: 文本字符串 (任意长度)

        Returns:
            (64,) float32 聚合视觉字形向量
        """
        if not text:
            return np.zeros(64, dtype=np.float32)

        char_vecs = []
        for ch in text:
            cv = AngularGyrus.encode_character_visual(ch)
            char_vecs.append(cv)

        if not char_vecs:
            return np.zeros(64, dtype=np.float32)

        # 加权平均: 前面的字权重更高 (阅读的序列位置效应)
        n = len(char_vecs)
        weights = np.exp(-0.5 * np.arange(n) / max(n, 1))
        weights /= weights.sum()

        result = np.zeros(64, dtype=np.float32)
        for cv, w in zip(char_vecs, weights):
            result += w * cv

        return result.astype(np.float32)

    # ================================================================
    # 阅读通路: 视觉字形 → 语音
    # ================================================================

    def read(self, text: str, context_vec: np.ndarray = None
            ) -> tuple[np.ndarray, float]:
        """阅读通路: 文字 → 视觉字形 → 语音表征.

        完整脑路径:
          text → encode_text_visual() → grapheme_to_phoneme.recall()
          → centroid[64:128] = 语音词向量

        如果词在快速缓存中 (已训练的词), 直接返回.
        否则通过 Hebb 网络检索最接近的视觉→语音映射.

        Args:
            text: 要阅读的文本
            context_vec: 语义上下文 (64,), 用于消歧

        Returns:
            (phonological_vec, confidence)
            - phonological_vec: 语音向量 (64,)
            - confidence: 阅读置信度 [0,1]
        """
        if not text:
            return np.zeros(64, dtype=np.float32), 0.0

        # ---- 快速路径: 缓存命中 ----
        if text in self._word_to_phoneme:
            return self._word_to_phoneme[text].copy(), 0.9

        # ---- 脑路径: 视觉字形 → Hebb 检索 ----
        visual_vec = self.encode_text_visual(text)

        if self.grapheme_to_phoneme.n_clusters == 0:
            # 空网络 → 回退: 用视觉向量做近似
            return visual_vec.copy(), 0.1

        q = np.zeros(D, dtype=np.float32)
        q[:64] = visual_vec.astype(np.float32)
        if context_vec is not None:
            q[128:192] = context_vec[:64].astype(np.float32)

        h = self.grapheme_to_phoneme.hash_features(q)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True  # 在视觉字形空间比较

        hash_key = self.grapheme_to_phoneme._hash_to_bucket(h)
        bucket = self.grapheme_to_phoneme.buckets.get(hash_key, [])

        if not bucket:
            bucket = self.grapheme_to_phoneme.clusters[:min(500,
                len(self.grapheme_to_phoneme.clusters))]

        best_sim = -1.0
        best_c = None
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            if sim > best_sim:
                best_sim = sim
                best_c = c

        if best_c is not None and best_sim >= self.grapheme_to_phoneme.theta.cluster_threshold:
            phonological_vec = best_c.centroid[64:128].copy().astype(np.float32)
            confidence = float(best_sim)
        else:
            # 回退: 视觉向量作为近似
            phonological_vec = visual_vec.copy()
            confidence = 0.1

        return phonological_vec, confidence

    def read_pathway(self, text: str, wernicke_comprehend=None,
                    context_vec: np.ndarray = None
                    ) -> dict:
        """完整阅读通路: 角回 → Wernicke → 理解.

        模拟 Wernicke-Geschwind 阅读模型:
          Visual(BA17) → Angular Gyrus(BA39) → Wernicke(BA22) → comprehension

        Args:
            text: 要阅读的文本
            wernicke_comprehend: Wernicke的comprehend函数 (可选)
            context_vec: 语义上下文

        Returns:
            {
                'phonological_vec': 语音向量,
                'reading_confidence': 阅读置信度,
                'comprehension': 理解结果 (如果wernicke可用),
                'pathway': 'angular_gyrus'
            }
        """
        phonological_vec, confidence = self.read(text, context_vec)

        result = {
            'phonological_vec': phonological_vec,
            'reading_confidence': confidence,
            'pathway': 'angular_gyrus',
        }

        # 如果有 Wernicke, 进一步走理解通路
        if wernicke_comprehend is not None:
            try:
                comp_vec, understanding = wernicke_comprehend(
                    phonological_vec, None
                )
                result['comprehension'] = comp_vec
                result['understanding'] = understanding
            except Exception:
                pass

        return result

    # ================================================================
    # 训练: 从词表学习字形→语音映射
    # ================================================================

    def train_from_vocabulary(self, word_list: list[str],
                             word_vecs: np.ndarray,
                             n_samples: int = None,
                             verbose: bool = True):
        """从词表训练角回的字形→语音 Hebb 映射.

        对每个词:
          1. 编码字符视觉 (encode_character_visual)
          2. 取词向量作为语音目标
          3. Hebb 学习: 视觉 ↔ 语音

        Args:
            word_list: 词列表
            word_vecs: 词向量 (N, 64) 或 (N, D)
            n_samples: 采样数 (None=全部)
            verbose: 是否打印进度
        """
        if verbose:
            print(f"  AngularGyrus: training on {len(word_list)} words...")

        n_total = len(word_list)
        indices = list(range(n_total))
        if n_samples is not None and n_samples < n_total:
            stride = max(1, n_total // n_samples)
            indices = list(range(0, n_total, stride))[:n_samples]

        n_learned = 0
        for idx in indices:
            word = word_list[idx]
            if not word:
                continue

            # 字符视觉编码
            visual = self.encode_text_visual(word)

            # 语音目标: 词向量的前64维 (音频频谱空间)
            wv = word_vecs[idx]
            if wv.ndim > 1:
                wv = wv.ravel()
            phoneme_target = wv[:64].astype(np.float32)
            # 归一化
            norm = np.linalg.norm(phoneme_target)
            if norm > 1e-8:
                phoneme_target = phoneme_target / norm

            # Hebb 学习
            self.learn_grapheme_phoneme(visual, phoneme_target)

            # 缓存
            if len(word) <= 4:  # 只缓存短词 (1-4字)
                self._word_to_phoneme[word] = phoneme_target.copy()

            n_learned += 1

        self._trained = True

        if verbose:
            print(f"  AngularGyrus: {n_learned} grapheme→phoneme "
                  f"associations ({self.grapheme_to_phoneme.n_clusters} clusters)")

    def learn_grapheme_phoneme(self, visual_vec: np.ndarray,
                              phoneme_vec: np.ndarray,
                              context_vec: np.ndarray = None,
                              weight: float = 1.0):
        """学习单个字形→语音映射.

        Args:
            visual_vec: 字符视觉向量 (64,)
            phoneme_vec: 语音向量 (64,)
            context_vec: 语义上下文 (64,) 可选
            weight: 学习权重
        """
        pattern = np.zeros(D, dtype=np.float32)
        pattern[:64] = visual_vec[:64].astype(np.float32)
        pattern[64:128] = phoneme_vec[:64].astype(np.float32)
        if context_vec is not None:
            pattern[128:192] = context_vec[:64].astype(np.float32)

        orig_lr = self.grapheme_to_phoneme.theta.learn_rate_l0
        self.grapheme_to_phoneme.theta.learn_rate_l0 = min(0.30,
            orig_lr * weight)
        self.grapheme_to_phoneme.learn(pattern)
        self.grapheme_to_phoneme.theta.learn_rate_l0 = orig_lr

        self._n_associations += 1

    # ================================================================
    # 诊断
    # ================================================================

    def can_read(self, text: str) -> bool:
        """检查是否能"阅读"这段文字 (是否有足够的字形→语音映射)."""
        if text in self._word_to_phoneme:
            return True
        return self.grapheme_to_phoneme.n_clusters > 0

    @property
    def n_grapheme_clusters(self) -> int:
        return self.grapheme_to_phoneme.n_clusters

    def get_state(self) -> dict:
        """返回当前状态 (供 dashboard 使用)."""
        return {
            'n_grapheme_clusters': self.n_grapheme_clusters,
            'n_cached_words': len(self._word_to_phoneme),
            'trained': self._trained,
            'brain_path_weight': float(self.brain_path_weight),
        }

    def __repr__(self) -> str:
        return (f"AngularGyrus(clusters={self.n_grapheme_clusters}, "
                f"cached={len(self._word_to_phoneme)} words, "
                f"trained={self._trained})")
