"""
phonological_loop.py — 语音回路 (Phonological Loop) [v5.6 新增]

对应脑区: 语音回路 (Baddeley工作记忆模型的语音组件)
所属层级: 大脑 → 额叶 → 语音回路

脑区标记: 缘上回 (BA40) + Broca区(BA44, 默读复述) + 颞上回后部(语音存储)

功能职责 (参考: 语言与大脑 §5.2 双重通路模型):
  - 语音存储 (Phonological Store): 保持最近听到的语音信息 ~2秒
  - 默读复述 (Articulatory Rehearsal): 无声复述刷新存储, 防止消退
  - 容量限制: ~7±2 个组块 (Miller定律)
  - 时间限制: 无复述时 ~2秒消退

神经基础:
  - 语音存储: 缘上回 (BA40) + 颞上回后部
  - 默读复述: Broca区 (BA44) + 前运动皮层 (BA6) — 无声"说话"
  - 背侧通路 (dorsal stream) → 声→运动映射 → 复述回路
  - 这个回路也是语言习得的关键: 听到新词→存储→复述→学会

在 NotMe 中的集成:
  - comprehend() → 写入语音存储 (听到人类的话)
  - broca.speak_from_state() → 读取存储维持长句连贯性
  - self-hearing → 自己的话也进入回路 → 自我监控
  - decay(): 每时间步消退 → 模拟 ~2秒记忆广度
  - rehearse(): 无声复述 → 维持工作记忆内容
  - 与弓状束协作: AF 背侧通路 = 默读复述的神经基质

神经学病症对应:
  - 传导性失语症: 语音存储完好, 但复述回路(AF)受损 → 无法复述
  - 语音短期记忆障碍: 缘上回损伤 → 存储容量急剧下降
  - 发展性语言障碍: 语音回路容量不足 → 词汇学习困难

参考:
  - Baddeley, A. D., & Hitch, G. (1974). Working memory.
  - Baddeley, A. D. (2003). Working memory: Looking back and looking forward.
    Nature Reviews Neuroscience.
  - Hickok, G., & Poeppel, D. (2007). The cortical organization of speech
    processing. Nature Reviews Neuroscience. (背侧通路 = 语音回路)
"""

import numpy as np
from typing import Optional


class PhonologicalLoop:
    """语音回路 — 言语工作记忆缓冲器.

    不是符号存储, 而是分布式向量缓冲器 —
    每个"组块"是一个词向量 (64-dim, 音频频谱空间),
    随时间和干扰逐渐消退.

    容量与时限:
      - 最大组块数: 7±2 (默认7)
      - 消退时间常数: ~2秒 (无复述时)
      - 复述刷新率: 每时间步一次
    """

    def __init__(self, capacity: int = 7, decay_per_step: float = 0.08):
        """初始化语音回路.

        Args:
            capacity: 最大组块数 (Miller定律 ~7±2)
            decay_per_step: 每步消退率 (默认 0.08 → ~12步=2秒 @10fps)
        """
        self.max_chunks = capacity
        self.decay_rate = decay_per_step

        # 语音存储: [(word_vector_64, activation, timestamp), ...]
        # activation下降 → 消退; 复述刷新 → activation回升
        self.store: list[tuple[np.ndarray, float, int]] = []

        # 默读复述追踪
        self.is_rehearsing: bool = False
        self.rehearsal_rate: float = 0.15  # 每次复述恢复的激活度
        self._step_count: int = 0

        # 追踪
        self.total_heard: int = 0
        self.total_rehearsed: int = 0
        self.total_decayed: int = 0

        # 语音相似性干扰: 相似声音互相干扰 → 加速消退
        self.phonological_interference: float = 0.0

        # 词长效应 (Word Length Effect):
        # 长词占用更多复述时间 → 容量effective变小
        # (中文单音节词多, 默认效应较弱)
        self.word_length_penalty: float = 0.05

    # ================================================================
    # 核心操作
    # ================================================================

    def hear(self, word_vec: np.ndarray, word_len: int = 1):
        """新词进入语音存储.

        听到一个词 → 进入语音回路, 同时可能推出最旧的组块.

        Args:
            word_vec: 词向量 (64,), 音频频谱空间
            word_len: 词长 (音节数, 影响词长效应)
        """
        if word_vec is None:
            return

        v = np.asarray(word_vec, dtype=np.float32).ravel()[:64].copy()

        # 归一化
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v = v / norm

        # 检查是否与已有组块重复 (同一词 → 刷新而非新增)
        for i, (existing, act, ts) in enumerate(self.store):
            sim = float(np.dot(existing, v))
            if sim > 0.9:
                # 重复词 → 刷新激活度
                self.store[i] = (existing, min(1.0, act + 0.2),
                                self._step_count)
                self.total_heard += 1
                return

        # 新组块: 激活度 = 0.8 (新鲜, 高激活)
        activation = 0.8 - word_len * self.word_length_penalty
        activation = max(0.3, min(1.0, activation))

        self.store.append((v, activation, self._step_count))
        self.total_heard += 1

        # 维持容量上限
        while len(self.store) > self.max_chunks:
            self._evict_weakest()

    def rehearse(self):
        """默读复述: 刷新所有存储组块的激活度.

        这模拟了无声"在心里重复"的过程 —
        Broca区+前运动皮层激活语音运动计划,
        但不执行实际发声. 复述通过背侧通路映射回语音存储.

        复述效果:
          - 所有组块激活度回升
          - 但容量上限不变 (复述不增加容量, 只是维持)
          - 长词复述更慢 → 词长效应的来源
        """
        if not self.store:
            return

        self.is_rehearsing = True
        refreshed = 0

        for i, (vec, act, ts) in enumerate(self.store):
            # 复述: 激活度回升 (但不回到初始值 — 有衰减)
            new_act = min(1.0, act + self.rehearsal_rate)
            self.store[i] = (vec, new_act, ts)
            if new_act > act + 0.01:
                refreshed += 1

        self.total_rehearsed += 1

    def decay(self, dt: float = 1.0):
        """被动消退: 无复述时激活度随时间下降.

        每时间步调用一次 (dt=1.0 = 一个时间步 ~100ms @10fps).
        激活度低于阈值 → 组块从存储中移除.

        Args:
            dt: 时间步数 (默认1步)
        """
        if not self.store:
            self.is_rehearsing = False
            return

        # 语音相似性干扰: 相似声音互相干扰 → 加速消退
        interference_factor = 1.0 + self._compute_interference()

        decayed = 0
        new_store = []

        for vec, act, ts in self.store:
            # 消退: 指数衰减
            new_act = act * (1.0 - self.decay_rate * dt * interference_factor)

            if new_act > 0.1:  # 低于阈值 → 移除
                new_store.append((vec, new_act, ts))
            else:
                decayed += 1

        self.store = new_store
        self.total_decayed += decayed
        self.is_rehearsing = False

    # ================================================================
    # 查询
    # ================================================================

    def get_loop_vector(self, recency_weighted: bool = True
                        ) -> np.ndarray:
        """获取当前回路内容的聚合向量.

        可用于:
          - comprehend() 上下文 (刚听到什么?)
          - broca.speak_from_state() 连贯性 (接下来该说什么?)
          - self-monitoring (我刚才说了什么?)

        Args:
            recency_weighted: 是否按近因加权 (True=最近的高权重)

        Returns:
            (64,) float32 聚合向量
        """
        if not self.store:
            return np.zeros(64, dtype=np.float32)

        if recency_weighted:
            # 激活度 × 近因 (时间戳越新权重越高)
            now = self._step_count
            weights = np.array([
                act * (1.0 / (1.0 + 0.1 * (now - ts)))
                for _, act, ts in self.store
            ])
        else:
            weights = np.array([act for _, act, _ in self.store])

        total_w = weights.sum()
        if total_w < 1e-8:
            return np.zeros(64, dtype=np.float32)

        weighted = np.zeros(64, dtype=np.float32)
        for (vec, _, _), w in zip(self.store, weights):
            weighted += w * vec

        return (weighted / total_w).astype(np.float32)

    def get_recent_words(self, n: int = 3) -> list[np.ndarray]:
        """获取最近 n 个词的向量 (按时间戳排序).

        Args:
            n: 返回的词数

        Returns:
            [word_vec_64, ...] 按时间从旧到新
        """
        sorted_items = sorted(self.store, key=lambda x: x[2])
        return [vec.copy() for vec, _, _ in sorted_items[-n:]]

    def can_repeat(self, n_words: int = 3) -> bool:
        """检查是否能复述 n 个词 (传导性失语症测试).

        需要: (1) 存储中有足够的激活组块
              (2) 组块之间有足够的激活度

        Args:
            n_words: 需要复述的词数

        Returns:
            True 如果可以复述
        """
        if len(self.store) < n_words:
            return False

        # 检查激活度: 所有需要的组块激活度 > 0.3
        sorted_items = sorted(self.store, key=lambda x: x[2])
        recent = sorted_items[-n_words:]
        return all(act > 0.3 for _, act, _ in recent)

    @property
    def n_chunks(self) -> int:
        """当前存储的组块数."""
        return len(self.store)

    @property
    def mean_activation(self) -> float:
        """平均激活度."""
        if not self.store:
            return 0.0
        return float(np.mean([act for _, act, _ in self.store]))

    @property
    def load(self) -> float:
        """回路负载 [0,1]."""
        return min(1.0, len(self.store) / self.max_chunks)

    # ================================================================
    # 内部方法
    # ================================================================

    def _evict_weakest(self):
        """移除激活度最低的组块 (容量管理)."""
        if not self.store:
            return
        min_idx = min(range(len(self.store)),
                     key=lambda i: self.store[i][1])
        self.store.pop(min_idx)

    def _compute_interference(self) -> float:
        """计算语音相似性干扰.

        存储中相似向量越多 → 互相干扰 → 消退加速.
        """
        if len(self.store) < 2:
            self.phonological_interference = 0.0
            return 0.0

        vecs = np.stack([v for v, _, _ in self.store])
        # 成对相似度平均值
        sims = np.dot(vecs, vecs.T)
        # 去掉对角线
        mask = np.eye(len(vecs), dtype=bool)
        mean_sim = sims[~mask].mean() if len(vecs) > 1 else 0.0

        self.phonological_interference = float(np.clip(mean_sim, 0.0, 0.5))
        return self.phonological_interference

    # ================================================================
    # 步进 (每时间步调用)
    # ================================================================

    def step(self, is_silent: bool = True):
        """每时间步更新.

        Args:
            is_silent: 是否无外部输入 (True=无人说话, 自然消退)
        """
        self._step_count += 1

        if is_silent and not self.is_rehearsing:
            # 无输入且未复述 → 消退
            self.decay(dt=1.0)

    # ================================================================
    # 序列级操作 (用于长句处理)
    # ================================================================

    def hear_sequence(self, word_vecs: list[np.ndarray]):
        """一次性听入整个词序列.

        用于: 完整句子一次性进入语音回路 (如自我言语监控).

        Args:
            word_vecs: 词向量列表 [(64,), ...]
        """
        for wv in word_vecs:
            self.hear(wv)
        # 听入后自动复述一次 (巩固)
        self.rehearse()

    def get_sequence_coherence(self, target_len: int = 7) -> float:
        """评估回路中序列的连贯性.

        高连贯性 = 组块之间存在自然的过渡模式
        (相邻组块的向量相似度适中 — 不太相同也不完全无关).

        Returns:
            coherence [0,1] — 越高越连贯
        """
        if len(self.store) < 2:
            return 0.5  # 中性

        vecs = [v for v, _, _ in self.store]
        adjacent_sims = []
        for i in range(len(vecs) - 1):
            sim = float(np.dot(vecs[i], vecs[i+1]))
            adjacent_sims.append(sim)

        # 理想连贯性: 相邻相似度 ~0.3-0.7 (有关联但不重复)
        coherence_scores = []
        for s in adjacent_sims:
            # 太高(>0.9) = 重复, 太低(<0.1) = 不连贯
            if s > 0.9:
                coherence_scores.append(0.2)
            elif s < 0.1:
                coherence_scores.append(0.2)
            else:
                coherence_scores.append(min(1.0, s * 1.5))

        return float(np.mean(coherence_scores)) if coherence_scores else 0.5

    # ================================================================
    # 诊断
    # ================================================================

    def get_state(self) -> dict:
        """返回当前回路状态 (供 dashboard 使用)."""
        return {
            'n_chunks': self.n_chunks,
            'max_chunks': self.max_chunks,
            'mean_activation': float(self.mean_activation),
            'load': float(self.load),
            'is_rehearsing': self.is_rehearsing,
            'phonological_interference': float(self.phonological_interference),
            'total_heard': self.total_heard,
            'total_decayed': self.total_decayed,
            'can_repeat_3': self.can_repeat(3),
            'sequence_coherence': float(self.get_sequence_coherence()),
        }

    def __repr__(self) -> str:
        return (f"PhonologicalLoop(chunks={self.n_chunks}/{self.max_chunks}, "
                f"activation={self.mean_activation:.2f}, "
                f"rehearsing={self.is_rehearsing})")
