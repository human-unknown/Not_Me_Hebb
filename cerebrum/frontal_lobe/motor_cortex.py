"""
motor_cortex.py — 运动皮层 (Motor Cortex) [v5.6 实现: 言语运动规划]

对应脑区: BA4 (初级运动皮层 M1) + BA6 (前运动皮层/SMA)
所属层级: 大脑 → 额叶 → 运动皮层

脑区标记: M1 (BA4, 口面部代表区) · SMA/pre-SMA (BA6, 序列编排) · 前岛叶(发音规划)

功能职责 (参考: 语言与大脑 §1.1 发音生理, §5.2 言语产出通路):
  - M1 (BA4): 驱动发音器官肌肉 — 唇/舌/下颌/喉 (运动小人 homunculus 口面区最大)
  - SMA (BA6): 音节序列的启动和序列化 — "运动程序"的编排
  - 前运动皮层 (BA6): 感觉引导的运动 — 听觉→运动映射 (背侧通路)
  - 前岛叶: 发音运动规划 — 将音位序列转换为运动指令

言语是人体最复杂的运动行为:
  - 每秒约 10-15 个音素
  - 超过 100 块肌肉的精密配合
  - 需要亚毫米级的时序精度

简化实现 (v5.6):
  - 使用 16 维发音特征空间 (非完整生理模型, 而是分布式表征)
  - Hebb 学习: 词向量 ↔ 发音特征 (从词频谱中无监督推导)
  - 序列编排: SMA 处理词间共发音过渡 (coarticulation)
  - 运动指令副本: efference copy → AF背侧通路 → 自我监控

在 NotMe 中的集成:
  - Broca 产出词序列 → MotorCortex.plan_sequence() → 发音计划
  - MotorCortex.efference_copy() → AF.dorsal_net → 预期听觉反馈
  - 预期 vs 实际听觉 → 发音PE → F_language (自我监控)

当前限制:
  - 不驱动真实 TTS 参数 (仍用 edge-tts 生成音频)
  - 运动计划是中间表征, 用于预测和自我监控
  - 未来版本可对接真实 articulatory synthesis

参考:
  - Guenther, F. H. (2016). Neural Control of Speech. MIT Press.
    (DIVA模型 — 言语产出的神经计算模型)
  - Graziano, M. (2006). The organization of behavioral repertoire in
    motor cortex. Annual Review of Neuroscience.
"""

import numpy as np
from typing import Optional


class MotorCortex:
    """运动皮层 — 言语发音的运动规划.

    16维发音特征空间 (简化分布式表征):
      [0]  lip_rounding     — 圆唇度 [-1=展唇, +1=圆唇]
      [1]  lip_spread       — 展唇度 [0=中性, 1=最大展唇]
      [2]  tongue_height    — 舌位高度 [0=低(a), 1=高(i/u)]
      [3]  tongue_backness  — 舌位前后 [0=前(i), 1=后(u)]
      [4]  tongue_tip       — 舌尖参与度 [0=无, 1=高(t/d/n/l)]
      [5]  velum            — 软腭 [0=抬起(口音), 1=降下(鼻音)]
      [6]  glottis          — 声门 [0=清音, 1=浊音(振动)]
      [7]  jaw              — 下颌开度 [0=闭, 1=开(a)]
      [8]  duration         — 时长 [0=短促, 1=长]
      [9]  stress           — 重音/强调 [0=弱, 1=强]
      [10] place_of_artic   — 发音部位编码 [双唇/齿龈/软腭...]
      [11] manner           — 发音方式编码 [塞/擦/鼻/边...]
      [12] voicing          — 清浊 [0=清, 1=浊]
      [13] nasal            — 鼻音性 [0=口音, 1=鼻音]
      [14] lateral          — 边音性 [0=非边音, 1=边音(l)]
      [15] rhotic           — r音性 [0=非r, 1=r音]

    注意: 这不是国际音标(IPA)的精确生理模型,
    而是从词频谱中无监督推导的分布式发音特征空间.
    实际物理含义是近似的 — 但分布式关系在 Hebb 学习中涌现.
    """

    # 发音特征维度
    ARTICULATORY_DIM = 16

    def __init__(self):
        """初始化运动皮层."""
        # 词→发音特征映射 (Hebb-like, 通过在线学习建立)
        # 存储为简单字典: {word_hash → articulatory_vector}
        self._articulatory_map: dict[int, np.ndarray] = {}

        # SMA: 序列计划缓冲区
        self._sequence_plan: list[np.ndarray] = []  # 当前计划的发音序列

        # 共发音过渡矩阵 (16x16): 发音特征间的平滑过渡权重
        self.coarticulation_weight: float = 0.3  # 过渡平滑度

        # 追踪
        self.n_plans: int = 0
        self.n_executed: int = 0

    # ================================================================
    # 发音规划
    # ================================================================

    def plan_articulation(self, word_vec: np.ndarray,
                         word: str = None) -> np.ndarray:
        """规划单个词的发音运动.

        词向量 → 发音特征空间 (分布式映射).
        初始映射通过无监督方法从词频谱中推导,
        后续通过 Hebb 学习在与实际发音反馈的互动中细化.

        Args:
            word_vec: 词向量 (64,), 音频频谱空间
            word: 词本身 (可选, 用于学习)

        Returns:
            (ARTICULATORY_DIM,) 发音特征向量
        """
        wv = np.asarray(word_vec, dtype=np.float32).ravel()

        # ---- 方法1: 已学习的映射 ----
        if word is not None:
            word_key = hash(word) % (2**31)
            if word_key in self._articulatory_map:
                return self._articulatory_map[word_key].copy()

        # ---- 方法2: 从词向量无监督推导 ----
        # 使用词向量的前16维作为初始发音特征代理
        # (词频谱的低维结构粗略编码了发音信息)
        artic = np.zeros(self.ARTICULATORY_DIM, dtype=np.float32)

        if len(wv) >= self.ARTICULATORY_DIM:
            raw = wv[:self.ARTICULATORY_DIM]
        else:
            raw = np.pad(wv, (0, self.ARTICULATORY_DIM - len(wv)))

        # 将词向量的低维结构映射到 [-1, 1] 范围 (发音特征是双极性的)
        artic = np.tanh(raw * 2.0).astype(np.float32)

        # 特殊维度处理:
        # voicing 和 nasal 应该偏向正值 (大多数语音是浊音)
        artic[6] = np.clip(artic[6], -0.3, 1.0)   # glottis/voicing
        artic[12] = np.clip(artic[12], -0.3, 1.0)  # voicing
        artic[13] = np.clip(artic[13], -0.5, 1.0)  # nasal

        # 学习: 缓存此映射 (用于未来快速查找)
        if word is not None:
            word_key = hash(word) % (2**31)
            self._articulatory_map[word_key] = artic.copy()

        self.n_plans += 1
        return artic

    def plan_sequence(self, word_vecs: list[np.ndarray],
                     words: list[str] = None) -> list[np.ndarray]:
        """SMA: 规划词序列的发音, 含共发音过渡.

        SMA (辅助运动区) 负责: 音节序列编排 + 启动 + 时序控制.
        共发音 (coarticulation): 相邻音的发音特征互相影响,
        使过渡平滑自然 (如 "ian" 中 i→a 的舌位连续变化).

        Args:
            word_vecs: 词向量列表 [(64,), ...]
            words: 词列表 (可选)

        Returns:
            [articulatory_plan, ...] 含共发音过渡的计划序列
        """
        if not word_vecs:
            return []

        # Step 1: 逐词规划原始发音
        if words and len(words) == len(word_vecs):
            raw_plans = [self.plan_articulation(wv, w)
                        for wv, w in zip(word_vecs, words)]
        else:
            raw_plans = [self.plan_articulation(wv)
                        for wv in word_vecs]

        # Step 2: 共发音平滑 (相邻计划的过渡)
        if len(raw_plans) <= 1:
            self._sequence_plan = raw_plans
            return raw_plans

        smoothed = [raw_plans[0]]
        for i in range(1, len(raw_plans)):
            prev = smoothed[-1]
            curr = raw_plans[i]

            # 线性插值: 前一个计划的部分特征"渗入"下一个
            # 模拟共发音的生理约束 (发音器官不能瞬间切换)
            blended = prev * self.coarticulation_weight + curr * (
                1.0 - self.coarticulation_weight)
            smoothed.append(blended.astype(np.float32))

        self._sequence_plan = smoothed
        return smoothed

    # ================================================================
    # 运动指令副本 (Efference Copy)
    # ================================================================

    def efference_copy(self, motor_plan: np.ndarray = None
                      ) -> np.ndarray:
        """生成运动指令副本 → 预测感觉后果.

        大脑发出运动指令的同时, 发送一份"副本"到感觉皮层.
        这用于预测行动的感觉后果, 是自我监控的基础.

        在言语中: 发音计划 → 预测自己会听到什么声音.
        这个预测与弓状束(AF)背侧通路协作:
          MotorCortex.efference_copy() → AF.dorsal_net → 预期听觉

        Args:
            motor_plan: 发音计划 (ARTICULATORY_DIM,) 或 None (用当前序列的最后一项)

        Returns:
            (64,) 预期听觉向量 (音频频谱空间)
        """
        if motor_plan is None:
            if self._sequence_plan:
                motor_plan = self._sequence_plan[-1]
            else:
                return np.zeros(64, dtype=np.float32)

        # 简化的"发音→声音"前向模型:
        # 发音特征 → 预期声学结果 (通过线性映射近似)
        mp = np.asarray(motor_plan, dtype=np.float32).ravel()
        expected_audio = np.zeros(64, dtype=np.float32)

        # 将16维发音特征映射到64维音频频谱空间
        # (这是一个简化的前向模型 — 实际DIVA模型要复杂得多)
        for i in range(min(self.ARTICULATORY_DIM, 64)):
            expected_audio[i] = float(mp[i % self.ARTICULATORY_DIM])
            if i + self.ARTICULATORY_DIM < 64:
                expected_audio[i + self.ARTICULATORY_DIM] = float(
                    mp[i] * 0.5)
            if i + 32 < 64:
                expected_audio[i + 32] = float(np.tanh(mp[i] * 2.0) * 0.3)

        # 归一化到合理范围
        norm = np.linalg.norm(expected_audio)
        if norm > 1e-8:
            expected_audio = expected_audio / norm * 0.5

        return expected_audio.astype(np.float32)

    # ================================================================
    # 在线学习
    # ================================================================

    def learn_from_feedback(self, word_vec: np.ndarray,
                           articulatory_plan: np.ndarray,
                           actual_auditory: np.ndarray,
                           learning_rate: float = 0.05):
        """从实际听觉反馈中学习, 细化发音→声音映射.

        当 Agent 听到自己说话的实际音频时,
        用实际 vs 预期的差异来更新内部前向模型.

        Args:
            word_vec: 原始词向量
            articulatory_plan: 发出的发音计划
            actual_auditory: 实际听到的音频反馈
            learning_rate: 学习率
        """
        # 计算预测误差
        expected = self.efference_copy(articulatory_plan)
        pe = actual_auditory[:64] - expected

        # 梯度下降: 调整发音特征 → 减小预测误差
        corrected_plan = articulatory_plan.copy()
        for i in range(min(self.ARTICULATORY_DIM, len(pe))):
            corrected_plan[i] += learning_rate * pe[i]

        # 裁剪到合理范围
        corrected_plan = np.clip(corrected_plan, -1.0, 1.0)

        # 更新内部映射
        # (对当前词更新发音→声音关联)
        word_key = hash(str(word_vec[:8].tobytes())) % (2**31)
        self._articulatory_map[word_key] = corrected_plan

    # ================================================================
    # 内部言语 (子 vocal speech)
    # ================================================================

    def subvocal_plan(self, word_vec: np.ndarray,
                     word: str = None) -> np.ndarray:
        """规划内部言语 (subvocal speech / inner speech).

        与 plan_articulation() 相同, 但不执行.
        用于:
          - 默读复述 (phonological loop 的 rehearsal)
          - 内部独白 (inner speech)
          - "在心里说话" — 运动计划但不激活 M1 执行

        Args:
            word_vec: 词向量
            word: 词本身

        Returns:
            发音计划 (与 plan_articulation 相同, 但标记为不执行)
        """
        return self.plan_articulation(word_vec, word)

    # ================================================================
    # 诊断
    # ================================================================

    @property
    def is_planning(self) -> bool:
        return len(self._sequence_plan) > 0

    def get_state(self) -> dict:
        """返回当前状态 (供 dashboard 使用)."""
        return {
            'n_learned_articulations': len(self._articulatory_map),
            'sequence_length': len(self._sequence_plan),
            'coarticulation_weight': float(self.coarticulation_weight),
            'n_plans': self.n_plans,
            'n_executed': self.n_executed,
        }

    def __repr__(self) -> str:
        return (f"MotorCortex(articulations={len(self._articulatory_map)}, "
                f"planning={self.is_planning})")
