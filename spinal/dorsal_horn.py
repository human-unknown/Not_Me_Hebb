"""
dorsal_horn.py — 脊髓背角 (Spinal Dorsal Horn) [v5.4]

对应脑区: 脊髓背角 (Rexed laminae I–VI)
所属层级: 脊髓 → 背角 (Level 4)
脑区标签: Lamina I (边缘层) · Lamina II (胶状质/SG) · Lamina V (WDR神经元)

功能职责:
  - 痛觉信号的第一级中枢处理站
  - 闸门控制 (Gate Control Theory, Melzack & Wall 1965)
  - 初级传入纤维整合 (Aδ/C/Aβ)
  - 中枢敏化 (Central Sensitization) — wind-up, allodynia, hyperalgesia

Rexed分层:
  - 板层I (边缘层): 投射神经元 — 伤害性信息输出中转站
  - 板层II (胶状质/SG): 抑制性中间神经元 — 闸门控制核心
  - 板层III–IV: Aβ触觉输入 — 关闭闸门
  - 板层V: WDR神经元 — 汇聚伤害+非伤害输入 → 中枢敏化位点

知觉规律:
  1. 闸门控制律 — Aβ粗纤维关闭闸门, Aδ/C细纤维打开闸门
  2. 痛觉非适应律 — 痛觉极少适应 (生存需要)
  3. 中枢致敏律 — 持续伤害输入 → WDR敏化 → allodynia/hyperalgesia
  4. 节段性抑制 — 同节段触觉抑制痛觉

在 NotMe 中的应用:
  - 接收模拟的伤害性输入 (组织损伤信号)
  - 输出闸门调控后的痛觉信号 → 上行通路
  - 中枢敏化状态追踪 → 慢性疼痛建模
  - 下行调控接收 (PAG→RVM→背角)
"""

import numpy as np
from typing import Optional, Tuple

# ============================================================
# 常量
# ============================================================

N_DH_OUTPUT = 16              # 背角闸门输出维度
N_WDR_CELLS = 8               # 广动力范围神经元数量
N_SG_CELLS = 12               # 胶状质抑制性中间神经元
GATE_OPEN_THRESHOLD = 0.3     # 闸门开启阈值
WINDUP_TAU = 20.0             # wind-up 时间常数 (步)
SENSITIZATION_DECAY = 0.995   # 中枢敏化自然衰减率


class SubstantiaGelatinosa:
    """板层II 胶状质 (SG) — 闸门控制核心.

    SG细胞 = 抑制性中间神经元 (GABA能/甘氨酸能)
    - 接收Aβ粗纤维的兴奋输入 → 抑制T细胞 (关闭闸门)
    - 接收Aδ/C细纤维的抑制输入 → 去抑制T细胞 (打开闸门)
    - 接收下行调控 (PAG→RVM→5-HT/NE) → 调制闸门阈值
    """

    def __init__(self, n_cells: int = N_SG_CELLS):
        self.n_cells = n_cells
        # SG细胞激活水平 [0, 1] — 高=抑制T细胞=闸门关闭
        self.activation: np.ndarray = np.ones(n_cells, dtype=np.float32) * 0.5
        # 下行调制信号 (来自RVM: 正=抑制, 负=易化)
        self._descending_mod: float = 0.0
        # 内源性阿片肽局部浓度
        self._opioid_tone: float = 0.1

    def process(self, abeta_input: float, adelta_c_input: float,
                descending_signal: float = 0.0) -> Tuple[np.ndarray, float]:
        """SG闸门动态.

        Args:
            abeta_input: Aβ粗纤维激活 [0, 1] (触觉/振动 → 关闭闸门)
            adelta_c_input: Aδ/C细纤维激活 [0, 1] (痛觉 → 打开闸门)
            descending_signal: 下行调控 [-1, 1] (正=抑制疼痛)

        Returns:
            (sg_activation, gate_state): SG激活 + 闸门状态 [0=开, 1=关]
        """
        # 更新下行调制EMA
        self._descending_mod = 0.8 * self._descending_mod + 0.2 * descending_signal

        # Aβ兴奋SG细胞 (关闭闸门)
        abeta_drive = abeta_input * 0.8
        # Aδ/C抑制SG细胞 (打开闸门)
        nociceptive_inhib = adelta_c_input * 0.6
        # 下行调控: 正信号 → 增强SG (关闭闸门=镇痛)
        descending_drive = max(0.0, self._descending_mod) * 0.4
        # 阿片肽基调
        opioid_boost = self._opioid_tone * 0.3

        # SG净输入
        net_input = (abeta_drive - nociceptive_inhib
                     + descending_drive + opioid_boost)

        # SG激活动态 (缓慢更新)
        target = np.clip(0.5 + net_input, 0.0, 1.0)
        self.activation = 0.85 * self.activation + 0.15 * target

        # 闸门状态: SG平均激活 → 高=闸门关闭
        gate_state = float(np.clip(np.mean(self.activation), 0.0, 1.0))

        return self.activation.copy(), gate_state

    def set_opioid_tone(self, tone: float):
        """设置局部阿片肽水平 (来自PAG→RVM下行释放)."""
        self._opioid_tone = float(np.clip(tone, 0.0, 1.0))


class WDRNeurons:
    """板层V 广动力范围神经元 (WDR) — 中枢敏化位点.

    WDR神经元接收Aβ + Aδ + C纤维的汇聚输入.
    在持续伤害性刺激下发生:
      - wind-up (逐步增强反应)
      - 长时程增强 (LTP)
      - 感受野扩大 → allodynia (触诱发痛)
    """

    def __init__(self, n_cells: int = N_WDR_CELLS):
        self.n_cells = n_cells
        # 基础兴奋性 [0, 1]
        self.excitability: np.ndarray = np.ones(n_cells, dtype=np.float32) * 0.3
        # wind-up 累积
        self._windup_accum: np.ndarray = np.zeros(n_cells, dtype=np.float32)
        # 中枢敏化水平 [0, 1]
        self.sensitization: float = 0.0
        # 敏化历史 (步数)
        self._sensitization_steps: int = 0

    def process(self, nociceptive_input: np.ndarray,
                gate_state: float) -> Tuple[np.ndarray, float]:
        """WDR处理 — 含wind-up和中枢敏化.

        Args:
            nociceptive_input: 伤害性输入 (n_cells,)
            gate_state: SG闸门状态 [0=开, 1=关]

        Returns:
            (wdr_output, sensitization_level): WDR输出 + 敏化水平
        """
        noci = np.asarray(nociceptive_input, dtype=np.float32).ravel()
        if len(noci) < self.n_cells:
            noci = np.pad(noci, (0, self.n_cells - len(noci)))[:self.n_cells]
        else:
            noci = noci[:self.n_cells]

        # 闸门调制: gate_state高 → 信号被抑制
        gated = noci * max(0.0, 1.0 - gate_state)

        # ---- Wind-up: 重复C纤维刺激 → 反应逐步增强 ----
        # 仅对持续高输入 (>0.5) 进行wind-up
        active_mask = (noci > 0.5).astype(np.float32)
        self._windup_accum = (0.95 * self._windup_accum
                              + 0.05 * active_mask * noci)
        windup_boost = np.tanh(self._windup_accum * 2.0)

        # ---- 中枢敏化: 长时间高强度输入 → 兴奋性阈值降低 ----
        if np.mean(noci) > 0.6:
            self._sensitization_steps += 1
            # 敏化增长: sigmoid (步数/tau)
            self.sensitization = float(np.clip(
                self.sensitization + 0.01 * (1.0 - self.sensitization),
                0.0, 1.0))
        else:
            # 敏化自然衰减 (缓慢)
            self.sensitization *= SENSITIZATION_DECAY
            self._sensitization_steps = max(0, self._sensitization_steps - 1)

        # 敏化放大: 敏化水平高 → 正常输入也被放大 (hyperalgesia)
        #           敏化水平极高 → 非伤害输入也触发 (allodynia)
        sensitized_gain = 1.0 + self.sensitization * 2.0

        # WDR输出 = gated信号 × (1 + windup) × sensitized_gain
        wdr_output = gated * (1.0 + windup_boost) * sensitized_gain
        wdr_output = np.clip(wdr_output, 0.0, 1.0)

        return wdr_output.astype(np.float32), self.sensitization

    def is_allodynia(self) -> bool:
        """触诱发痛: 敏化水平 > 0.7 → 非伤害触觉也被感知为疼痛."""
        return self.sensitization > 0.7

    def is_hyperalgesia(self) -> bool:
        """痛觉过敏: 敏化水平 > 0.4 → 疼痛刺激被放大."""
        return self.sensitization > 0.4


class DorsalHorn:
    """脊髓背角 — 痛觉闸门控制 + 初级整合.

    组装板层I (投射), 板层II (SG闸门), 板层V (WDR敏化).

    用法:
      dh = DorsalHorn()
      output = dh.process(
          nociceptive_input=0.8,   # C纤维伤害性输入
          abeta_input=0.2,         # Aβ触觉输入
          descending_signal=0.0,   # 来自RVM的下行调控
      )
      # output['pain_signal'] → 闸门调控后的痛觉信号
      # output['gate_state'] → 闸门开度
      # output['sensitization'] → 中枢敏化水平
    """

    def __init__(self):
        # 亚区
        self.sg = SubstantiaGelatinosa()
        self.wdr = WDRNeurons()

        # 初级传入状态
        self._adelta_fiber: float = 0.0   # Aδ快痛纤维
        self._c_fiber: float = 0.0        # C慢痛纤维
        self._abeta_fiber: float = 0.0    # Aβ触觉纤维

        # 投射神经元 (板层I) — 输出到上行通路
        self._projection: np.ndarray = np.zeros(N_DH_OUTPUT, dtype=np.float32)

        # 非适应追踪: 痛觉极少适应 (vs其他感觉)
        self._adaptation: float = 0.0     # 接近0 = 不适应

        # 内源性阿片肽 (来自PAG→RVM)
        self._endorphin_level: float = 0.1

        # 下行调控累积
        self._descending_cumulative: float = 0.0

    def process(self,
                nociceptive_input: float = 0.0,
                abeta_input: float = 0.0,
                tissue_damage: float = 0.0,
                descending_signal: float = 0.0,
                endorphin_level: float = 0.1) -> dict:
        """脊髓背角单步处理.

        Args:
            nociceptive_input: 伤害性信号强度 [0, 1] (模拟组织损伤→C纤维激活)
            abeta_input: Aβ触觉输入 [0, 1] (按摩/触摸 → 关闭闸门)
            tissue_damage: 组织损伤程度 [0, 1]
            descending_signal: 下行调控信号 [-1, 1] (来自RVM: 正=镇痛, 负=易化)
            endorphin_level: 内源性阿片肽水平 [0, 1] (来自PAG)

        Returns:
            dict with:
              'pain_signal': 闸门调控后的痛觉信号 (上行)
              'gate_state': 闸门开度 [0=全关, 1=全开]
              'sensitization': 中枢敏化水平 [0, 1]
              'allodynia': 触诱发痛标志
              'hyperalgesia': 痛觉过敏标志
              'adaptation': 痛觉适应水平 (近0=不适应)
              'fast_pain': Aδ快痛分量
              'slow_pain': C慢痛分量
              'projection': 板层I投射神经元输出
        """
        # ---- 纤维激活分解 ----
        # Aδ: 快痛, 尖锐, 定位明确 (组织损伤的快速成分)
        self._adelta_fiber = 0.3 * self._adelta_fiber + 0.7 * (
            nociceptive_input * 0.7)  # Aδ对尖锐刺激敏感
        # C: 慢痛, 灼烧, 定位模糊 (持续伤害成分 + 炎症)
        self._c_fiber = 0.5 * self._c_fiber + 0.5 * (
            nociceptive_input * 0.9 + tissue_damage * 0.3)
        # Aβ: 触觉 (非伤害)
        self._abeta_fiber = 0.2 * self._abeta_fiber + 0.8 * abeta_input

        # ---- 更新内源性阿片水平 ----
        self._endorphin_level = (0.9 * self._endorphin_level
                                 + 0.1 * endorphin_level)
        self.sg.set_opioid_tone(self._endorphin_level)

        # ---- SG闸门处理 ----
        # 综合伤害性输入 (Aδ + C)
        combined_nociceptive = 0.4 * self._adelta_fiber + 0.6 * self._c_fiber
        sg_act, gate_state = self.sg.process(
            abeta_input=self._abeta_fiber,
            adelta_c_input=combined_nociceptive,
            descending_signal=descending_signal,
        )

        # ---- WDR敏化处理 ----
        noci_vec = np.array([combined_nociceptive] * N_WDR_CELLS)
        wdr_out, sensitization = self.wdr.process(noci_vec, gate_state)

        # ---- 板层I投射神经元: 整合WDR输出 ----
        # 投射到上行通路 (脊髓丘脑束等)
        wdr_padded = np.pad(wdr_out, (0, max(0, N_DH_OUTPUT - len(wdr_out))))[:N_DH_OUTPUT]
        self._projection = 0.7 * self._projection + 0.3 * wdr_padded

        # ---- 痛觉非适应律: 痛觉极少适应 ----
        # adapt_factor 接近0 → 信号不衰减
        # 仅在高强度持续输入时微弱适应 (生存保护)
        if nociceptive_input > 0.8:
            self._adaptation = min(0.05, self._adaptation + 0.001)
        else:
            self._adaptation = max(0.0, self._adaptation - 0.002)
        adapted_signal = self._projection * (1.0 - self._adaptation * 0.5)

        # ---- 快痛 vs 慢痛分离 ----
        fast_pain = self._adelta_fiber * max(0.0, 1.0 - gate_state)  # Aδ经闸门
        slow_pain = self._c_fiber * max(0.0, 1.0 - gate_state)       # C经闸门

        return {
            'pain_signal': adapted_signal.astype(np.float32),
            'gate_state': gate_state,
            'sg_activation': sg_act,
            'sensitization': sensitization,
            'allodynia': self.wdr.is_allodynia(),
            'hyperalgesia': self.wdr.is_hyperalgesia(),
            'adaptation': self._adaptation,
            'fast_pain': float(fast_pain),
            'slow_pain': float(slow_pain),
            'projection': self._projection.copy(),
            'abeta_fiber': float(self._abeta_fiber),
            'adelta_fiber': float(self._adelta_fiber),
            'c_fiber': float(self._c_fiber),
            'endorphin_level': self._endorphin_level,
        }

    def receive_descending(self, pag_rvm_signal: float,
                          opioid_level: float = 0.0):
        """接收下行调控信号 (来自PAG→RVM通路).

        Args:
            pag_rvm_signal: PAG→RVM下行调控 [-1, 1]
                           正=抑制疼痛 (OFF细胞激活), 负=易化疼痛 (ON细胞激活)
            opioid_level: 内源性阿片肽释放量 [0, 1]
        """
        self._descending_cumulative = (
            0.8 * self._descending_cumulative + 0.2 * pag_rvm_signal)
        self._endorphin_level = (0.9 * self._endorphin_level
                                 + 0.1 * opioid_level)

    # ================================================================
    # 预测编码接口
    # ================================================================

    def get_prediction(self) -> np.ndarray:
        """返回背角对上行痛觉信号的预测."""
        return self._projection.copy()

    def compute_prediction_error(self, actual: np.ndarray) -> np.ndarray:
        """计算痛觉预测误差 (用于上行更新)."""
        actual = np.asarray(actual, dtype=np.float32).ravel()
        if len(actual) < N_DH_OUTPUT:
            actual = np.pad(actual, (0, N_DH_OUTPUT - len(actual)))
        return (actual[:N_DH_OUTPUT] - self._projection).astype(np.float32)

    def reset_sensitization(self):
        """重置中枢敏化 (模拟完全恢复)."""
        self.wdr.sensitization = 0.0
        self.wdr._windup_accum = np.zeros(N_WDR_CELLS, dtype=np.float32)
        self.wdr._sensitization_steps = 0
