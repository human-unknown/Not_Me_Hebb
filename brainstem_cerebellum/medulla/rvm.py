"""
rvm.py — 延髓头端腹内侧区 (Rostral Ventromedial Medulla) [v5.4]

对应脑区: 延髓头端腹内侧区 (RVM)
所属层级: 脑干 → 延髓 (Level 3)
脑区标签: RVM (大缝核 · 旁巨细胞网状核)

功能职责:
  - PAG与脊髓背角之间的关键中继站
  - 下行抑制和下行易化的最终共同输出节点
  - 含三种功能不同的神经元: OFF细胞, ON细胞, 中性细胞

三种神经元类型:
  - OFF细胞: 伤害性反射时放电停止 → 抑制伤害性传递 → 镇痛
  - ON细胞: 伤害性反射时放电增加 → 易化伤害性传递 → 加重疼痛
  - 中性细胞: 无明显变化

阿片类药物作用机制:
  - 抑制ON细胞 (减少易化)
  - 激活OFF细胞 (增强抑制)
  → 净效应 = 镇痛

慢性疼痛:
  - ON细胞持续活跃 → 下行易化 → 痛觉敏化维持
  - OFF细胞功能下降 → 下行抑制不足

知觉规律:
  1. OFF/ON平衡律 — 镇痛vs易化的动态平衡
  2. 阿片调制律 — μ受体激活 → OFF↑ ON↓
  3. 慢性痛易化律 — ON细胞持续活跃 → 痛觉维持

在 NotMe 中的应用:
  - 接收PAG下行信号 → 中继到脊髓背角
  - OFF/ON细胞动态建模
  - 输出调制信号到 DorsalHorn
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

N_OFF_CELLS = 6               # OFF细胞数量
N_ON_CELLS = 6                # ON细胞数量
N_NEUTRAL_CELLS = 4           # 中性细胞数量
N_RVM_OUTPUT = 8              # RVM输出维度


class OffOnDynamics:
    """OFF/ON细胞动态 — 下行调控的最终共同输出.

    OFF细胞: 放电停止于伤害性反射 → 抑制 = 镇痛
    ON细胞: 放电增加于伤害性反射 → 易化 = 加重疼痛

    平衡决定最终下行信号的净效应.
    """

    def __init__(self, n_off: int = N_OFF_CELLS, n_on: int = N_ON_CELLS):
        self.n_off = n_off
        self.n_on = n_on

        # 细胞激活水平 [0, 1]
        self.off_activation: np.ndarray = np.ones(n_off, dtype=np.float32) * 0.3
        self.on_activation: np.ndarray = np.ones(n_on, dtype=np.float32) * 0.3

        # 自发放电率 (基线)
        self.off_baseline: float = 0.3
        self.on_baseline: float = 0.3

        # μ阿片受体敏感度
        self.off_opioid_sensitivity: float = 0.8   # OFF细胞对阿片敏感 → 激活
        self.on_opioid_sensitivity: float = 0.7    # ON细胞对阿片敏感 → 抑制

        # 慢性疼痛 → ON细胞持续活跃
        self._chronic_pain_shift: float = 0.0

    def process(self, pag_signal: float, opioid_level: float,
                chronic_pain: float = 0.0) -> dict:
        """OFF/ON细胞动态更新.

        Args:
            pag_signal: PAG下行信号 [-1, 1] (正=镇痛驱动)
            opioid_level: 内源性阿片水平 [0, 1]
            chronic_pain: 慢性疼痛程度 [0, 1]

        Returns:
            dict with:
              'net_modulation': 净下行调制 [-1, 1] (正=镇痛)
              'off_activation': OFF细胞激活
              'on_activation': ON细胞激活
              'off_on_ratio': OFF/ON比值
        """
        # ---- 慢性疼痛偏移: ON细胞持续活跃, OFF细胞功能下降 ----
        self._chronic_pain_shift = (0.9 * self._chronic_pain_shift
                                    + 0.1 * chronic_pain)

        # ---- OFF细胞: PAG信号(正) → 激活OFF细胞 ----
        # 阿片去抑制: 抑制GABA中间神经元 → OFF细胞被释放
        off_drive = (
            0.5 * max(0.0, pag_signal) +          # PAG兴奋驱动
            0.3 * opioid_level * self.off_opioid_sensitivity +  # 阿片去抑制
            0.2 * self.off_baseline               # 自发放电
        )
        # 慢性疼痛 → OFF功能下降
        off_drive *= max(0.3, 1.0 - self._chronic_pain_shift * 0.7)

        off_target = np.clip(off_drive, 0.0, 1.0)
        self.off_activation = (0.8 * self.off_activation
                               + 0.2 * off_target)

        # ---- ON细胞: PAG信号(负) + 慢性疼痛 → 激活ON细胞 ----
        on_drive = (
            0.3 * max(0.0, -pag_signal) +          # PAG易化驱动
            0.1 * self.on_baseline +                # 自发放电
            0.4 * self._chronic_pain_shift +        # 慢性疼痛 → ON↑
            0.2 * (1.0 - opioid_level)              # 低阿片 → ON不受抑制
        )
        # 阿片抑制ON细胞
        on_drive *= max(0.2, 1.0 - opioid_level * self.on_opioid_sensitivity)

        on_target = np.clip(on_drive, 0.0, 1.0)
        self.on_activation = (0.8 * self.on_activation
                              + 0.2 * on_target)

        # ---- 净调制: OFF - ON ----
        off_strength = float(np.mean(self.off_activation))
        on_strength = float(np.mean(self.on_activation))

        # 净调制: [-1, 1], 正=镇痛, 负=易化
        net_modulation = float(np.clip(
            off_strength - on_strength, -1.0, 1.0))

        # OFF/ON比值: >1 = 镇痛主导, <1 = 易化主导
        off_on_ratio = (off_strength + 1e-8) / (on_strength + 1e-8)

        return {
            'net_modulation': net_modulation,
            'off_activation': self.off_activation.copy(),
            'on_activation': self.on_activation.copy(),
            'off_on_ratio': off_on_ratio,
            'off_strength': off_strength,
            'on_strength': on_strength,
        }


class RostralVentromedialMedulla:
    """RVM — PAG→脊髓的关键中继 + OFF/ON平衡.

    组装OFF/ON细胞动态, 接收PAG信号, 输出到脊髓背角.

    用法:
      rvm = RostralVentromedialMedulla()
      output = rvm.process(pag_signal=0.5, opioid_level=0.6)
      # output['descending_signal'] → 到 DorsalHorn
      # output['off_on_ratio'] → OFF/ON平衡
    """

    def __init__(self):
        self.off_on = OffOnDynamics()

        # 下行输出状态
        self._descending_signal: float = 0.0
        self._serotonin_tone: float = 0.3      # 5-HT基调 (大缝核)
        self._norepinephrine_tone: float = 0.2  # NE基调 (来自蓝斑)

    def process(self,
                pag_signal: float = 0.0,
                opioid_level: float = 0.1,
                chronic_pain: float = 0.0,
                pain_intensity: float = 0.0,
                arousal: float = 0.5) -> dict:
        """RVM单步处理.

        Args:
            pag_signal: PAG下行信号 [-1, 1]
            opioid_level: 内源性阿片水平 [0, 1]
            chronic_pain: 慢性疼痛程度 [0, 1]
            pain_intensity: 当前疼痛强度 [0, 1] (反馈)
            arousal: 唤醒度 [0, 1]

        Returns:
            dict with:
              'descending_signal': 到脊髓的下行信号 [-1, 1]
              'off_on_ratio': OFF/ON平衡
              'serotonin_tone': 5-HT基调
              'norepinephrine_tone': NE基调
              'is_inhibitory': 当前为抑制模式
        """
        # ---- OFF/ON动态 ----
        oo_result = self.off_on.process(
            pag_signal=pag_signal,
            opioid_level=opioid_level,
            chronic_pain=chronic_pain,
        )

        # ---- 5-HT (血清素) 基调: 来自大缝核 ----
        # 急性痛 → 5-HT↑ (抑制), 慢性痛 → 5-HT可转为易化
        if chronic_pain > 0.5:
            # 慢性痛: 5-HT系统可转为易化 (5-HT2A/3受体)
            self._serotonin_tone = float(np.clip(
                0.9 * self._serotonin_tone + 0.1 * (0.7 * chronic_pain),
                0.0, 0.8))
        else:
            # 急性: 5-HT主要抑制
            self._serotonin_tone = float(np.clip(
                0.9 * self._serotonin_tone + 0.1 * (0.7 * pain_intensity),
                0.0, 0.6))

        # ---- NE (去甲肾上腺素) 基调: 来自蓝斑 ----
        # α2受体 → 镇痛, 受唤醒度调制
        self._norepinephrine_tone = float(np.clip(
            0.9 * self._norepinephrine_tone + 0.1 * arousal * 0.5,
            0.0, 0.7))

        # ---- 综合下行信号 ----
        # OFF/ON净调制 + 5-HT成分 + NE成分
        raw_signal = (
            0.55 * oo_result['net_modulation'] +
            0.25 * (self._serotonin_tone if chronic_pain < 0.5
                    else -self._serotonin_tone * 0.5) +  # 慢性: 5-HT部分易化
            0.20 * self._norepinephrine_tone              # NE总是抑制性
        )

        self._descending_signal = float(np.clip(raw_signal, -1.0, 1.0))

        return {
            'descending_signal': self._descending_signal,
            'off_on_ratio': oo_result['off_on_ratio'],
            'off_activation': oo_result['off_activation'],
            'on_activation': oo_result['on_activation'],
            'off_strength': oo_result['off_strength'],
            'on_strength': oo_result['on_strength'],
            'serotonin_tone': self._serotonin_tone,
            'norepinephrine_tone': self._norepinephrine_tone,
            'is_inhibitory': self._descending_signal > 0.1,
            'is_facilitatory': self._descending_signal < -0.1,
        }

    def reset(self):
        """重置RVM状态."""
        self._descending_signal = 0.0
        self._serotonin_tone = 0.3
        self._norepinephrine_tone = 0.2
        self.off_on = OffOnDynamics()
