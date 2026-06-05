"""
hypothalamus.py — 下丘脑 (Hypothalamus) [v5.5]

对应脑区: 下丘脑诸核团 (视前区、室旁核、弓状核、外侧下丘脑等)
所属层级: 大脑 → 边缘系统 → 下丘脑

功能职责:
  - 稳态调节中枢 — 体温、饥饿、口渴、睡眠、昼夜节律
  - 内分泌控制 — HPA轴 (CRH→ACTH→皮质醇)
  - 自主神经调控 — 交感/副交感平衡
  - 动机行为 — 摄食、饮水、攻击
  - 应激反应 — 慢性应激负荷

在 NotMe 中的实现 (v5.5):
  1. SetpointModel: 动态调定点 (昼夜节律 + 异稳态偏移)
  2. DriveSystem: 驱力计算 (偏离 setpoint → 驱力强度)
  3. Hypothalamus: 顶层编排 — 稳态调节 + HPA轴 + 自主神经平衡

设计参考:
  - Sterling, P. (2012). Allostasis: A model of predictive regulation.
  - Saper, C. B., & Lowell, B. B. (2014). The hypothalamus.
  - McEwen, B. S. (1998). Stress, adaptation, and disease: Allostasis and allostatic load.
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

N_BODY_DIMS = 9              # BodyVector 维度 (text mode, v5.4)

# 各维度的驱力敏感度 (sensitivity): 偏离 setpoint 的单位驱力响应
# b[0]社交 b[1]能量 b[2]压力 b[3]新颖性 b[4]专注 b[5]视觉 b[6]听觉 b[7]认知 b[8]痛觉
DEFAULT_SENSITIVITIES = np.array(
    [0.8, 1.0, 1.5, 0.6, 0.7, 0.5, 0.5, 0.8, 1.2], dtype=np.float32)

# ============================================================
# 调定点模型
# ============================================================

class SetpointModel:
    """动态调定点 — 基础调定点 + 昼夜节律 + 异稳态负荷偏移.

    调定点不是固定的——它们随昼夜节律、应激水平、和累积身体偏离而动态变化。
    这体现了 allostasis (异稳态): 大脑根据预测性需求调整稳态目标。

    用法:
      sp = SetpointModel()
      shifted = sp.process(body_setpoints, time_of_day=0.5, allostatic_load=0.2,
                           stress_level=0.3)
    """

    def __init__(self):
        # 昼夜节律相位 [0, 2π]
        self._circadian_phase: float = 0.0
        self._step_counter: int = 0

        # 异稳态负荷 EMA (累积身体偏离)
        self._allostatic_load_ema: float = 0.0

    def process(self,
                base_setpoints: np.ndarray,
                time_of_day: float = 0.5,
                allostatic_load: float = 0.0,
                stress_level: float = 0.0) -> np.ndarray:
        """计算当前步的动态调定点.

        Args:
            base_setpoints: 基础调定点 (9,) — 来自 BodyVector.setpoints
            time_of_day: 归一化时间 [0, 1] (0=午夜, 0.25=6am, 0.5=正午, 0.75=6pm)
            allostatic_load: 当前异稳态负荷 [0, 1]
            stress_level: 当前应激水平 [0, 1]

        Returns:
            shifted_setpoints: 动态调定点 (9,)
        """
        base = np.asarray(base_setpoints, dtype=np.float32).copy()

        # ---- 昼夜节律调制 ----
        # 用正弦波模拟 ~24h 周期
        phase_rad = 2.0 * np.pi * time_of_day
        # cos(phase): +1 在午夜/正午, -1 在 6am/6pm
        circadian_cos = float(np.cos(phase_rad))
        # sin(phase): +1 在 6am, -1 在 6pm
        circadian_sin = float(np.sin(phase_rad))

        # b[4] 专注/警觉: 白天↑ 夜间↓ (峰值在正午附近, sin(2π×0.25)=1)
        focus_shift = 0.15 * circadian_sin
        base[4] = float(np.clip(base[4] + focus_shift, 0.05, 0.95))

        # b[2] 压力/疲劳: 夜间↑ (褪黑素时段)
        stress_shift = -0.08 * circadian_sin  # 夜间 (sin<0) → ↑ stress setpoint
        base[2] = float(np.clip(base[2] + stress_shift, 0.0, 0.5))

        # b[1] 能量: 活动期↑ (白天需要更多能量)
        energy_shift = 0.05 * circadian_sin
        base[1] = float(np.clip(base[1] + energy_shift, 0.3, 0.95))

        # b[0] 社交需求: 白天↑ 夜间↓
        social_shift = 0.1 * circadian_sin
        base[0] = float(np.clip(base[0] + social_shift, 0.3, 0.95))

        # ---- 异稳态负荷偏移 ----
        # 高异稳态负荷 → 调定点向保守方向偏移 (保存能量, 减少活动)
        if allostatic_load > 0.4:
            allo_shift = (allostatic_load - 0.4) * 0.3
            # 降低活动相关调定点: 社交↓, 能量保存↑, 认知负荷↓
            base[0] = float(np.clip(base[0] - allo_shift * 0.5, 0.2, 1.0))
            base[1] = float(np.clip(base[1] - allo_shift * 0.3, 0.2, 1.0))
            base[7] = float(np.clip(base[7] - allo_shift * 0.4, 0.1, 1.0))

        # ---- 应激水平偏移 ----
        # 高应激 → 痛觉调定点偏移 (应激诱导痛觉过敏)
        if stress_level > 0.5:
            pain_shift = (stress_level - 0.5) * 0.3
            base[8] = float(np.clip(base[8] + pain_shift, 0.0, 0.6))

        # 更新内部状态
        self._allostatic_load_ema = 0.95 * self._allostatic_load_ema + 0.05 * allostatic_load
        self._step_counter += 1

        return base.astype(np.float32)

    @property
    def allostatic_load_ema(self) -> float:
        return self._allostatic_load_ema

    def reset(self):
        self._circadian_phase = 0.0
        self._step_counter = 0
        self._allostatic_load_ema = 0.0


# ============================================================
# 驱力系统
# ============================================================

class DriveSystem:
    """从调定点偏离计算驱力强度.

    每维的驱力 = sensitivity_i × (b_i - setpoint_i)² × sign
    总驱力 = 加权和, 用于行动动机和行为激活。

    用法:
      ds = DriveSystem()
      drives = ds.compute_drive(body_b, setpoints)
    """

    def __init__(self, sensitivities: Optional[np.ndarray] = None):
        if sensitivities is None:
            self.sensitivities = DEFAULT_SENSITIVITIES.copy()
        else:
            self.sensitivities = np.asarray(sensitivities, dtype=np.float32)

    def compute_drive(self, body: np.ndarray,
                      setpoints: np.ndarray) -> np.ndarray:
        """计算每维的驱力信号.

        Args:
            body: 当前身体状态 (M,)
            setpoints: 当前调定点 (M,)

        Returns:
            drive_vector: (M,) 带符号驱力 (正=需要增加, 负=需要减少)
        """
        b = np.asarray(body, dtype=np.float32).ravel()
        sp = np.asarray(setpoints, dtype=np.float32).ravel()
        n = min(len(b), len(sp), len(self.sensitivities))

        drives = np.zeros(n, dtype=np.float32)
        for i in range(n):
            dev = b[i] - sp[i]
            # 平方驱力 × 敏感度 × 偏差符号
            drives[i] = float(self.sensitivities[i] * (dev ** 2) * np.sign(dev))

        return drives

    def compute_total_drive(self, drive_vector: np.ndarray) -> float:
        """计算总驱力 (allostatic load 的瞬时指标)."""
        return float(np.sum(np.abs(drive_vector)))

    def compute_drive_labels(self, drive_vector: np.ndarray) -> dict:
        """将驱力向量映射为人类可读的驱力标签."""
        labels = {
            0: 'social', 1: 'energy', 2: 'rest',
            3: 'explore', 4: 'focus', 5: 'visual',
            6: 'auditory', 7: 'cognitive', 8: 'pain_relief',
        }
        result = {}
        for i, d in enumerate(drive_vector):
            if i in labels:
                result[labels[i]] = float(d)
        return result


# ============================================================
# 下丘脑 — 顶层稳态调节器
# ============================================================

class Hypothalamus:
    """下丘脑稳态调节器 — 整合调定点、驱力、HPA轴、自主神经平衡.

    用法:
      hypo = Hypothalamus()
      result = hypo.process(
          body_vector=body,
          time_of_day=0.5,
          stress_level=0.2,
          arousal=0.5,
      )
      # result['drives'] → 各维驱力
      # result['hpa_activation'] → HPA轴激活 [0, 1]
      # result['autonomic_balance'] → 交感/副交感比
    """

    def __init__(self):
        self.setpoint_model = SetpointModel()
        self.drive_system = DriveSystem()

        # HPA 轴状态
        self._crh_level: float = 0.05     # CRH (促皮质素释放激素) EMA
        self._cortisol_level: float = 0.1  # 皮质醇 EMA

        # 自主神经平衡: >0 = 交感主导, <0 = 副交感主导
        self._autonomic_balance: float = 0.0

        # 驱力历史
        self._drive_history: list[np.ndarray] = []
        self._total_drive_history: list[float] = []

    def process(self,
                body_vector=None,
                time_of_day: float = 0.5,
                stress_level: float = 0.0,
                arousal: float = 0.5) -> dict:
        """单步稳态调节.

        Args:
            body_vector: BodyVector 对象 (含 b, setpoints, decays)
            time_of_day: 归一化时间 [0, 1]
            stress_level: 当前应激水平 [0, 1]
            arousal: 当前唤醒度 [0, 1]

        Returns:
            dict with:
              'shifted_setpoints': 动态调定点 (9,)
              'drives': 驱力向量 (9,)
              'total_drive': 总驱力标量
              'drive_labels': 驱力标签映射
              'hpa_activation': HPA轴激活 [0, 1]
              'cortisol': 皮质醇水平 [0, 1]
              'autonomic_balance': 自主神经平衡 [-1, 1]
              'regulatory_urgency': 调节紧迫度 [0, 1]
        """
        if body_vector is None:
            # 回退: 使用默认值
            body_b = np.ones(N_BODY_DIMS, dtype=np.float32) * 0.5
            base_setpoints = body_b.copy()
        else:
            body_b = body_vector.b.copy()
            base_setpoints = body_vector.setpoints.copy()

        # ---- 1. 动态调定点 ----
        total_drive_ema = (self._total_drive_history[-1]
                           if self._total_drive_history else 0.0)
        shifted_sp = self.setpoint_model.process(
            base_setpoints=base_setpoints,
            time_of_day=time_of_day,
            allostatic_load=total_drive_ema,
            stress_level=stress_level,
        )

        # ---- 2. 驱力计算 ----
        drives = self.drive_system.compute_drive(body_b, shifted_sp)
        total_drive = self.drive_system.compute_total_drive(drives)

        self._drive_history.append(drives.copy())
        self._total_drive_history.append(total_drive)
        if len(self._drive_history) > 100:
            self._drive_history = self._drive_history[-100:]
            self._total_drive_history = self._total_drive_history[-100:]

        # ---- 3. HPA 轴 (CRH → ACTH → 皮质醇) ----
        # CRH 分泌: 驱力↑ + 应激↑ → CRH↑
        crh_drive = 0.6 * total_drive + 0.4 * stress_level
        self._crh_level = float(np.clip(
            0.9 * self._crh_level + 0.1 * crh_drive, 0.0, 1.0))

        # 皮质醇: 慢速跟随 CRH (生理延迟 ~20-30min, 此处用 EMA 近似)
        self._cortisol_level = float(np.clip(
            0.95 * self._cortisol_level + 0.05 * self._crh_level, 0.0, 1.0))

        # HPA 激活度 = CRH 和皮质醇的混合
        hpa_activation = float(np.clip(
            0.4 * self._crh_level + 0.6 * self._cortisol_level, 0.0, 1.0))

        # ---- 4. 自主神经平衡 ----
        # 交感驱动: 高唤醒 + 高应激 + 高驱力 → 战斗/逃跑
        symp_drive = 0.4 * arousal + 0.3 * stress_level + 0.3 * total_drive
        # 副交感驱动: 低唤醒 + 休息状态 → 休养/消化
        parasymp_drive = 0.6 * (1.0 - arousal) + 0.4 * (1.0 - stress_level)

        # 平衡: [-1, 1], 正=交感主导, 负=副交感主导
        self._autonomic_balance = float(np.clip(
            symp_drive - parasymp_drive, -1.0, 1.0))

        # ---- 5. 调节紧迫度 ----
        # 高驱力 + 高应激 → 必须行动
        regulatory_urgency = float(np.clip(
            0.5 * total_drive + 0.3 * stress_level + 0.2 * arousal,
            0.0, 1.0))

        return {
            'shifted_setpoints': shifted_sp,
            'drives': drives,
            'total_drive': total_drive,
            'drive_labels': self.drive_system.compute_drive_labels(drives),
            'hpa_activation': hpa_activation,
            'crh': self._crh_level,
            'cortisol': self._cortisol_level,
            'autonomic_balance': self._autonomic_balance,
            'sympathetic_dominant': self._autonomic_balance > 0.2,
            'parasympathetic_dominant': self._autonomic_balance < -0.2,
            'regulatory_urgency': regulatory_urgency,
        }

    def reset(self):
        """重置下丘脑状态."""
        self.setpoint_model.reset()
        self._crh_level = 0.05
        self._cortisol_level = 0.1
        self._autonomic_balance = 0.0
        self._drive_history = []
        self._total_drive_history = []
