"""
somatosensory.py — 体感皮层 (Somatosensory Cortex) [v5.4]

对应脑区: BA3, BA1, BA2 (初级体感皮层 S1) + S2 (次级体感皮层)
所属层级: 大脑 → 顶叶 → 体感皮层
脑区标签: S1 (BA3a/3b/1/2) · S2 (顶叶岛盖) · PPC (后顶叶)

功能职责:
  BA3a: 肌梭输入 (本体感觉)
  BA3b: 皮肤慢适应触觉 (纹理、形状)
  BA1:  皮肤快适应触觉 (振动、滑动)
  BA2:  深部组织 (关节位置、压力) + 疼痛定位
  S2:   感觉整合 + 认知评估 + 疼痛强度编码

v5.4 新增痛觉编码:
  - 接收丘脑VPL/VPM的痛觉中继信号
  - 编码疼痛的定位、强度、时间特征 (感觉-辨别维度)
  - 将疼痛信息传递到岛叶 (内感受) 和 ACC (情感)
  - 触觉-痛觉交互 (Aβ触觉抑制疼痛的皮层对应)

知觉规律:
  1. 痛觉定位律 — S1编码"在哪里, 有多痛"
  2. 感觉侏儒律 — 皮层面积 ∝ 感受器密度
  3. 侧抑制律 — 增强疼痛边界对比
  4. 两点辨阈律 — 空间分辨率由感受野密度决定

在 NotMe 中的应用:
  - 接收丘脑痛觉中继 → 编码感觉-辨别维度
  - 触觉编码: 环境反馈中的触觉成分 → 体感向量
  - 本体感觉: BodyVector的身体状态读出
  - 输出到岛叶 + ACC (分布式痛觉网络)
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

N_S1_OUTPUT = 16              # S1输出维度
N_S2_OUTPUT = 12              # S2输出维度
N_TOUCH_CHANNELS = 8          # 触觉通道数
N_PAIN_S1_CHANNELS = 8        # S1痛觉通道数

# 两点辨阈 (mm) — 不同身体部位的空间分辨率
TWO_POINT_THRESHOLDS = {
    'fingertip': 2.0,
    'lip': 2.0,
    'palm': 10.0,
    'forearm': 40.0,
    'back': 50.0,
}


class PrimarySomatosensory:
    """S1 — 初级体感皮层 (BA3a/3b/1/2).

    感觉侏儒图: 手/脸占据大部分皮层面积.
    痛觉编码: 定位、强度、性质 (感觉-辨别维度).
    """

    def __init__(self, n_output: int = N_S1_OUTPUT):
        self.n_output = n_output

        # 体感拓扑图 (simplified homunculus)
        self._body_map: np.ndarray = np.zeros(n_output, dtype=np.float32)

        # 触觉状态
        self._touch_state: np.ndarray = np.zeros(N_TOUCH_CHANNELS, dtype=np.float32)
        # 痛觉定位表征
        self._pain_localization: np.ndarray = np.zeros(N_PAIN_S1_CHANNELS, dtype=np.float32)

        # 侧抑制参数
        self._lateral_inhibition_strength: float = 0.3

        # 触觉-痛觉交互 (Aβ触觉抑制疼痛的皮层对应)
        self._touch_pain_interaction: float = 0.0

    def process(self,
                thalamic_pain: np.ndarray,
                thalamic_touch: Optional[np.ndarray] = None,
                body_vector: Optional[np.ndarray] = None,
                gate_state: float = 0.5) -> dict:
        """S1体感处理.

        Args:
            thalamic_pain: 丘脑痛觉中继 (来自VPL)
            thalamic_touch: 丘脑触觉中继 (来自VPL, 可选)
            body_vector: 身体状态向量
            gate_state: 脊髓闸门状态 [0=关, 1=开]

        Returns:
            dict with pain_location, touch_state, body_map
        """
        pain = np.asarray(thalamic_pain, dtype=np.float32).ravel()

        # ---- 痛觉定位编码 ----
        # S1将丘脑的痛觉信号映射到体感拓扑
        pain_padded = np.pad(pain, (0, max(0, N_PAIN_S1_CHANNELS - len(pain))))
        pain_input = pain_padded[:N_PAIN_S1_CHANNELS]

        # 侧抑制: 增强激活最强的通道, 抑制邻近通道
        pain_lateral = self._apply_lateral_inhibition(pain_input)

        # 痛觉定位EMA
        self._pain_localization = (0.6 * self._pain_localization
                                   + 0.4 * pain_lateral)

        # ---- 触觉编码 ----
        if thalamic_touch is not None:
            touch = np.asarray(thalamic_touch, dtype=np.float32).ravel()
            touch_padded = np.pad(touch, (0, max(0, N_TOUCH_CHANNELS - len(touch))))
            self._touch_state = (0.5 * self._touch_state
                                 + 0.5 * touch_padded[:N_TOUCH_CHANNELS])
        else:
            self._touch_state *= 0.9  # 衰减

        # ---- 触觉-痛觉交互 ----
        # Aβ触觉激活 → 抑制痛觉感知 (皮层层面的闸门对应)
        touch_strength = float(np.mean(np.abs(self._touch_state)))
        self._touch_pain_interaction = float(np.clip(
            touch_strength * 0.5, 0.0, 0.5))

        # 触觉抑制痛觉感知
        pain_perceived = self._pain_localization * max(
            0.0, 1.0 - self._touch_pain_interaction)

        # ---- 身体图式 (身体状态 → 空间参考) ----
        if body_vector is not None:
            bv = np.asarray(body_vector, dtype=np.float32).ravel()
            n_bv = min(len(bv), self.n_output)
            body_padded = np.pad(bv, (0, max(0, self.n_output - len(bv))))
            self._body_map = (0.8 * self._body_map
                              + 0.2 * body_padded[:self.n_output])
        else:
            self._body_map *= 0.95

        return {
            'pain_localization': pain_perceived,
            'pain_intensity': float(np.mean(np.abs(pain_perceived))),
            'touch_state': self._touch_state.copy(),
            'touch_pain_interaction': self._touch_pain_interaction,
            'body_map': self._body_map.copy(),
            'pain_raw': pain_input,
        }

    def _apply_lateral_inhibition(self, signal: np.ndarray) -> np.ndarray:
        """侧抑制: 增强峰值, 抑制邻道.

        激活最强的通道抑制邻近通道 → 增强疼痛边界对比.
        """
        result = signal.copy()
        n = len(signal)
        for i in range(n):
            # 每个通道抑制左右邻居
            inhibition = 0.0
            if i > 0:
                inhibition += signal[i] * self._lateral_inhibition_strength * 0.5
            if i < n - 1:
                inhibition += signal[i] * self._lateral_inhibition_strength * 0.5
            # 邻居被抑制
            if i > 0:
                result[i - 1] -= inhibition * 0.3
            if i < n - 1:
                result[i + 1] -= inhibition * 0.3
        return np.clip(result, 0.0, 1.0)


class SecondarySomatosensory:
    """S2 — 次级体感皮层 (顶叶岛盖).

    功能:
      - 感觉整合: 触觉 + 痛觉 + 温度觉的多模态融合
      - 疼痛强度编码: 对疼痛强度的精细辨别
      - 认知评估: 疼痛的意义和威胁评估
      - 触觉学习: 触觉模式的识别和记忆
    """

    def __init__(self, n_output: int = N_S2_OUTPUT):
        self.n_output = n_output

        # 整合表征
        self._integrated: np.ndarray = np.zeros(n_output, dtype=np.float32)
        # 疼痛强度追踪
        self._pain_intensity_ema: float = 0.0
        # 威胁评估
        self._threat_assessment: float = 0.0

        # 韦伯分数 (痛觉JND ≈ 0.1-0.3, 高于触觉的0.02-0.05)
        self._weber_fraction: float = 0.15

    def process(self, s1_output: dict,
                pain_intensity: float = 0.0,
                arousal: float = 0.5,
                novelty: float = 0.0) -> dict:
        """S2感觉整合 + 疼痛强度编码.

        Args:
            s1_output: S1输出 dict
            pain_intensity: 疼痛强度 [0, 1]
            arousal: 唤醒度 [0, 1]
            novelty: 新颖性 [0, 1]

        Returns:
            dict with integrated, intensity, threat
        """
        # ---- 感觉整合: S1输出融合 ----
        pain_loc = s1_output.get('pain_localization',
                                 np.zeros(N_PAIN_S1_CHANNELS, dtype=np.float32))
        touch = s1_output.get('touch_state',
                              np.zeros(N_TOUCH_CHANNELS, dtype=np.float32))
        body_map = s1_output.get('body_map',
                                 np.zeros(N_S1_OUTPUT, dtype=np.float32))

        # 多模态融合
        integrated_raw = np.concatenate([
            pain_loc[:4],
            touch[:4],
            body_map[:4],
        ]).astype(np.float32)

        if len(integrated_raw) < self.n_output:
            integrated_raw = np.pad(integrated_raw,
                                    (0, self.n_output - len(integrated_raw)))
        self._integrated = (0.7 * self._integrated
                            + 0.3 * integrated_raw[:self.n_output])

        # ---- 韦伯定律 (变式): 疼痛JND ----
        # ΔI/I = k, k_pain ≈ 0.15
        # 当前强度下的可觉察差异
        jnd = self._weber_fraction * pain_intensity + 0.02  # 最小基线

        # 疼痛强度EMA
        self._pain_intensity_ema = (0.7 * self._pain_intensity_ema
                                    + 0.3 * pain_intensity)

        # ---- 威胁评估 ----
        # 疼痛的认知评估: 这个伤害有多危险?
        # 高唤醒 + 高新颖 + 高强度 → 高威胁
        self._threat_assessment = float(np.clip(
            0.7 * self._threat_assessment
            + 0.3 * (0.5 * pain_intensity
                     + 0.3 * arousal
                     + 0.2 * novelty),
            0.0, 1.0))

        return {
            'integrated': self._integrated.copy(),
            'pain_intensity_ema': self._pain_intensity_ema,
            'jnd': jnd,
            'weber_fraction': self._weber_fraction,
            'threat_assessment': self._threat_assessment,
        }


class SomatosensoryCortex:
    """体感皮层 — S1 + S2 完整处理.

    组装 S1 (感觉-辨别) + S2 (整合-评估).

    用法:
      sc = SomatosensoryCortex()
      output = sc.process(
          thalamic_pain=thalamus_output,
          pain_intensity=0.6,
          arousal=0.7,
      )
      # output['sensory'] → 体感感知向量
      # output['pain_location'] → 疼痛定位
      # output['threat'] → 威胁评估
    """

    def __init__(self):
        self.s1 = PrimarySomatosensory()
        self.s2 = SecondarySomatosensory()

        # 整体输出
        self._activation: np.ndarray = np.zeros(
            N_S1_OUTPUT + N_S2_OUTPUT, dtype=np.float32)

    def process(self,
                thalamic_pain: np.ndarray,
                thalamic_touch: Optional[np.ndarray] = None,
                body_vector: Optional[np.ndarray] = None,
                gate_state: float = 0.5,
                pain_intensity: float = 0.0,
                arousal: float = 0.5,
                novelty: float = 0.0) -> dict:
        """体感皮层单步处理.

        Args:
            thalamic_pain: 丘脑痛觉中继信号
            thalamic_touch: 丘脑触觉中继信号
            body_vector: 身体状态
            gate_state: 脊髓闸门状态
            pain_intensity: 综合疼痛强度
            arousal: 唤醒度
            novelty: 新颖性

        Returns:
            dict with:
              'pain_localization': S1痛觉定位
              'pain_intensity_ema': S2疼痛强度EMA
              'threat_assessment': 威胁评估
              'touch_state': 触觉状态
              'body_map': 身体图式
              'activation': 整体激活向量
              'sensory_discriminative': 感觉-辨别分量 (汇入ACC/岛叶)
        """
        # S1: 感觉-辨别
        s1_out = self.s1.process(
            thalamic_pain=thalamic_pain,
            thalamic_touch=thalamic_touch,
            body_vector=body_vector,
            gate_state=gate_state,
        )

        # S2: 整合-评估
        s2_out = self.s2.process(
            s1_output=s1_out,
            pain_intensity=pain_intensity,
            arousal=arousal,
            novelty=novelty,
        )

        # 整体激活
        self._activation = np.concatenate([
            s1_out['pain_localization'][:N_S1_OUTPUT],
            s2_out['integrated'][:N_S2_OUTPUT],
        ]).astype(np.float32)

        # 感觉-辨别分量 (传递到ACC和岛叶)
        sensory_discriminative = s1_out['pain_localization']

        return {
            'pain_localization': s1_out['pain_localization'],
            'pain_intensity_raw': s1_out['pain_intensity'],
            'pain_intensity_ema': s2_out['pain_intensity_ema'],
            'jnd': s2_out['jnd'],
            'threat_assessment': s2_out['threat_assessment'],
            'touch_state': s1_out['touch_state'],
            'touch_pain_interaction': s1_out['touch_pain_interaction'],
            'body_map': s1_out['body_map'],
            'activation': self._activation.copy(),
            'sensory_discriminative': sensory_discriminative,
            's1_output': s1_out,
            's2_output': s2_out,
        }

    # ================================================================
    # 兼容旧接口
    # ================================================================

    def encode_touch(self, contact_info: np.ndarray) -> np.ndarray:
        """触觉编码 (兼容旧接口)."""
        return self.s1._touch_state.copy()

    def encode_proprioception(self, body_state: np.ndarray,
                              action: int = -1) -> np.ndarray:
        """本体感觉读出 (兼容旧接口)."""
        return body_state[:8] if len(body_state) >= 8 else body_state

    def body_schema(self, body_vector: np.ndarray) -> np.ndarray:
        """身体图式 (兼容旧接口)."""
        self.s1.process(
            thalamic_pain=np.zeros(N_PAIN_S1_CHANNELS),
            body_vector=body_vector,
        )
        return self.s1._body_map.copy()
