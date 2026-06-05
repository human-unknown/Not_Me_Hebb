"""
thalamus.py — 丘脑 (Thalamus) [v5.4]

对应脑区: 丘脑诸核团
所属层级: 大脑 → 丘脑 (v4.1: 按图2独立为 Level 3 结构，与边缘系统平级)

功能职责:
  - 感觉中继站 — 除嗅觉外的所有感觉在此中继 → 皮层
  - 感觉门控 — 调节信息流 (选择性传递/抑制)
  - 意识调节 — 丘脑-皮层节律 (睡眠纺锤波、觉醒)
  - 注意力 — 丘脑网状核(TRN) → 探照灯效应

核心核团:
  感觉中继核:
    外侧膝状体 (LGN)         — 视觉中继 (视网膜→V1) [已实现: lgn.py]
    内侧膝状体 (MGB)         — 听觉中继 (下丘→A1) [已实现: mgb.py]
    腹后外侧核 (VPL)         — 躯体感觉中继 (身体→S1) [v5.4 痛觉中继]
    腹后内侧核 (VPM)         — 面部感觉中继 (三叉神经→S1)
  痛觉特异性:
    丘脑后核群 (Po)           — 痛觉整合 → S2 + 岛叶
    背内侧核 (MD)             — 痛觉情感 → ACC + PFC
    板内核群 (CM-Pf)          — 痛觉觉醒/弥散 → 广泛皮层 + 纹状体

v5.4 新增痛觉中继:
  - VPL: 外侧脊髓丘脑束 → S1 (疼痛定位/强度)
  - CM-Pf: 内侧脊髓丘脑束 → 广泛皮层 (痛觉觉醒/情绪)
  - MD: 内侧通路 → ACC (痛觉情感-动机)
  - Po: 痛觉整合 → S2 + 岛叶 (认知评估)
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

N_THALAMIC_PAIN = 8           # 丘脑痛觉总输出维度


class VPL:
    """腹后外侧核 — 躯体感觉中继 (身体→S1).

    外侧脊髓丘脑束 → VPL → S1 (BA3a/3b/1/2)
    编码: 疼痛位置、强度、时间特征 (感觉-辨别)
    """

    def __init__(self):
        # 中继激活
        self._relay: np.ndarray = np.zeros(8, dtype=np.float32)
        # 门控状态 (TRN调制)
        self._gate: float = 0.8
        # tonic/burst 模式
        self._tonic_mode: float = 0.7  # >0.5=tonic (线性中继), <0.5=burst (检测)

    def process(self, lateral_spinothalamic: np.ndarray,
                arousal: float = 0.5,
                attention_gate: float = 0.8) -> np.ndarray:
        """VPL痛觉中继.

        Args:
            lateral_spinothalamic: 外侧脊髓丘脑束信号 (感觉-辨别)
            arousal: 唤醒度 [0, 1]
            attention_gate: 注意力门控 [0, 1] (TRN输出)

        Returns:
            中继到S1的信号
        """
        lateral = np.asarray(lateral_spinothalamic, dtype=np.float32).ravel()

        # Tonic模式 (高唤醒) vs Burst模式 (低唤醒/睡眠)
        self._tonic_mode = 0.7 * self._tonic_mode + 0.3 * arousal

        # 门控: TRN注意力 + 唤醒
        effective_gate = attention_gate * (0.5 + 0.5 * self._tonic_mode)

        # 中继: 在tonic模式下线性传递, burst模式下非线性放大
        if self._tonic_mode > 0.5:
            relayed = lateral * effective_gate
        else:
            # Burst模式: 检测到信号→爆发 (非线形)
            relayed = np.where(lateral > 0.3, lateral * 1.5, lateral * 0.3)
            relayed *= effective_gate

        # 截断到8维
        if len(relayed) < 8:
            relayed = np.pad(relayed, (0, 8 - len(relayed)))
        self._relay = relayed[:8]

        return self._relay.copy()


class CMPf:
    """中央内侧核-束旁核 (CM-Pf) — 非特异性痛觉中继.

    内侧脊髓丘脑束 → CM-Pf → 广泛皮层 + 纹状体
    编码: 痛觉的觉醒成分 + 情绪反应
    特点: 感受野大、弥散、无精确定位
    """

    def __init__(self):
        self._relay: np.ndarray = np.zeros(4, dtype=np.float32)
        self._arousal_signal: float = 0.0

    def process(self, medial_spinothalamic: np.ndarray,
                arousal: float = 0.5) -> np.ndarray:
        """CM-Pf弥散中继.

        Args:
            medial_spinothalamic: 内侧脊髓丘脑束信号 (情感-动机)
            arousal: 唤醒度 [0, 1]

        Returns:
            弥散到广泛皮层+纹状体的信号
        """
        medial = np.asarray(medial_spinothalamic, dtype=np.float32).ravel()

        # CM-Pf: 痛觉觉醒信号 — 放大+弥散
        signal_strength = float(np.mean(np.abs(medial)))
        self._arousal_signal = float(np.clip(
            0.8 * self._arousal_signal + 0.2 * signal_strength * arousal,
            0.0, 1.0))

        # 弥散中继 (低空间精度, 高唤醒驱动)
        diffuse = np.ones(4, dtype=np.float32) * self._arousal_signal
        self._relay = 0.7 * self._relay + 0.3 * diffuse

        return self._relay.copy()


class MedialDorsal:
    """背内侧核 (MD) — 痛觉情感中继.

    内侧通路 → MD → ACC + PFC
    编码: 痛觉的情感-动机成分
    功能: 疼痛的"厌恶体验" + 认知评估
    """

    def __init__(self):
        self._relay: np.ndarray = np.zeros(4, dtype=np.float32)
        self._affective_load: float = 0.0

    def process(self, medial_spinothalamic: np.ndarray,
                unpleasantness: float = 0.0) -> np.ndarray:
        """MD情感中继.

        Args:
            medial_spinothalamic: 内侧脊髓丘脑束信号
            unpleasantness: 疼痛不愉快度 [0, 1] (来自岛叶反馈)

        Returns:
            中继到ACC/PFC的情感信号
        """
        medial = np.asarray(medial_spinothalamic, dtype=np.float32).ravel()

        # MD: 将疼痛信号 + 不愉快度整合 → 情感-动机信号
        pain_avg = float(np.mean(np.abs(medial)))
        self._affective_load = float(np.clip(
            0.7 * self._affective_load
            + 0.3 * (0.6 * pain_avg + 0.4 * unpleasantness),
            0.0, 1.0))

        # 情感中继 → ACC
        affective = np.array([
            self._affective_load,          # 情感负荷
            pain_avg,                       # 疼痛强度
            unpleasantness,                  # 不愉快度
            np.tanh(self._affective_load * 3.0),  # 动机驱动
        ], dtype=np.float32)
        self._relay = 0.7 * self._relay + 0.3 * affective

        return self._relay.copy()


class PosteriorThalamus:
    """丘脑后核群 (Po) — 痛觉整合中继.

    脊髓丘脑束 → Po → S2 + 岛叶
    编码: 痛觉的认知评估 + 感觉整合
    """

    def __init__(self):
        self._relay: np.ndarray = np.zeros(4, dtype=np.float32)

    def process(self, lateral_spinothalamic: np.ndarray,
                medial_spinothalamic: np.ndarray) -> np.ndarray:
        """Po整合中继.

        Args:
            lateral_spinothalamic: 外侧通路
            medial_spinothalamic: 内侧通路

        Returns:
            整合后的中继信号 → S2 + 岛叶
        """
        lateral = np.asarray(lateral_spinothalamic, dtype=np.float32).ravel()
        medial = np.asarray(medial_spinothalamic, dtype=np.float32).ravel()

        # Po整合外侧(感觉) + 内侧(情感)
        lat_avg = float(np.mean(np.abs(lateral)))
        med_avg = float(np.mean(np.abs(medial)))

        integrated = np.array([
            lat_avg,                        # 感觉分量
            med_avg,                        # 情感分量
            (lat_avg + med_avg) / 2,        # 整合强度
            abs(lat_avg - med_avg),         # 感觉-情感不一致 (预测编码用)
        ], dtype=np.float32)
        self._relay = 0.7 * self._relay + 0.3 * integrated

        return self._relay.copy()


class Thalamus:
    """丘脑总装 — 感觉中继 + 痛觉特异核团.

    组装 VPL (感觉-辨别) + CM-Pf (觉醒) + MD (情感) + Po (整合).

    用法:
      thalamus = Thalamus()
      output = thalamus.relay_pain(
          lateral_stt=stt_output['lateral'],
          medial_stt=stt_output['medial'],
          arousal=0.6,
          unpleasantness=0.4,
      )
      # output['vpl'] → S1
      # output['cm_pf'] → 广泛皮层
      # output['md'] → ACC/PFC
      # output['po'] → S2/岛叶
    """

    def __init__(self):
        self.vpl = VPL()
        self.cm_pf = CMPf()
        self.md = MedialDorsal()
        self.po = PosteriorThalamus()

        # TRN (丘脑网状核) — 注意门控
        self._trn_gate: float = 0.8

        # 整体输出
        self._output: np.ndarray = np.zeros(N_THALAMIC_PAIN, dtype=np.float32)

    def set_trn_gate(self, attention: float):
        """设置TRN注意门控 (来自FPN探照灯)."""
        self._trn_gate = float(np.clip(attention, 0.1, 1.0))

    def relay_pain(self,
                   lateral_stt: np.ndarray,
                   medial_stt: np.ndarray,
                   arousal: float = 0.5,
                   unpleasantness: float = 0.0,
                   attention_gate: float = 0.8) -> dict:
        """痛觉丘脑中继.

        Args:
            lateral_stt: 外侧脊髓丘脑束 (感觉-辨别)
            medial_stt: 内侧脊髓丘脑束 (情感-动机)
            arousal: 唤醒度 [0, 1]
            unpleasantness: 不愉快度 [0, 1]
            attention_gate: TRN注意力门控 [0, 1]

        Returns:
            dict with vpl, cm_pf, md, po, combined
        """
        # VPL: 感觉-辨别 → S1
        vpl_out = self.vpl.process(
            lateral_stt, arousal=arousal,
            attention_gate=attention_gate)

        # CM-Pf: 觉醒/弥散 → 广泛皮层
        cmpf_out = self.cm_pf.process(
            medial_stt, arousal=arousal)

        # MD: 情感-动机 → ACC/PFC
        md_out = self.md.process(
            medial_stt, unpleasantness=unpleasantness)

        # Po: 整合 → S2/岛叶
        po_out = self.po.process(lateral_stt, medial_stt)

        # 综合输出
        self._output = np.concatenate([
            vpl_out[:2],
            cmpf_out[:2],
            md_out[:2],
            po_out[:2],
        ]).astype(np.float32)

        return {
            'vpl': vpl_out,
            'cm_pf': cmpf_out,
            'md': md_out,
            'po': po_out,
            'combined': self._output.copy(),
            'trn_gate': self._trn_gate,
        }

    # ================================================================
    # 兼容旧接口
    # ================================================================

    def gate_sensory(self, sensory_input: np.ndarray,
                     attention: float, arousal: float) -> np.ndarray:
        """通用感觉门控 (兼容旧接口)."""
        gate = attention * (0.5 + 0.5 * arousal)
        return sensory_input * gate

    def relay_visual(self, retina_output, gate):
        """视觉中继 (由 lgn.py 处理)."""
        return retina_output

    def relay_auditory(self, cochlear_output, gate):
        """听觉中继 (由 mgb.py 处理)."""
        return cochlear_output
