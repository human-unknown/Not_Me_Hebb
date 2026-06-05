"""
superior_olivary.py — 上橄榄复合体 (Superior Olivary Complex) [v5.2]

对应脑区: 上橄榄复合体 (脑桥下部)
所属层级: 脑干 → 脑桥 (Level 3)
脑区标签: MSO · LSO · MNTB

功能职责:
  - 双耳听觉信息首次汇聚和整合的场所
  - 声音水平定位的核心计算中心
  - ITD检测 (MSO) + ILD检测 (LSO)

三大核团:
  - MSO (内侧上橄榄核): Jeffress巧合检测器 → ITD编码
    树突接受双侧AVCN输入，不同轴突延迟线编码不同时间差
  - LSO (外侧上橄榄核): E-I交互 → ILD编码
    同侧兴奋 + 对侧抑制(经MNTB甘氨酸能)
  - MNTB (内侧斜方体核): 抑制性中继
    花萼状突触(Calyx of Held) — CNS最大突触末梢之一
    以亚毫秒级时间精度向LSO传递对侧抑制

知觉规律:
  1. 双重理论 (Duplex Theory): ITD用于<1.5kHz, ILD用于>3kHz
  2. ITD巧合检测 (Jeffress Model): 延迟线 + 巧合检测神经元
  3. ILD兴奋-抑制 (E-I Interaction): 同侧E + 对侧I

在 NotMe 中的应用:
  - 接收双侧耳蜗核tonotopic输出
  - 输出空间位置向量 (ITD通道 + ILD通道)
  - 为InferiorColliculus提供双耳空间信息
"""

import numpy as np
from typing import Optional, Tuple

# ============================================================
# 常量
# ============================================================

N_ITD_CHANNELS = 12            # ITD检测通道数
N_ILD_CHANNELS = 12            # ILD检测通道数
MAX_ITD_US = 700.0             # 人类最大ITD (μs) — 声源在正侧方
MIN_ITD_US = 10.0              # 最小可检测ITD (μs)
ITD_LOW_FREQ_CUTOFF = 1500.0   # ITD有效频率上界 (Hz)
ILD_HIGH_FREQ_CUTOFF = 3000.0  # ILD有效频率下界 (Hz)
MAX_ILD_DB = 20.0              # 最大ILD (dB) — 极端侧方高频
TAU_DECAY = 0.9                # 神经元响应衰减时间常数


class MedialSuperiorOlive:
    """MSO — 内侧上橄榄核: ITD巧合检测器.

    Jeffress模型 (1948): 来自双侧AVCN的轴突作为延迟线,
    MSO神经元当双侧输入同时到达时放电 (巧合检测),
    不同神经元偏好不同的特征延迟 → ITD空间映射.

    在人类中: 最大ITD ≈ 650-700 μs, 最小可辨 ≈ 10 μs.
    MSO只对低频 (<1.5kHz) 的ITD有精确调谐.
    """

    def __init__(self, n_channels: int = N_ITD_CHANNELS):
        self.n_channels = n_channels

        # ITD调谐中心 (μs): 从 -MAX_ITD 到 +MAX_ITD
        # 负值 = 声音先到左耳 (声源在左侧)
        # 正值 = 声音先到右耳 (声源在右侧)
        self.itd_centers = np.linspace(-MAX_ITD_US, MAX_ITD_US,
                                       n_channels, dtype=np.float32)

        # 调谐宽度 (每个神经元的ITD敏感范围)
        self.tuning_width = MAX_ITD_US / (n_channels / 4)  # ~233 μs

        # 响应衰减
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

    def process(self, left_cn: np.ndarray, right_cn: np.ndarray,
                itd_hint: Optional[float] = None) -> dict:
        """从双侧耳蜗核输出计算ITD空间编码.

        Args:
            left_cn: 左耳耳蜗核tonotopic输出 (N_FREQ_CHANNELS,)
            right_cn: 右耳耳蜗核tonotopic输出 (N_FREQ_CHANNELS,)
            itd_hint: 外部提供的ITD提示 (μs), 可选

        Returns:
            dict with:
              'itd_activation': ITD通道激活 (N_ITD_CHANNELS,)
              'azimuth_estimate': 估计的水平方位角 (度)
              'itd_confidence': ITD估计置信度 [0, 1]
        """
        left = np.asarray(left_cn, dtype=np.float32).ravel()
        right = np.asarray(right_cn, dtype=np.float32).ravel()

        # ---- ITD估计 ----
        if itd_hint is not None:
            # 外部提供ITD → 直接激活对应通道
            estimated_itd = float(itd_hint)
            confidence = 1.0
        else:
            # 从双侧信号估计ITD
            estimated_itd, confidence = self._estimate_itd_from_signals(
                left, right)

        # ---- Jeffress延迟线模型: 各通道响应 ----
        # 每个通道对特定ITD有最大响应 (高斯调谐)
        itd_activation = np.zeros(self.n_channels, dtype=np.float32)
        for i, center in enumerate(self.itd_centers):
            # 高斯调谐: 实际ITD越接近通道中心 → 响应越强
            distance = (estimated_itd - center) / (self.tuning_width + 1e-8)
            itd_activation[i] = np.exp(-0.5 * distance ** 2)

        # ---- 时间平滑 (神经元动力学) ----
        self._activation = (TAU_DECAY * self._activation +
                           (1.0 - TAU_DECAY) * itd_activation)

        # ---- 方位角估计 ----
        # ITD → 方位角映射: sin(azimuth) ≈ ITD / MAX_ITD
        # 这是简化的几何模型 (球头)
        normalized_itd = np.clip(estimated_itd / MAX_ITD_US, -1.0, 1.0)
        azimuth = float(np.arcsin(normalized_itd) * 180.0 / np.pi)

        return {
            'itd_activation': self._activation.copy(),
            'azimuth_estimate': azimuth,
            'itd_confidence': confidence,
            'estimated_itd_us': estimated_itd,
        }

    def _estimate_itd_from_signals(self, left: np.ndarray,
                                   right: np.ndarray) -> Tuple[float, float]:
        """从双侧信号估计ITD (无外部hint时的交叉相关法).

        Returns:
            (estimated_itd_us, confidence)
        """
        # 交叉相关: 找出左右信号的相位偏移
        # 只对低频部分做交叉相关 (高频相位锁定不可靠)
        n_freq = len(left)
        low_cut = max(1, int(n_freq * 0.4))  # 只看低频 ~40% 通道

        left_low = left[:low_cut]
        right_low = right[:low_cut]

        # 简化: 计算低频段能量加权的相位差
        # 左右总能量差 → 粗略ITD方向
        left_energy = float(np.sum(np.abs(left_low)))
        right_energy = float(np.sum(np.abs(right_low)))
        total_energy = left_energy + right_energy + 1e-8

        # 能量偏向 → ITD方向
        energy_bias = (right_energy - left_energy) / total_energy

        # 各频率通道的相位差 → ITD
        # 低通道 → 低频率 → 大ITD范围
        weighted_itd = 0.0
        total_weight = 0.0
        for i in range(low_cut):
            if left[i] > 0.01 and right[i] > 0.01:
                # 该通道的激活差异 → ITD贡献
                diff = right[i] - left[i]
                # 低频权重更高 (更可靠的时间编码)
                weight = (low_cut - i) / low_cut
                weighted_itd += diff * weight * MAX_ITD_US
                total_weight += weight

        if total_weight > 1e-8:
            signal_itd = weighted_itd / total_weight
        else:
            signal_itd = energy_bias * MAX_ITD_US

        # 混合: 交叉相关 + 能量偏向
        estimated_itd = 0.7 * signal_itd + 0.3 * energy_bias * MAX_ITD_US

        # 置信度: 能量越高越可信
        confidence = min(1.0, total_energy * 2.0)

        return float(np.clip(estimated_itd, -MAX_ITD_US, MAX_ITD_US)), confidence


class LateralSuperiorOlive:
    """LSO — 外侧上橄榄核: ILD检测.

    E-I交互: 同侧耳蜗核→兴奋, 对侧耳蜗核→抑制(经MNTB甘氨酸能).
    当同侧声音更强 → 兴奋 > 抑制 → 高放电率.
    当对侧声音更强 → 抑制 > 兴奋 → 低放电率.

    LSO主要对高频 (>3kHz) ILD有调谐 (声影效应显著).
    """

    def __init__(self, n_channels: int = N_ILD_CHANNELS):
        self.n_channels = n_channels

        # ILD调谐中心 (dB): 从 -MAX_ILD 到 +MAX_ILD
        # 负值 = 左耳更响 (声源在左侧)
        # 正值 = 右耳更响 (声源在右侧)
        self.ild_centers = np.linspace(-MAX_ILD_DB, MAX_ILD_DB,
                                       n_channels, dtype=np.float32)
        self.tuning_width = MAX_ILD_DB / (n_channels / 6)  # ~10 dB

        # 响应衰减
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

    def process(self, left_cn: np.ndarray, right_cn: np.ndarray,
                ild_hint: Optional[float] = None) -> dict:
        """从双侧耳蜗核输出计算ILD空间编码.

        Args:
            left_cn: 左耳耳蜗核tonotopic输出 (N_FREQ_CHANNELS,)
            right_cn: 右耳耳蜗核tonotopic输出 (N_FREQ_CHANNELS,)
            ild_hint: 外部提供的ILD提示 (dB), 可选

        Returns:
            dict with:
              'ild_activation': ILD通道激活 (N_ILD_CHANNELS,)
              'ild_estimate': 估计的ILD (dB)
              'ild_confidence': ILD估计置信度 [0, 1]
        """
        left = np.asarray(left_cn, dtype=np.float32).ravel()
        right = np.asarray(right_cn, dtype=np.float32).ravel()

        # ---- ILD估计 (E-I交互) ----
        n_freq = len(left)
        high_start = max(0, int(n_freq * 0.5))  # 只看高频 ~50% 通道

        left_high = left[high_start:]
        right_high = right[high_start:]

        left_power = float(np.mean(np.abs(left_high)) + 1e-8)
        right_power = float(np.mean(np.abs(right_high)) + 1e-8)

        # ILD = 20 * log10(right/left) — 正值=右侧响
        if ild_hint is not None:
            estimated_ild = float(ild_hint)
            confidence = 1.0
        else:
            estimated_ild = 20.0 * np.log10(right_power / left_power)
            estimated_ild = float(np.clip(estimated_ild, -MAX_ILD_DB, MAX_ILD_DB))

            # 置信度: 高频能量越高越可信 (声影效应需要高频)
            hf_energy = left_power + right_power
            confidence = min(1.0, hf_energy * 3.0)

        # ---- E-I交互模型: 各LSO通道响应 ----
        # 每个LSO神经元: activation = sigmoid(excitation - inhibition)
        # excitation 来自同侧, inhibition 来自对侧(经MNTB)
        ild_activation = np.zeros(self.n_channels, dtype=np.float32)
        for i, center in enumerate(self.ild_centers):
            # 高斯调谐: 实际ILD越接近通道中心 → 响应越强
            distance = (estimated_ild - center) / (self.tuning_width + 1e-8)
            ild_activation[i] = np.exp(-0.5 * distance ** 2)

        # ---- 时间平滑 ----
        self._activation = (TAU_DECAY * self._activation +
                           (1.0 - TAU_DECAY) * ild_activation)

        return {
            'ild_activation': self._activation.copy(),
            'ild_estimate': estimated_ild,
            'ild_confidence': confidence,
        }


class MedialNucleusTrapezoidBody:
    """MNTB — 内侧斜方体核: 抑制性中继.

    接受对侧AVCN输入, 向同侧LSO提供甘氨酸能抑制.
    花萼状突触 (Calyx of Held) 确保极高时间精度.
    在ILD计算中提供精确的对侧抑制信号.
    """

    def __init__(self):
        # MNTB主要作为中继, 本身不产生新的编码
        # 它将AVCN的兴奋性信号转换为抑制性信号
        self._inhibition_strength: float = 1.0

    def relay_inhibition(self, contralateral_cn: np.ndarray) -> np.ndarray:
        """将对侧AVCN信号转换为抑制信号 (→LSO).

        Args:
            contralateral_cn: 对侧耳蜗核tonotopic输出

        Returns:
            inhibition: 抑制信号 (与输入同维度)
        """
        cn = np.asarray(contralateral_cn, dtype=np.float32)
        # MNTB: 高保真转换 — 几乎是1:1的信号转换
        # 甘氨酸能抑制: 信号幅度调制
        inhibition = cn * self._inhibition_strength
        return inhibition.astype(np.float32)


# ============================================================
# SuperiorOlive — 上橄榄复合体 (MSO + LSO + MNTB 整合)
# ============================================================

class SuperiorOlive:
    """上橄榄复合体 — 双耳空间信息处理中心.

    组装 MSO (ITD), LSO (ILD), MNTB (抑制中继),
    实现双重理论 (Duplex Theory) 的双耳定位.

    用法:
      soc = SuperiorOlive()
      output = soc.process(left_cn_output, right_cn_output)
      # output['spatial']      → (N_ITD+N_ILD,) 空间特征向量
      # output['azimuth']      → 估计方位角
      # output['duplex_weights'] → 双重理论频率权重
    """

    def __init__(self, n_itd: int = N_ITD_CHANNELS,
                 n_ild: int = N_ILD_CHANNELS):
        self.mso = MedialSuperiorOlive(n_channels=n_itd)
        self.lso = LateralSuperiorOlive(n_channels=n_ild)
        self.mntb = MedialNucleusTrapezoidBody()

    def process(self, left_cn: dict, right_cn: dict,
                azimuth_hint: Optional[float] = None) -> dict:
        """处理双侧耳蜗核输出, 生成空间信息.

        Args:
            left_cn: 左耳耳蜗核输出 dict (来自 CochlearNucleus.process())
            right_cn: 右耳耳蜗核输出 dict
            azimuth_hint: 外部方位角提示 (度), -90=左, +90=右, 0=正前方

        Returns:
            dict with:
              'spatial': 空间特征向量 (N_ITD + N_ILD,)
              'azimuth': 综合方位角估计 (度)
              'mso_output': MSO输出详情
              'lso_output': LSO输出详情
              'duplex_weights': 双重理论权重 (ITD_weight, ILD_weight)
        """
        left_tono = left_cn['tonotopic']
        right_tono = right_cn['tonotopic']

        # 从方位角hint推导ITD和ILD
        itd_hint = None
        ild_hint = None
        if azimuth_hint is not None:
            # 简化几何模型
            az_rad = float(azimuth_hint) * np.pi / 180.0
            itd_hint = float(np.sin(az_rad) * MAX_ITD_US)
            # ILD ≈ 方位角的函数 (高频声影效应)
            ild_hint = float(np.sin(az_rad) * MAX_ILD_DB)

        # ---- MSO: ITD处理 ----
        mso_out = self.mso.process(left_tono, right_tono, itd_hint=itd_hint)

        # ---- LSO: ILD处理 (含MNTB对侧抑制) ----
        # MNTB将左CN转抑制信号给右LSO (反之亦然)
        # 这里简化: LSO内部完成E-I交互
        lso_out = self.lso.process(left_tono, right_tono, ild_hint=ild_hint)

        # ---- 双重理论 (Duplex Theory): 频率相关权重 ----
        # ITD可靠用于低频 (<1.5kHz), ILD可靠用于高频 (>3kHz)
        # 计算输入信号的频谱重心 → 判断低频/高频主导
        n_freq = len(left_tono)
        freq_weights = np.linspace(0, 1, n_freq)
        spectral_centroid = float(np.average(freq_weights,
            weights=np.abs(left_tono) + np.abs(right_tono) + 1e-8))

        # 频谱重心低 → ITD权重高; 重心高 → ILD权重高
        itd_weight = float(np.clip(1.0 - spectral_centroid * 1.5, 0.1, 0.9))
        ild_weight = float(np.clip(spectral_centroid * 1.5, 0.1, 0.9))

        # ---- 综合方位角估计 ----
        azimuth_itd = mso_out['azimuth_estimate']
        azimuth_ild = lso_out['ild_estimate'] / MAX_ILD_DB * 90.0  # ILD→度

        # 双重理论融合
        azimuth = (itd_weight * azimuth_itd + ild_weight * azimuth_ild)

        # ---- 空间特征向量 ----
        spatial = np.concatenate([
            mso_out['itd_activation'],
            lso_out['ild_activation'],
        ]).astype(np.float32)

        return {
            'spatial': spatial,
            'azimuth': float(azimuth),
            'mso_output': mso_out,
            'lso_output': lso_out,
            'duplex_weights': (itd_weight, ild_weight),
        }
