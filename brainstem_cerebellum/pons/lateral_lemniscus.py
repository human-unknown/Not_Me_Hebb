"""
lateral_lemniscus.py — 外侧丘系 (Lateral Lemniscus) [v5.2]

对应脑区: 外侧丘系 (脑桥-中脑交界)
所属层级: 脑干 → 脑桥 (Level 3)
脑区标签: VNLL · DNLL

功能职责:
  - SOC → IC 的上行纤维束中继
  - VNLL: 时间模式加工 (精确起始反应)
  - DNLL: 双耳时间加工进一步整合 (GABA能抑制)

两个核团:
  - VNLL (腹侧核): 对声音起始精确反应, 参与时间模式编码
  - DNLL (背侧核): 接受双侧输入, 向对侧IC提供GABA能抑制
                   参与双耳时间信息的高级整合

在 NotMe 中的应用:
  - 接收双侧CN + SOC输出
  - 增强时间精度
  - 中继到 InferiorColliculus
"""

import numpy as np
from typing import Optional


class VentralNucleusLateralLemniscus:
    """VNLL — 外侧丘系腹侧核: 时间模式加工.

    对声音起始有精确反应, 参与时间模式编码.
    神经元以极高时间精度响应起始信号.
    """

    def __init__(self, n_channels: int = 32):
        self.n_channels = n_channels
        self._prev_input: Optional[np.ndarray] = None
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

    def process(self, cn_onset: np.ndarray,
                soc_output: Optional[np.ndarray] = None) -> np.ndarray:
        """增强和锐化时间模式.

        Args:
            cn_onset: 耳蜗核起始检测输出 (N_FREQ_CHANNELS,)
            soc_output: SOC空间输出 (N_ITD+N_ILD,) — 可选

        Returns:
            temporal_pattern: 增强的时间模式 (n_channels,)
        """
        onset = np.asarray(cn_onset, dtype=np.float32).ravel()

        # VNLL: 锐化起始信号 — 类似PVCN但更选择性
        # 只保留最强的起始分量
        threshold = np.mean(onset) + np.std(onset)
        sharp_onset = np.where(onset > threshold, onset, 0.0)

        # 时间整合: 当前起始 + 衰减的前一帧
        if self._prev_input is not None:
            # 短时积分 (几ms的时间窗口)
            integrated = sharp_onset + 0.3 * self._prev_input
        else:
            integrated = sharp_onset

        self._prev_input = sharp_onset.copy()

        # 结合SOC的空间信息调制时间模式
        if soc_output is not None:
            soc = np.asarray(soc_output, dtype=np.float32).ravel()
            # SOC空间信息调制VNLL的时间响应
            spatial_mod = float(np.mean(np.abs(soc))) * 0.5 + 0.5
            integrated = integrated * spatial_mod

        # 平滑
        self._activation = 0.7 * self._activation + 0.3 * integrated

        # 输出调整为 n_channels
        if len(self._activation) != self.n_channels:
            result = np.zeros(self.n_channels, dtype=np.float32)
            n_copy = min(len(self._activation), self.n_channels)
            result[:n_copy] = self._activation[:n_copy]
        else:
            result = self._activation.copy()

        return result.astype(np.float32)


class DorsalNucleusLateralLemniscus:
    """DNLL — 外侧丘系背侧核: 双耳时间整合.

    接受双侧输入, 向对侧IC提供GABA能抑制.
    这创建了双耳时间加工的"推-拉"动态,
    增强空间对比度 (类似触觉的侧抑制).
    """

    def __init__(self, n_channels: int = 32):
        self.n_channels = n_channels
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

    def process(self, ipsilateral_input: np.ndarray,
                contralateral_input: np.ndarray) -> np.ndarray:
        """双侧输入的双耳抑制性整合.

        DNLL接受双侧输入, 但对侧IC发送抑制.
        这增强了空间选择性: 同侧更强的响应被保留,
        双侧相等的响应被抑制 → 空间对比度增强.

        Args:
            ipsilateral_input: 同侧信号
            contralateral_input: 对侧信号

        Returns:
            enhanced: 双侧抑制增强后的输出
        """
        ipsi = np.asarray(ipsilateral_input, dtype=np.float32).ravel()
        contra = np.asarray(contralateral_input, dtype=np.float32).ravel()

        # 确保长度匹配
        if len(ipsi) != self.n_channels:
            tmp = np.zeros(self.n_channels, dtype=np.float32)
            n = min(len(ipsi), self.n_channels)
            tmp[:n] = ipsi[:n]
            ipsi = tmp
        if len(contra) != self.n_channels:
            tmp = np.zeros(self.n_channels, dtype=np.float32)
            n = min(len(contra), self.n_channels)
            tmp[:n] = contra[:n]
            contra = tmp

        # DNLL: 同侧兴奋 − 对侧抑制 (GABA能)
        # 这增强了双耳差异 → 空间对比度提升
        enhanced = np.maximum(0.0, ipsi - 0.5 * contra)

        # 平滑
        self._activation = 0.8 * self._activation + 0.2 * enhanced

        return self._activation.astype(np.float32)


class LateralLemniscus:
    """外侧丘系 — SOC→IC中继 + 时间模式加工.

    组装VNLL (时间精度) + DNLL (双侧抑制).
    向上位核团 (InferiorColliculus) 中继.

    用法:
      ll = LateralLemniscus()
      output = ll.process(cn_output, soc_output, contralateral_input)
    """

    def __init__(self, n_channels: int = 32):
        self.vnll = VentralNucleusLateralLemniscus(n_channels=n_channels)
        self.dnll = DorsalNucleusLateralLemniscus(n_channels=n_channels)

    def process(self, cn_output: dict, soc_output: dict,
                contralateral_cn: Optional[dict] = None) -> dict:
        """处理CN+SOC输出, 中继到IC.

        Args:
            cn_output: 耳蜗核输出 dict
            soc_output: 上橄榄复合体输出 dict
            contralateral_cn: 对侧CN输出 (用于DNLL双侧抑制)

        Returns:
            dict with:
              'temporal': VNLL增强的时间模式
              'spatial_enhanced': DNLL空间增强输出
              'relay': 中继到IC的综合信号
        """
        # VNLL: 时间模式加工
        soc_spatial = soc_output.get('spatial', None)
        temporal = self.vnll.process(cn_output['onset'],
                                     soc_output=soc_spatial)

        # DNLL: 双耳抑制 (如果对侧输入可用)
        if contralateral_cn is not None:
            spatial_enhanced = self.dnll.process(
                cn_output['tonotopic'],
                contralateral_cn['tonotopic'],
            )
        else:
            # 无对侧输入 → 用SOC空间信息代替
            spatial_enhanced = np.zeros(self.vnll.n_channels, dtype=np.float32)

        # 中继信号: 时间模式 + 空间信息
        relay = np.concatenate([
            temporal[:16],
            spatial_enhanced[:16],
        ]).astype(np.float32)

        return {
            'temporal': temporal.astype(np.float32),
            'spatial_enhanced': spatial_enhanced.astype(np.float32),
            'relay': relay,
        }
