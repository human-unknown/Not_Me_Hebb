"""
inferior_colliculus.py — 下丘 (Inferior Colliculus) [v5.2]

对应脑区: 下丘 (中脑顶盖)
所属层级: 脑干 → 中脑 (Level 3)
脑区标签: ICc (中央核) · ICd (背皮层) · ICx (外皮层)

功能职责:
  - 听觉中脑核心结构 — 几乎所有低位脑干听觉核团在此汇聚
  - 频率×空间×时间 三维信息整合
  - 多感官整合 (听觉×视觉, 与上丘协作)
  - 新颖性检测 — 声音变化检测 → 朝向反射

三大亚区:
  - 中央核 (ICc): 清晰板层结构, 严格tonotopic
                    窄带频率调谐 + 多种时间反应模式
  - 背皮层 (ICd): 非经典听觉通路输入, 更复杂的反应特性
  - 外皮层 (ICx): 多感官整合 (听觉+视觉+体感)
                    参与引导头颈朝向声源的定向运动

知觉规律:
  1. 频率×空间×时间 三维整合
  2. FM扫频方向选择性
  3. 新颖性检测 (变化检测优先原则)
  4. 多感官整合 (与SuperiorColliculus协作)

在 NotMe 中的应用:
  - 接收SOC + LL输出, 整合为统一的听觉表征
  - 生成新颖性信号 → 汇入扣带回/ACC 突显网络
  - 与上丘 (SuperiorColliculus) 进行视听多感官整合
  - 输出到 MGB (内侧膝状体) → 听皮层
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

N_IC_SPATIAL = 24             # IC输出空间维度
N_IC_TEMPORAL = 16            # IC时间模式维度
FM_SWEEP_CHANNELS = 8         # FM扫频方向检测通道


class CentralNucleus:
    """ICc — 中央核: tonotopic频率×空间整合.

    维持清晰的音调拓扑组织,
    整合来自SOC的双耳空间信息,
    对FM扫频方向和速度有选择性.
    """

    def __init__(self, n_spatial: int = N_IC_SPATIAL,
                 n_temporal: int = N_IC_TEMPORAL):
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal

        # FM扫频方向检测器 (上/下扫频)
        self._fm_detectors = np.zeros((FM_SWEEP_CHANNELS, 2), dtype=np.float32)
        self._prev_tonotopic: Optional[np.ndarray] = None

        # 神经元激活 (多种时间反应模式)
        self._sustained: np.ndarray = np.zeros(n_spatial, dtype=np.float32)
        self._onset: np.ndarray = np.zeros(n_spatial, dtype=np.float32)
        self._pauser: np.ndarray = np.zeros(n_spatial, dtype=np.float32)

    def process(self, cn_tonotopic: np.ndarray,
                soc_spatial: np.ndarray,
                ll_relay: Optional[np.ndarray] = None) -> dict:
        """整合频率+空间+时间信息.

        Args:
            cn_tonotopic: CN tonotopic输出 (N_FREQ_CHANNELS,)
            soc_spatial: SOC空间向量 (N_ITD+N_ILD,)
            ll_relay: LL中继信号 (可选)

        Returns:
            dict with:
              'spatial_map': 频率×空间整合表征
              'temporal_pattern': 时间反应模式
              'fm_sweep': FM扫频方向激活
        """
        tono = np.asarray(cn_tonotopic, dtype=np.float32).ravel()
        spa = np.asarray(soc_spatial, dtype=np.float32).ravel()

        # ---- 频率×空间整合: 外积 → 联合表征 ----
        # 每个频率通道和每个空间通道的联合激活
        n_freq = min(len(tono), 16)
        n_spa = min(len(spa), 12)

        # 简化的联合编码: 取前n_freq个频率通道
        tono_reduced = tono[:n_freq] if len(tono) > n_freq else \
            np.pad(tono, (0, max(0, n_freq - len(tono))))

        spa_reduced = spa[:n_spa] if len(spa) > n_spa else \
            np.pad(spa, (0, max(0, n_spa - len(spa))))

        # 联合矩阵 → 展平为 spatial_map
        joint = np.outer(tono_reduced, spa_reduced)
        spatial_map = joint.ravel()[:self.n_spatial]  # 截取到 n_spatial
        if len(spatial_map) < self.n_spatial:
            spatial_map = np.pad(spatial_map,
                (0, self.n_spatial - len(spatial_map)))

        # ---- 多种时间反应模式 ----
        # Pad tono_reduced to n_spatial if needed
        if len(tono_reduced) < self.n_spatial:
            tono_spatial = np.pad(tono_reduced,
                (0, self.n_spatial - len(tono_reduced)))
        else:
            tono_spatial = tono_reduced[:self.n_spatial]

        # sustained: 持续响应 (能量均值)
        self._sustained = 0.8 * self._sustained + 0.2 * tono_spatial

        # onset: 起始响应 (正差分)
        if self._prev_tonotopic is not None:
            prev_spatial = (np.pad(self._prev_tonotopic,
                (0, max(0, self.n_spatial - len(self._prev_tonotopic))))
                if len(self._prev_tonotopic) < self.n_spatial
                else self._prev_tonotopic[:self.n_spatial])
            diff = np.maximum(0.0, tono_spatial - prev_spatial)
        else:
            diff = tono_spatial
        self._onset = 0.6 * self._onset + 0.4 * diff

        # pauser: 起始后暂停 → 对持续高强度声音暂停
        strong_mask = (self._sustained > 0.5).astype(np.float32)
        self._pauser = 0.7 * self._pauser + 0.3 * (self._onset * (1.0 - strong_mask))

        # 时间模式: 三种反应的拼接
        temporal_pattern = np.concatenate([
            self._sustained[:self.n_temporal // 3],
            self._onset[:self.n_temporal // 3],
            self._pauser[:self.n_temporal // 3 + self.n_temporal % 3],
        ])

        # ---- FM扫频方向检测 ----
        fm_sweep = self._detect_fm_sweep(tono)

        self._prev_tonotopic = tono.copy()

        return {
            'spatial_map': spatial_map.astype(np.float32),
            'temporal_pattern': temporal_pattern.astype(np.float32),
            'fm_sweep': fm_sweep.astype(np.float32),
        }

    def _detect_fm_sweep(self, tonotopic: np.ndarray) -> np.ndarray:
        """检测频率调制 (FM) 扫频方向.

        FM扫频是语音和许多自然声音的基本特征。
        部分IC神经元对FM扫频方向(向上/向下)和速度有选择性。
        """
        fm = np.zeros(FM_SWEEP_CHANNELS, dtype=np.float32)

        if self._prev_tonotopic is None or len(tonotopic) < FM_SWEEP_CHANNELS:
            return fm

        # 将tonotopic分成FM_SWEEP_CHANNELS个频段
        n_per_band = max(1, len(tonotopic) // FM_SWEEP_CHANNELS)

        for i in range(FM_SWEEP_CHANNELS):
            start = i * n_per_band
            end = min(start + n_per_band, len(tonotopic))

            curr_energy = float(np.mean(tonotopic[start:end]))
            prev_energy = float(np.mean(self._prev_tonotopic[start:end]))

            # 能量在频率轴上移动 → FM扫频
            # 正 = 能量向高频移动 (向上扫频)
            # 负 = 能量向低频移动 (向下扫频)
            if i < FM_SWEEP_CHANNELS - 1:
                next_start = (i + 1) * n_per_band
                next_end = min(next_start + n_per_band, len(tonotopic))
                next_curr = float(np.mean(tonotopic[next_start:next_end]))
                # 当前频段能量下降 + 相邻高频段能量上升 → 向上扫频
                fm[i] = float(np.tanh(
                    (next_curr - curr_energy) * 5.0 -
                    (curr_energy - prev_energy) * 3.0
                ))

        return fm


class ExternalCortex:
    """ICx — 外皮层: 多感官整合.

    接受听觉+视觉+体感输入,
    参与引导头颈朝向声源的定向运动 (与上丘协作).
    """

    def __init__(self):
        self._activation: np.ndarray = np.zeros(16, dtype=np.float32)
        self._multisensory_map: np.ndarray = np.zeros(16, dtype=np.float32)

    def process(self, auditory_spatial: np.ndarray,
                visual_spatial: Optional[np.ndarray] = None) -> dict:
        """多感官 (听觉×视觉) 空间整合.

        Args:
            auditory_spatial: 听觉空间表征
            visual_spatial: 视觉空间信息 (来自 SuperiorColliculus 或 V1)

        Returns:
            dict with:
              'multisensory': 多感官整合表征
              'cross_modal_conflict': 视听冲突信号
        """
        aud = np.asarray(auditory_spatial, dtype=np.float32).ravel()

        # 调整到16维
        if len(aud) != 16:
            tmp = np.zeros(16, dtype=np.float32)
            n = min(len(aud), 16)
            tmp[:n] = aud[:n]
            aud = tmp

        if visual_spatial is not None:
            vis = np.asarray(visual_spatial, dtype=np.float32).ravel()
            if len(vis) != 16:
                tmp = np.zeros(16, dtype=np.float32)
                n = min(len(vis), 16)
                tmp[:n] = vis[:n]
                vis = tmp

            # 多感官融合: 加权平均 (贝叶斯加权)
            aud_conf = float(np.mean(np.abs(aud)))
            vis_conf = float(np.mean(np.abs(vis)))
            total_conf = aud_conf + vis_conf + 1e-8

            aud_weight = aud_conf / total_conf
            vis_weight = vis_conf / total_conf

            self._multisensory_map = aud_weight * aud + vis_weight * vis

            # 视听冲突: 空间信息不一致时增大
            cross_modal_conflict = float(
                np.mean(np.abs(aud - vis)) /
                (np.mean(np.abs(aud)) + np.mean(np.abs(vis)) + 1e-8)
            )
        else:
            self._multisensory_map = aud
            cross_modal_conflict = 0.0

        self._activation = 0.7 * self._activation + 0.3 * self._multisensory_map

        return {
            'multisensory': self._activation.copy(),
            'cross_modal_conflict': cross_modal_conflict,
        }


class InferiorColliculus:
    """下丘 — 听觉中脑整合核心.

    组装 ICc (频率×空间整合) + ICx (多感官整合).
    几乎所有上行听觉信息在此汇聚整合.

    用法:
      ic = InferiorColliculus()
      output = ic.process(cn_output, soc_output, ll_output, visual_spatial)
      # output['integrated'] → (N_IC_SPATIAL+N_IC_TEMPORAL,) 整合特征
      # output['novelty'] → 新颖性信号 → 汇入 ACC
    """

    def __init__(self, n_spatial: int = N_IC_SPATIAL,
                 n_temporal: int = N_IC_TEMPORAL):
        self.icc = CentralNucleus(n_spatial=n_spatial,
                                  n_temporal=n_temporal)
        self.icx = ExternalCortex()

        # 新颖性检测状态
        self._prev_integrated: Optional[np.ndarray] = None
        self._novelty_ema: float = 0.0

        # 预测状态 (用于预测编码)
        self._prediction: Optional[np.ndarray] = None

    def process(self, cn_output: dict, soc_output: dict,
                ll_output: Optional[dict] = None,
                visual_spatial: Optional[np.ndarray] = None) -> dict:
        """下丘整合处理.

        Args:
            cn_output: 耳蜗核输出 dict
            soc_output: SOC输出 dict
            ll_output: LL输出 dict (可选)
            visual_spatial: 视觉空间信息 (来自SC, 可选)

        Returns:
            dict with:
              'integrated': 整合听觉特征 (N_SPATIAL+N_TEMPORAL,)
              'novelty': 新颖性信号 [0, 1]
              'fm_sweep': FM扫频方向
              'multisensory': 多感官整合
              'cross_modal_conflict': 视听冲突信号
        """
        # ICc: 频率×空间整合
        ll_relay = ll_output['relay'] if ll_output is not None else None
        icc_out = self.icc.process(
            cn_output['tonotopic'],
            soc_output['spatial'],
            ll_relay=ll_relay,
        )

        # 整合特征 = 空间映射 + 时间模式 + FM扫频
        integrated = np.concatenate([
            icc_out['spatial_map'],
            icc_out['temporal_pattern'][:8],
            icc_out['fm_sweep'],
        ]).astype(np.float32)

        # 调整到输出维度
        target_len = self.icc.n_spatial + self.icc.n_temporal
        if len(integrated) != target_len:
            tmp = np.zeros(target_len, dtype=np.float32)
            n = min(len(integrated), target_len)
            tmp[:n] = integrated[:n]
            integrated = tmp

        # ---- 新颖性检测 ----
        novelty = self._detect_novelty(integrated)

        # ---- ICx: 多感官整合 ----
        icx_out = self.icx.process(icc_out['spatial_map'],
                                   visual_spatial=visual_spatial)

        # ---- 预测编码: 更新内部预测 ----
        if self._prediction is None:
            self._prediction = integrated.copy()
        else:
            # 预测: 前一帧 + 平滑趋势
            self._prediction = 0.9 * self._prediction + 0.1 * integrated

        return {
            'integrated': integrated,
            'novelty': novelty,
            'fm_sweep': icc_out['fm_sweep'],
            'multisensory': icx_out['multisensory'],
            'cross_modal_conflict': icx_out['cross_modal_conflict'],
        }

    def _detect_novelty(self, current: np.ndarray) -> float:
        """新颖性检测: 当前输入与历史模式的差异.

        变化检测优先原则: 新异/变化的信息被放大。
        """
        current = np.asarray(current, dtype=np.float32)

        if self._prev_integrated is None:
            self._prev_integrated = current.copy()
            return 0.5  # 首次输入 → 中等新颖

        # 与上一帧差异
        frame_diff = float(np.mean(np.abs(current - self._prev_integrated)))

        # 与EMA的差异 (检测趋势变化)
        ema = 0.9 * self._prev_integrated + 0.1 * current
        trend_diff = float(np.mean(np.abs(current - ema)))

        self._prev_integrated = ema.copy()

        # 新颖性 = 帧间差异 + 趋势差异
        raw_novelty = frame_diff * 2.0 + trend_diff * 1.0

        # 平滑 + 归一化
        self._novelty_ema = 0.85 * self._novelty_ema + 0.15 * raw_novelty
        novelty = float(np.tanh(self._novelty_ema * 5.0))

        return novelty

    # ================================================================
    # 预测编码接口
    # ================================================================

    def get_prediction(self) -> np.ndarray:
        """返回当前预测 (供MGB反馈使用)."""
        if self._prediction is None:
            return np.zeros(N_IC_SPATIAL + N_IC_TEMPORAL, dtype=np.float32)
        return self._prediction.copy()

    def receive_feedback(self, prediction_error: np.ndarray,
                         lr: float = 0.1):
        """接收MGB的预测误差反馈.

        Args:
            prediction_error: 上位核团计算的预测误差
            lr: 学习率
        """
        if self._prediction is not None:
            pe = np.asarray(prediction_error, dtype=np.float32).ravel()
            target_len = len(self._prediction)
            if len(pe) == target_len:
                self._prediction += lr * pe

    def compute_prediction_error(self) -> np.ndarray:
        """计算当前输入的预测误差."""
        if self._prediction is None or self._prev_integrated is None:
            return np.zeros(N_IC_SPATIAL + N_IC_TEMPORAL, dtype=np.float32)
        return (self._prev_integrated - self._prediction).astype(np.float32)
