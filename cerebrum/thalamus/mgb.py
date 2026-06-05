"""
mgb.py — 内侧膝状体 (Medial Geniculate Body) [v5.2]

对应脑区: 内侧膝状体 (丘脑)
所属层级: 大脑 → 丘脑 (Level 3)
脑区标签: MGv · MGd · MGm

功能职责:
  - 听觉信息到达皮层前的最后一个突触中继站
  - MGv → A1: 精确tonotopic中继
  - MGd → Belt: 复杂特征中继
  - MGm: 多感官 + 弥散投射
  - 状态门控: 唤醒度/注意力调制信号传递
  - 睡眠/麻醉 → 反应显著减弱

三个亚区:
  - MGv (腹侧分部): 严格tonotopic, 是A1的主要输入源
  - MGd (背侧分部): 反应特性更复杂, 投射至非初级听皮层
  - MGm (内侧分部): 多感官输入, 投射弥散

知觉规律:
  1. 丘脑门控 (感觉选择) — 按需传递/抑制感觉信息
  2. 状态依赖处理 — 唤醒度调制 (睡眠→信号衰减)

在 NotMe 中的应用:
  - 接收IC输出, 中继到听皮层
  - 唤醒度门控: 低唤醒→衰减, 高唤醒→增强
  - FPN注意调制: 选择性增强任务相关听觉特征
"""

import numpy as np
from typing import Optional

# ============================================================
# 常量
# ============================================================

N_MGV_CHANNELS = 24           # MGv tonotopic输出维度 (→ A1)
N_MGD_CHANNELS = 16           # MGd 复杂特征输出维度 (→ Belt)
N_MGM_CHANNELS = 8            # MGm 多感官输出维度


class VentralMGB:
    """MGv — 腹侧分部: 精确tonotopic中继 → A1.

    严格维持音调拓扑组织, 是初级听皮层(A1)的主要输入来源.
    神经元有相对窄的频率调谐.
    """

    def __init__(self, n_channels: int = N_MGV_CHANNELS):
        self.n_channels = n_channels
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

        # 调谐锐化: MGv对IC输出进行频率调谐锐化
        self._sharpening_matrix: Optional[np.ndarray] = None

    def process(self, ic_integrated: np.ndarray,
                arousal: float = 0.8) -> np.ndarray:
        """IC输出 → 精确tonotopic中继 → A1.

        Args:
            ic_integrated: IC整合特征向量
            arousal: 脑干唤醒度 [0, 1] — 低唤醒→衰减

        Returns:
            tonotopic_relay: 精确频率-空间中继 (N_MGV_CHANNELS,)
        """
        ic = np.asarray(ic_integrated, dtype=np.float32).ravel()

        # 懒初始化调谐锐化矩阵
        if self._sharpening_matrix is None:
            n_ic = max(len(ic), 1)
            # 窄带宽映射: IC→MGv 频率调谐锐化
            self._sharpening_matrix = np.zeros(
                (self.n_channels, n_ic), dtype=np.float32)
            for i in range(self.n_channels):
                # 每个MGv神经元从IC中窄带采样
                center = int(n_ic * i / max(self.n_channels, 1))
                width = max(1, n_ic // max(self.n_channels, 1))
                start = max(0, center - width // 2)
                end = min(n_ic, center + width // 2)
                if end > start:
                    self._sharpening_matrix[i, start:end] = 1.0 / (end - start)

        # 确保矩阵维度匹配当前ic长度
        if self._sharpening_matrix.shape[1] != len(ic):
            n_ic = max(len(ic), 1)
            self._sharpening_matrix = np.zeros(
                (self.n_channels, n_ic), dtype=np.float32)
            for i in range(self.n_channels):
                center = int(n_ic * i / max(self.n_channels, 1))
                width = max(1, n_ic // max(self.n_channels, 1))
                start = max(0, center - width // 2)
                end = min(n_ic, center + width // 2)
                if end > start:
                    self._sharpening_matrix[i, start:end] = 1.0 / (end - start)

        # 调谐锐化: IC → MGv (窄带频率调谐)
        relay = self._sharpening_matrix @ ic

        # 唤醒度门控: 低唤醒→信号衰减
        arousal_gain = float(np.clip(arousal, 0.1, 1.0))

        # 时间平滑 + 唤醒门控
        self._activation = (0.8 * self._activation +
                           0.2 * relay * arousal_gain)

        return self._activation.astype(np.float32)


class DorsalMGB:
    """MGd — 背侧分部: 复杂特征中继 → Belt.

    反应特性比MGv更复杂 — 对FM扫频、谐波结构等有选择性.
    投射至非初级听皮层 (belt area).
    """

    def __init__(self, n_channels: int = N_MGD_CHANNELS):
        self.n_channels = n_channels
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

    def process(self, ic_integrated: np.ndarray,
                ic_fm_sweep: Optional[np.ndarray] = None,
                arousal: float = 0.8) -> np.ndarray:
        """IC输出 → 复杂特征提取 → Belt.

        Args:
            ic_integrated: IC整合特征
            ic_fm_sweep: IC FM扫频方向 (可选)
            arousal: 脑干唤醒度

        Returns:
            complex_features: 复杂听觉特征 (N_MGD_CHANNELS,)
        """
        ic = np.asarray(ic_integrated, dtype=np.float32).ravel()

        # MGd: 从IC中提取复杂特征
        # 使用随机投影模拟"复杂反应特性"
        rng = np.random.RandomState(13)
        if not hasattr(self, '_proj_matrix') or \
           self._proj_matrix.shape[1] != len(ic):
            self._proj_matrix = rng.randn(self.n_channels, len(ic)).astype(
                np.float32) * 0.3

        complex_features = np.tanh(self._proj_matrix @ ic)

        # FM扫频信息整合
        if ic_fm_sweep is not None:
            fm = np.asarray(ic_fm_sweep, dtype=np.float32).ravel()
            # FM特征调制复杂特征
            fm_influence = np.zeros(self.n_channels, dtype=np.float32)
            n_fm = min(len(fm), self.n_channels)
            fm_influence[:n_fm] = fm[:n_fm]
            complex_features = 0.7 * complex_features + 0.3 * fm_influence

        # 唤醒度门控
        arousal_gain = float(np.clip(arousal, 0.1, 1.0))

        # 时间平滑
        self._activation = (0.75 * self._activation +
                           0.25 * complex_features * arousal_gain)

        return self._activation.astype(np.float32)


class MedialMGB:
    """MGm — 内侧分部: 多感官 + 弥散投射.

    接受多感官输入, 投射弥散至听皮层各层.
    参与听觉-躯体感觉-视觉的跨模态整合.
    """

    def __init__(self, n_channels: int = N_MGM_CHANNELS):
        self.n_channels = n_channels
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

    def process(self, multisensory_input: Optional[np.ndarray] = None,
                arousal: float = 0.8) -> np.ndarray:
        """多感官弥散投射.

        Args:
            multisensory_input: ICx多感官整合输出 (可选)
            arousal: 脑干唤醒度

        Returns:
            diffuse: 弥散投射 (N_MGM_CHANNELS,)
        """
        if multisensory_input is not None:
            ms = np.asarray(multisensory_input, dtype=np.float32).ravel()
            # 调整到输出维度
            diffuse = np.zeros(self.n_channels, dtype=np.float32)
            n = min(len(ms), self.n_channels)
            diffuse[:n] = ms[:n]
        else:
            diffuse = np.zeros(self.n_channels, dtype=np.float32)

        # MGm受唤醒度影响更大 (睡眠时显著减弱)
        arousal_gain = float(np.clip(arousal, 0.05, 1.0)) ** 1.5  # 非线性门控

        self._activation = (0.8 * self._activation +
                           0.2 * diffuse * arousal_gain)

        return self._activation.astype(np.float32)


class MedialGeniculateBody:
    """内侧膝状体 — 丘脑听觉中继站.

    组装 MGv (→A1), MGd (→Belt), MGm (弥散).
    实现唤醒度门控 + FPN注意调制.

    用法:
      mgb = MedialGeniculateBody()
      output = mgb.process(ic_output, arousal=0.8, fpn=fpn)
    """

    def __init__(self):
        self.mgv = VentralMGB(n_channels=N_MGV_CHANNELS)
        self.mgd = DorsalMGB(n_channels=N_MGD_CHANNELS)
        self.mgm = MedialMGB(n_channels=N_MGM_CHANNELS)

        # 预测状态 (用于预测编码)
        self._prediction: Optional[np.ndarray] = None

    def process(self, ic_output: dict,
                arousal: float = 0.8,
                fpn: Optional[object] = None) -> dict:
        """IC输出 → MGB中继 → 听皮层.

        Args:
            ic_output: IC输出 dict
            arousal: 脑干唤醒度 [0, 1]
            fpn: FPN模块 (用于注意力调制, 可选)

        Returns:
            dict with:
              'tonotopic': MGv输出 → A1 (N_MGV_CHANNELS,)
              'complex': MGd输出 → Belt (N_MGD_CHANNELS,)
              'diffuse': MGm输出 → 多感官/弥散 (N_MGM_CHANNELS,)
              'relay': 综合中继向量 (N_MGV+N_MGD+N_MGM,)
        """
        ic_integrated = ic_output['integrated']
        ic_fm = ic_output.get('fm_sweep', None)
        ic_multisensory = ic_output.get('multisensory', None)

        # MGv: tonotopic中继
        tono_relay = self.mgv.process(ic_integrated, arousal=arousal)

        # MGd: 复杂特征中继
        complex_relay = self.mgd.process(ic_integrated, ic_fm_sweep=ic_fm,
                                         arousal=arousal)

        # MGm: 多感官弥散
        diffuse_relay = self.mgm.process(multisensory_input=ic_multisensory,
                                         arousal=arousal)

        # ---- FPN 注意调制 ----
        if fpn is not None and hasattr(fpn, 'attention_template'):
            # FPN探照灯增强任务相关通道
            template = np.asarray(fpn.attention_template, dtype=np.float32)
            # 使用模板的后段调制听觉 (视觉段在前, 听觉段在后)
            # FPN模板D=372, 取可用部分
            attn_strength = float(np.mean(np.abs(template))) * 0.3 + 0.7
            tono_relay = tono_relay * attn_strength
            complex_relay = complex_relay * attn_strength

        # ---- 综合中继 ----
        relay = np.concatenate([
            tono_relay,
            complex_relay,
            diffuse_relay,
        ]).astype(np.float32)

        # ---- 预测编码 ----
        if self._prediction is None:
            self._prediction = relay.copy()

        return {
            'tonotopic': tono_relay,
            'complex': complex_relay,
            'diffuse': diffuse_relay,
            'relay': relay,
        }

    def get_prediction(self) -> np.ndarray:
        """返回当前预测 (供听皮层反馈使用)."""
        if self._prediction is None:
            return np.zeros(N_MGV_CHANNELS + N_MGD_CHANNELS + N_MGM_CHANNELS,
                           dtype=np.float32)
        return self._prediction.copy()

    def receive_feedback(self, prediction_error: np.ndarray,
                         lr: float = 0.1):
        """接收听皮层的预测误差反馈.

        Args:
            prediction_error: 听皮层计算的预测误差
            lr: 学习率
        """
        if self._prediction is not None:
            pe = np.asarray(prediction_error, dtype=np.float32).ravel()
            if len(pe) == len(self._prediction):
                self._prediction += lr * pe

    def compute_prediction_error(self, relay: np.ndarray) -> np.ndarray:
        """计算中继信号的预测误差."""
        if self._prediction is None:
            return np.zeros_like(relay)
        return (relay - self._prediction).astype(np.float32)
