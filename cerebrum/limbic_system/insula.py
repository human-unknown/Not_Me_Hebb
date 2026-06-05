"""
insula.py — 岛叶 (Insula / Insular Cortex) [v5.4]

对应脑区: 岛叶 (Brodmann 13-16)
所属层级: 大脑 → 边缘系统 / 岛叶
脑区标签: 前岛叶 (AI) · 后岛叶 (PI) · 中岛叶

功能职责:
  - 内感受 (Interoception) — 身体内部状态的感知
  - 痛觉的情感-动机评估 (疼痛有多"难受")
  - 内感受预测编码 — 身体状态的生成模型
  - 突显网络核心节点 (与dACC协作)
  - 情绪意识 — 将身体信号转化为情感体验

岛叶三分区:
  - 后岛叶 (PI): 初级内感受皮层 — 接收丘脑/脊髓上传的身体信号
                    (心跳、呼吸、胃肠、疼痛强度)
  - 中岛叶 (MI): 内感受整合 — 身体状态的多模态融合
  - 前岛叶 (AI): 高阶内感受 — 情感意识、主观感受、
                  突显网络核心 (与dACC协作切换DMN/TPN)

知觉规律:
  1. 内感受精度律 — 岛叶编码身体状态的precision (不确定性)
  2. 痛觉情感律 — 前岛叶将疼痛强度转化为"难受"体验
  3. 内感受预测编码 — 身体信号的生成模型, 预测误差 → 突显
  4. 岛叶-ACC耦合律 — AI+dACC = 突显网络, 检测显著事件

在 NotMe 中的应用:
  - 接收脊髓背角/丘脑的痛觉上行信号
  - 将身体信号转化为内感受表征
  - 计算痛觉的情感-动机分量 (疼痛有多"不愉快")
  - 输出到 ACC (突显网络) + PAG (下行调控)
  - 身体状态预测编码 → interoceptive prediction error
"""

import numpy as np
from typing import Optional, Tuple

# ============================================================
# 常量
# ============================================================

N_INSULA_OUTPUT = 16          # 岛叶输出维度
N_INTERO_CHANNELS = 12        # 内感受通道数
N_AFFECT_CHANNELS = 8         # 情感评估通道数


class PosteriorInsula:
    """后岛叶 (PI) — 初级内感受皮层.

    接收来自丘脑 (VMb/VPI) 的身体信号:
      - 疼痛强度/位置
      - 内脏状态 (心跳, 呼吸, 胃肠)
      - 温度感觉
    保持对身体状态的客观表征 (感觉-辨别).
    """

    def __init__(self, n_channels: int = N_INTERO_CHANNELS):
        self.n_channels = n_channels
        # 身体状态表征
        self._body_state: np.ndarray = np.zeros(n_channels, dtype=np.float32)
        # 内感受精度 (逆方差) — 对身体信号的置信度
        self._interoceptive_precision: np.ndarray = np.ones(n_channels,
                                                            dtype=np.float32) * 0.5

    def process(self, pain_signal: np.ndarray,
                body_vector: Optional[np.ndarray] = None,
                thalamic_relay: Optional[np.ndarray] = None) -> dict:
        """后岛叶内感受处理.

        Args:
            pain_signal: 来自丘脑的痛觉信号
            body_vector: 身体状态向量 (来自BodyVector)
            thalamic_relay: 丘脑中继信号

        Returns:
            dict with body_map, precision
        """
        pain = np.asarray(pain_signal, dtype=np.float32).ravel()

        # 将痛觉+身体信号映射到内感受通道
        interoceptive = np.zeros(self.n_channels, dtype=np.float32)

        # 前4通道: 痛觉强度/性质 (来自spinothalamic)
        n_pain = min(len(pain), 4)
        interoceptive[:n_pain] = pain[:n_pain]

        # 中4通道: 身体状态 (来自BodyVector)
        if body_vector is not None:
            bv = np.asarray(body_vector, dtype=np.float32).ravel()
            n_bv = min(len(bv), 4)
            interoceptive[4:4+n_bv] = bv[:n_bv]

        # 后4通道: 丘脑中继
        if thalamic_relay is not None:
            tr = np.asarray(thalamic_relay, dtype=np.float32).ravel()
            n_tr = min(len(tr), 4)
            interoceptive[8:8+n_tr] = tr[:n_tr]

        # 内感受精度: 基于信号一致性
        signal_consistency = 1.0 / (1.0 + np.std(interoceptive))
        self._interoceptive_precision = (
            0.8 * self._interoceptive_precision
            + 0.2 * signal_consistency)

        # 身体状态EMA
        self._body_state = (0.7 * self._body_state
                            + 0.3 * interoceptive)

        return {
            'body_map': self._body_state.copy(),
            'interoceptive_precision': self._interoceptive_precision.copy(),
            'raw_interoceptive': interoceptive,
        }


class AnteriorInsula:
    """前岛叶 (AI) — 高阶内感受 + 情感意识.

    将身体信号转化为:
      - 主观情感体验 (身体感觉 → "感受")
      - 疼痛的不愉快度 (疼痛强度 → "有多难受")
      - 突显检测 (内感受变化 → "需要注意")
      - 风险/不确定性评估

    AI是突显网络核心节点 — 与dACC协作:
      AI: "发生了什么" (内感受突显)
      dACC: "该做什么" (行动选择/冲突监测)
    """

    def __init__(self, n_affect: int = N_AFFECT_CHANNELS):
        self.n_affect = n_affect
        # 内感受→情感映射权重 (Hebb学习)
        self._affect_mapping: np.ndarray = np.eye(
            N_INTERO_CHANNELS, n_affect, dtype=np.float32)[:N_INTERO_CHANNELS, :n_affect] * 0.5

        # 情感评估状态
        self._affective_state: np.ndarray = np.zeros(n_affect, dtype=np.float32)
        # 不愉快度 [0, 1]
        self._unpleasantness: float = 0.0
        # 内感受突显 [0, 1]
        self._interoceptive_salience: float = 0.0
        # 身体预测误差
        self._body_prediction_error: float = 0.0

        # 预测编码: 对身体状态的内部预测
        self._body_prediction: Optional[np.ndarray] = None

    def process(self, posterior_output: dict,
                pain_intensity: float = 0.0,
                valence: float = 0.0,
                arousal: float = 0.5) -> dict:
        """前岛叶情感评估.

        Args:
            posterior_output: 后岛叶输出 dict
            pain_intensity: 疼痛强度 [0, 1]
            valence: 当前效价 [-1, 1]
            arousal: 当前唤醒 [0, 1]

        Returns:
            dict with affective_state, unpleasantness, salience
        """
        body_map = posterior_output.get('body_map',
                                        np.zeros(N_INTERO_CHANNELS, dtype=np.float32))
        precision = posterior_output.get('interoceptive_precision',
                                         np.ones(N_INTERO_CHANNELS, dtype=np.float32) * 0.5)

        # ---- 身体预测编码 ----
        if self._body_prediction is None:
            self._body_prediction = body_map.copy()
        else:
            # 预测更新
            self._body_prediction = (0.9 * self._body_prediction
                                     + 0.1 * body_map)

        # 身体预测误差
        body_pe = body_map - self._body_prediction
        self._body_prediction_error = float(
            np.mean(np.abs(body_pe)) / (np.mean(precision) + 1e-8))

        # ---- 内感受→情感映射 ----
        # 身体信号通过习得映射转化为情感表征
        body_padded = np.pad(body_map, (0, max(0,
                             N_INTERO_CHANNELS - len(body_map))))[:N_INTERO_CHANNELS]
        mapped = body_padded @ self._affect_mapping[:N_INTERO_CHANNELS, :]

        # 情感状态EMA
        self._affective_state = (0.7 * self._affective_state
                                 + 0.3 * mapped)

        # ---- 不愉快度 (Unpleasantness) ----
        # 疼痛的情感维度: 不仅取决于强度, 还受上下文调制
        # 负效价 + 高疼痛 → 极度不愉快
        # 正效价 + 低唤醒 → 即使有疼痛也不那么难受
        base_unpleasantness = pain_intensity * 0.7
        valence_mod = max(0.0, -valence) * 0.3   # 负效价放大不愉快
        arousal_mod = arousal * 0.2               # 高唤醒放大感受

        self._unpleasantness = float(np.clip(
            0.8 * self._unpleasantness
            + 0.2 * (base_unpleasantness + valence_mod + arousal_mod),
            0.0, 1.0))

        # ---- 内感受突显 ----
        # 身体预测误差大 → 内感受突显 → 吸引注意
        # 高精度 → 高置信 → 更需要关注 (贝叶斯突显)
        raw_salience = (
            0.5 * self._body_prediction_error +
            0.3 * pain_intensity +
            0.2 * np.mean(precision)
        )
        self._interoceptive_salience = float(np.clip(
            0.7 * self._interoceptive_salience + 0.3 * raw_salience,
            0.0, 1.0))

        return {
            'affective_state': self._affective_state.copy(),
            'unpleasantness': self._unpleasantness,
            'interoceptive_salience': self._interoceptive_salience,
            'body_prediction_error': self._body_prediction_error,
            'body_prediction': self._body_prediction.copy() if self._body_prediction is not None else None,
        }

    def update_affect_mapping(self, interoceptive: np.ndarray,
                              affective: np.ndarray, lr: float = 0.01):
        """Hebb学习: 更新内感受→情感映射权重.

        一起激活的内感受通道和情感通道 → 连接增强.
        """
        intero = np.asarray(interoceptive, dtype=np.float32).ravel()
        affec = np.asarray(affective, dtype=np.float32).ravel()

        n_i = min(len(intero), N_INTERO_CHANNELS)
        n_a = min(len(affec), self.n_affect)

        # Hebb outer product update
        delta = lr * np.outer(intero[:n_i], affec[:n_a])
        self._affect_mapping[:n_i, :n_a] += delta
        # 归一化
        norms = np.linalg.norm(self._affect_mapping, axis=1, keepdims=True) + 1e-8
        self._affect_mapping /= norms


class Insula:
    """岛叶 — 内感受 + 痛觉情感评估.

    组装后岛叶 (PI, 内感受) + 前岛叶 (AI, 情感意识).

    用法:
      insula = Insula()
      output = insula.process(
          pain_signal=dh_output['pain_signal'],
          body_vector=body.b,
          pain_intensity=0.6,
          valence=-0.3,
          arousal=0.7,
      )
      # output['unpleasantness'] → 疼痛有多"难受"
      # output['interoceptive_salience'] → 汇入突显网络 (ACC)
      # output['affective_state'] → 汇入PAG (下行调控)
    """

    def __init__(self):
        self.posterior = PosteriorInsula()
        self.anterior = AnteriorInsula()

        # 岛叶整体激活 (用于输出)
        self._activation: np.ndarray = np.zeros(N_INSULA_OUTPUT, dtype=np.float32)

    def process(self,
                pain_signal: np.ndarray,
                body_vector: Optional[np.ndarray] = None,
                thalamic_relay: Optional[np.ndarray] = None,
                pain_intensity: float = 0.0,
                valence: float = 0.0,
                arousal: float = 0.5) -> dict:
        """岛叶单步处理.

        Args:
            pain_signal: 痛觉上行信号 (来自丘脑/脊髓)
            body_vector: 身体状态向量
            thalamic_relay: 丘脑中继信号
            pain_intensity: 疼痛强度 [0, 1]
            valence: 当前效价 [-1, 1]
            arousal: 当前唤醒 [0, 1]

        Returns:
            dict with:
              'body_map': 后岛叶身体状态图
              'unpleasantness': 疼痛不愉快度 [0, 1]
              'interoceptive_salience': 内感受突显 [0, 1]
              'affective_state': 情感评估向量
              'body_prediction_error': 身体预测误差
              'activation': 岛叶整体激活向量
        """
        # ---- 后岛叶: 内感受 ----
        pi_out = self.posterior.process(
            pain_signal=pain_signal,
            body_vector=body_vector,
            thalamic_relay=thalamic_relay,
        )

        # ---- 前岛叶: 情感评估 ----
        ai_out = self.anterior.process(
            posterior_output=pi_out,
            pain_intensity=pain_intensity,
            valence=valence,
            arousal=arousal,
        )

        # ---- 整体激活向量 ----
        self._activation = np.concatenate([
            pi_out['body_map'][:8],
            ai_out['affective_state'][:8],
        ]).astype(np.float32)

        return {
            'body_map': pi_out['body_map'],
            'interoceptive_precision': pi_out['interoceptive_precision'],
            'unpleasantness': ai_out['unpleasantness'],
            'interoceptive_salience': ai_out['interoceptive_salience'],
            'affective_state': ai_out['affective_state'],
            'body_prediction_error': ai_out['body_prediction_error'],
            'activation': self._activation.copy(),
        }

    def get_interoceptive_salience(self) -> float:
        """获取内感受突显信号 → 汇入ACC突显网络."""
        return self.anterior._interoceptive_salience

    def get_unpleasantness(self) -> float:
        """获取疼痛不愉快度 → 汇入PAG下行调控."""
        return self.anterior._unpleasantness

    # ================================================================
    # 预测编码接口
    # ================================================================

    def get_body_prediction(self) -> Optional[np.ndarray]:
        """返回身体状态预测."""
        return self.anterior._body_prediction

    def compute_interoceptive_pe(self) -> float:
        """计算内感受预测误差."""
        return self.anterior._body_prediction_error
