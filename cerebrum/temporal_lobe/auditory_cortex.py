"""
auditory_cortex.py — 听皮层 (Auditory Cortex) [v5.2 完整实现]

对应脑区: BA41 (初级听皮层 A1 / Heschl's gyrus) + BA42 (次级听皮层 Belt)
          + BA22前部 (Parabelt / 听觉联合皮层)
所属层级: 大脑 → 颞叶 (Level 3)

功能职责:
  - A1 (Core / BA41):      tonotopic组织 + 频率调谐 + 双耳空间处理
  - Belt (BA42):           FM检测 + 谐波分析 + 音色特征 + 复杂频谱调谐
  - Parabelt (BA22前):     听觉场景分析 (Bregman ASA) + 流分离
  - "What" 腹侧流:         声音识别 → Wernicke区 (语义)
  - "Where" 背侧流:        声源空间定位 → 顶叶FPN (空间注意)
  - 预测编码:              自上而下预测 → MGB/IC
  - Hebb可塑性:            听觉记忆形成 (经验依赖可塑性)

三层结构:
  ┌─────────────────────────────────────┐
  │ Parabelt (旁带区)                    │
  │  听觉场景分析 | 流分离 | What/Where   │
  │  ┌───────────────────────────────┐  │
  │  │ Belt (带区)                     │  │
  │  │  FM | 谐波 | 音色 | 频谱组合     │  │
  │  │  ┌─────────────────────────┐  │  │
  │  │  │ A1 (核心区 / BA41)        │  │  │
  │  │  │  tonotopic | 频率调谐     │  │  │
  │  │  └─────────────────────────┘  │  │
  │  └───────────────────────────────┘  │
  └─────────────────────────────────────┘

知觉规律:
  1. 音调拓扑组织 — A1维持频率→空间映射
  2. 听觉场景分析 (ASA) — 谐波性/共同起始/空间连贯/FM连贯
  3. 连续性错觉 — 被中断声音仍知觉为连续
  4. "What"/"Where"双流 — 腹侧语义 + 背侧空间
  5. 预测编码 — 自上而下预期
  6. 经验依赖可塑性 — Hebb学习重组tonotopic map

参考:
  - Rauschecker & Tian (2000). Mechanisms and streams for processing
    of "what" and "where" in auditory cortex.
  - Bregman, A. S. (1990). Auditory Scene Analysis.
  - Griffiths & Warren (2002). The planum temporale as a computational hub.
"""

import numpy as np
from typing import Optional, Tuple

# ============================================================
# 常量
# ============================================================

N_A1_CHANNELS = 24            # A1 tonotopic输出
N_BELT_CHANNELS = 16          # Belt复杂特征输出
N_PARABELT_WHAT = 16          # Parabelt "What"流
N_PARABELT_WHERE = 8          # Parabelt "Where"流
N_ASA_STREAMS = 4             # 听觉流分离数 (最多追踪4个声源)

# 听觉场景分析分组线索权重
ASA_WEIGHTS = {
    'harmonicity': 0.35,       # 谐波性: 基频整倍数关系
    'common_onset': 0.30,      # 共同起始: 同时开始
    'spatial_coherence': 0.20, # 空间连贯: 相同位置
    'fm_coherence': 0.15,      # FM连贯: 同步频率调制
}


# ============================================================
# A1 — 初级听皮层 (Core)
# ============================================================

class PrimaryAuditoryCortex:
    """A1 — 初级听皮层 (BA41, Heschl's gyrus).

    tonotopic组织 + 窄带频率调谐 + 双耳空间处理.
    接收MGv的精确tonotopic输入.
    """

    def __init__(self, n_channels: int = N_A1_CHANNELS):
        self.n_channels = n_channels
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

        # 频率调谐曲线 (tonotopic map): 每个通道的最佳频率
        self._best_frequencies = np.linspace(0, 1, n_channels,
                                             dtype=np.float32)
        # 调谐锐度 (更窄=更精确的频率选择性)
        self._tuning_sharpness: float = 3.0

        # 预测状态
        self._prediction: Optional[np.ndarray] = None

        # Hebb可塑性: 经验依赖的频率调谐偏移
        self._plasticity_trace: np.ndarray = np.zeros(n_channels,
                                                       dtype=np.float32)

    def process(self, mgv_input: np.ndarray,
                arousal: float = 0.8) -> dict:
        """处理MGv输入, 输出A1激活模式.

        Args:
            mgv_input: MGv tonotopic输出 (N_MGV_CHANNELS,)
            arousal: 脑干唤醒度 [0, 1]

        Returns:
            dict with:
              'tonotopic': A1频率调谐激活
              'frequency_map': 激活的频率分布
        """
        mgv = np.asarray(mgv_input, dtype=np.float32).ravel()

        # 频率调谐: 锐化MGv输入
        # 侧抑制: 增强频率对比度 (相邻通道互相抑制)
        tuned = np.zeros(self.n_channels, dtype=np.float32)

        # 确保维度匹配
        if len(mgv) != self.n_channels:
            tmp = np.zeros(self.n_channels, dtype=np.float32)
            n = min(len(mgv), self.n_channels)
            tmp[:n] = mgv[:n]
            mgv = tmp

        for i in range(self.n_channels):
            # 高斯调谐: 每个A1神经元对特定频率范围敏感
            for j in range(self.n_channels):
                freq_dist = abs(i - j) / max(1, self.n_channels)
                tuned[i] += mgv[j] * np.exp(
                    -0.5 * (freq_dist * self._tuning_sharpness) ** 2)

        # 侧抑制: 减去邻居的平均激活
        for i in range(self.n_channels):
            neighbors = []
            if i > 0:
                neighbors.append(tuned[i - 1])
            if i < self.n_channels - 1:
                neighbors.append(tuned[i + 1])
            if neighbors:
                inhibition = np.mean(neighbors) * 0.3
                tuned[i] = max(0.0, tuned[i] - inhibition)

        # 唤醒度调制
        tuned = tuned * float(np.clip(arousal, 0.2, 1.0))

        # 时间平滑
        self._activation = 0.7 * self._activation + 0.3 * tuned

        # Hebb可塑性追踪: 频繁激活的通道增强
        self._plasticity_trace = (0.95 * self._plasticity_trace +
                                  0.05 * self._activation)

        # 预测更新
        if self._prediction is None:
            self._prediction = self._activation.copy()

        return {
            'tonotopic': self._activation.copy(),
            'frequency_map': self._activation.copy(),
        }

    def compute_prediction_error(self) -> np.ndarray:
        """计算A1的预测误差."""
        if self._prediction is None:
            return np.zeros(self.n_channels, dtype=np.float32)
        return (self._activation - self._prediction).astype(np.float32)

    def receive_feedback(self, prediction: np.ndarray, lr: float = 0.1):
        """接收Belt自上而下的预测."""
        if self._prediction is not None:
            pred = np.asarray(prediction, dtype=np.float32).ravel()
            if len(pred) == len(self._prediction):
                self._prediction = (1.0 - lr) * self._prediction + lr * pred


# ============================================================
# Belt — 带区 (次级听皮层)
# ============================================================

class BeltArea:
    """Belt — 带区听皮层 (BA42).

    围绕A1核心区.
    更复杂的特征检测: FM扫频、谐波结构、音色、频谱组合.
    接收MGd输入 + A1前馈.
    """

    def __init__(self, n_channels: int = N_BELT_CHANNELS):
        self.n_channels = n_channels
        self._activation: np.ndarray = np.zeros(n_channels, dtype=np.float32)

        # 特征检测器: FM检测 | 谐波分析 | 音色
        self._fm_response: np.ndarray = np.zeros(n_channels, dtype=np.float32)
        self._harmonic_response: np.ndarray = np.zeros(n_channels // 2,
                                                       dtype=np.float32)
        self._timbre_response: np.ndarray = np.zeros(n_channels // 2,
                                                      dtype=np.float32)

        # 预测状态
        self._prediction: Optional[np.ndarray] = None

    def process(self, a1_output: dict, mgd_input: np.ndarray,
                ic_fm_sweep: Optional[np.ndarray] = None) -> dict:
        """处理A1 + MGd输入, 提取复杂特征.

        Args:
            a1_output: A1输出 dict
            mgd_input: MGd复杂特征输出
            ic_fm_sweep: IC FM扫频方向 (可选)

        Returns:
            dict with:
              'complex_features': 复杂听觉特征
              'fm': FM检测响应
              'harmonic': 谐波分析
              'timbre': 音色特征
        """
        a1 = a1_output['tonotopic']
        mgd = np.asarray(mgd_input, dtype=np.float32).ravel()

        # 确保维度匹配
        if len(a1) != self.n_channels:
            tmp = np.zeros(self.n_channels, dtype=np.float32)
            n = min(len(a1), self.n_channels)
            tmp[:n] = a1[:n]
            a1 = tmp

        # ---- FM扫频响应 ----
        # Belt神经元对FM扫频方向和速度有选择性
        if ic_fm_sweep is not None:
            fm = np.asarray(ic_fm_sweep, dtype=np.float32).ravel()
            fm_in = np.zeros(self.n_channels, dtype=np.float32)
            n = min(len(fm), self.n_channels)
            fm_in[:n] = fm[:n]
            self._fm_response = 0.7 * self._fm_response + 0.3 * fm_in
        else:
            self._fm_response = 0.7 * self._fm_response

        # ---- 谐波分析 ----
        # 检测频率成分是否呈谐波关系 (f, 2f, 3f, 4f...)
        # 谐波性 = 频率通道激活模式的周期性
        half = self.n_channels // 2
        if len(a1) >= 4:
            # 自相关法: 检测频谱中的等间距峰值
            for lag in range(1, half):
                seg_a = a1[:half]
                seg_b = np.roll(a1[:half], -lag)
                std_a, std_b = np.std(seg_a), np.std(seg_b)
                if std_a > 1e-8 and std_b > 1e-8:
                    corr = np.corrcoef(seg_a, seg_b)[0, 1]
                    if not np.isnan(corr):
                        self._harmonic_response[lag % half] += 0.1 * max(0.0, corr)
            self._harmonic_response = np.clip(self._harmonic_response, 0.0, 1.0)

        # ---- 音色特征 ----
        # 音色由频谱包络和共振峰结构决定
        # 频谱能量分布 + 高频/低频比
        low_energy = float(np.mean(np.abs(a1[:self.n_channels // 2])))
        high_energy = float(np.mean(np.abs(a1[self.n_channels // 2:])))
        spectral_tilt = float(np.tanh((high_energy - low_energy) * 3.0))
        # 频谱平坦度 (噪声-like vs 纯音-like)
        spectral_flatness = float(
            np.exp(np.mean(np.log(np.abs(a1) + 1e-8))) /
            (np.mean(np.abs(a1)) + 1e-8)
        )

        self._timbre_response = np.array(
            [spectral_tilt, spectral_flatness] * (half // 2 + 1),
            dtype=np.float32)[:half]

        # ---- 整合复杂特征 ----
        # MGd前馈 + A1前馈 + FM + 谐波 + 音色
        mgd_in = np.zeros(self.n_channels, dtype=np.float32)
        n_mgd = min(len(mgd), self.n_channels)
        mgd_in[:n_mgd] = mgd[:n_mgd]

        complex_features = (
            0.3 * a1 +
            0.3 * mgd_in +
            0.2 * self._fm_response +
            0.1 * np.pad(self._harmonic_response,
                         (0, self.n_channels - len(self._harmonic_response))) +
            0.1 * np.pad(self._timbre_response,
                         (0, self.n_channels - len(self._timbre_response)))
        )

        # 时间平滑
        self._activation = 0.75 * self._activation + 0.25 * complex_features

        # 预测更新
        if self._prediction is None:
            self._prediction = self._activation.copy()

        return {
            'complex_features': self._activation.copy(),
            'fm': self._fm_response.copy(),
            'harmonic': self._harmonic_response.copy(),
            'timbre': self._timbre_response.copy(),
        }

    def predict_to_A1(self) -> np.ndarray:
        """Belt → A1 自上而下预测."""
        return self._activation.copy()

    def compute_prediction_error(self) -> np.ndarray:
        """计算Belt的预测误差."""
        if self._prediction is None:
            return np.zeros(self.n_channels, dtype=np.float32)
        return (self._activation - self._prediction).astype(np.float32)

    def receive_feedback(self, prediction: np.ndarray, lr: float = 0.1):
        """接收Parabelt自上而下的预测."""
        if self._prediction is not None:
            pred = np.asarray(prediction, dtype=np.float32).ravel()
            if len(pred) == len(self._prediction):
                self._prediction = (1.0 - lr) * self._prediction + lr * pred


# ============================================================
# Parabelt — 旁带区 (听觉联合皮层)
# ============================================================

class ParabeltArea:
    """Parabelt — 旁带区 (BA22前部, 听觉联合皮层).

    听觉场景分析 (Auditory Scene Analysis):
      - 将混合声波分离为独立的"听觉流" (声源)
      - 4大分组线索: 谐波性 | 共同起始 | 空间连贯 | FM连贯
      - "What"腹侧流 → Wernicke区 (声音识别/语义)
      - "Where"背侧流 → 顶叶/FPN (声源定位/空间注意)

    知觉规律:
      1. 和声连续律 (Harmonicity)
      2. 共同起始/终止律 (Common Onset/Offset)
      3. 空间连贯律 (Spatial Coherence)
      4. 频率调制连贯律 (FM Coherence)
      5. 连续律 (Continuity) — 被中断的声音仍知觉为连续
    """

    def __init__(self, n_what: int = N_PARABELT_WHAT,
                 n_where: int = N_PARABELT_WHERE,
                 n_streams: int = N_ASA_STREAMS):
        self.n_what = n_what
        self.n_where = n_where
        self.n_streams = n_streams

        # 听觉流追踪: 每个流有独立的特征状态
        self._streams: np.ndarray = np.zeros((n_streams, n_what),
                                              dtype=np.float32)
        self._stream_spatial: np.ndarray = np.zeros((n_streams, n_where),
                                                     dtype=np.float32)
        self._stream_activity: np.ndarray = np.ones(n_streams,
                                                     dtype=np.float32) * 0.01

        # 前一帧特征 (用于连续性检测)
        self._prev_features: Optional[np.ndarray] = None
        self._prev_spatial: Optional[np.ndarray] = None

        # "What"流: 腹侧通路 → Wernicke (声音对象识别)
        self._what_stream: np.ndarray = np.zeros(n_what, dtype=np.float32)

        # "Where"流: 背侧通路 → 顶叶/FPN (空间位置)
        self._where_stream: np.ndarray = np.zeros(n_where, dtype=np.float32)

        # 预测状态
        self._prediction: Optional[np.ndarray] = None

    def process(self, belt_output: dict,
                a1_output: dict,
                soc_spatial: Optional[np.ndarray] = None,
                ic_multisensory: Optional[np.ndarray] = None) -> dict:
        """听觉场景分析 + 双流分离.

        Args:
            belt_output: Belt输出 dict
            a1_output: A1输出 dict
            soc_spatial: SOC空间信息
            ic_multisensory: IC多感官整合

        Returns:
            dict with:
              'what_stream': "What"流 — 声音对象特征 → Wernicke
              'where_stream': "Where"流 — 空间位置 → 顶叶
              'streams': 分离的听觉流
              'asa_grouping': ASA分组线索激活
        """
        features = belt_output['complex_features']
        harmonic = belt_output.get('harmonic', np.zeros(N_BELT_CHANNELS // 2))
        spatial = (np.asarray(soc_spatial, dtype=np.float32).ravel()
                   if soc_spatial is not None else np.zeros(N_PARABELT_WHERE))

        # 确保特征维度匹配
        features = np.asarray(features, dtype=np.float32).ravel()
        if len(features) != self.n_what:
            tmp = np.zeros(self.n_what, dtype=np.float32)
            n = min(len(features), self.n_what)
            tmp[:n] = features[:n]
            features = tmp

        # ---- 听觉场景分析 (ASA): 计算分组线索 ----
        asa_cues = self._compute_asa_cues(features, harmonic, spatial)

        # ---- 流分离 (Stream Segregation) ----
        # 基于ASA线索将特征分配到不同听觉流
        self._assign_to_streams(features, spatial, asa_cues)

        # ---- "What"腹侧流: 声音对象识别 ----
        # 综合所有活跃流 → 统一的声音对象表征
        active_mask = self._stream_activity > 0.1
        if np.any(active_mask):
            self._what_stream = np.mean(
                self._streams[active_mask], axis=0)
        else:
            self._what_stream = features

        # ---- "Where"背侧流: 声源空间位置 ----
        # 综合空间信息 → 声源定位
        if soc_spatial is not None:
            spatial = np.asarray(soc_spatial, dtype=np.float32).ravel()
            if len(spatial) > self.n_where:
                self._where_stream = spatial[:self.n_where]
            else:
                self._where_stream = np.zeros(self.n_where, dtype=np.float32)
                self._where_stream[:len(spatial)] = spatial

        # ---- 连续性错觉 (Continuity Illusion) ----
        # 短暂中断的声音被知觉为连续 (基于预测填补)
        if self._prev_features is not None:
            continuity_mask = self._compute_continuity(features)
            # 连续性补全: 中断部分用预测填补
            self._what_stream = (0.7 * self._what_stream +
                                 0.3 * continuity_mask * self._prev_features[:self.n_what])

        self._prev_features = features.copy()
        if soc_spatial is not None:
            self._prev_spatial = spatial.copy()

        # 预测更新
        if self._prediction is None:
            self._prediction = self._what_stream.copy()

        return {
            'what_stream': self._what_stream.copy(),
            'where_stream': self._where_stream.copy(),
            'streams': self._streams.copy(),
            'asa_grouping': asa_cues,
            'n_active_streams': int(np.sum(active_mask)),
        }

    def _compute_asa_cues(self, features: np.ndarray,
                          harmonic: np.ndarray,
                          spatial: np.ndarray) -> dict:
        """计算听觉场景分析的4大分组线索.

        Returns:
            dict: 各线索的激活强度 [0, 1]
        """
        n_f = len(features)

        # 1. 谐波性 (Harmonicity): 频谱中等间距峰值
        harmonicity = float(np.clip(np.mean(np.abs(harmonic)), 0.0, 1.0))

        # 2. 共同起始 (Common Onset): 特征突变检测
        if self._prev_features is not None:
            onset_sync = float(np.mean(
                np.abs(features[:n_f] - self._prev_features[:n_f])))
            onset_sync = float(np.tanh(onset_sync * 5.0))
        else:
            onset_sync = 0.5

        # 3. 空间连贯 (Spatial Coherence): 空间位置一致性
        spatial_coherence = float(np.clip(np.mean(np.abs(spatial)), 0.0, 1.0))

        # 4. FM连贯 (FM Coherence): 频率调制同步性
        # 用特征的时间变化的一致性来表示
        if self._prev_features is not None and len(features) >= 2:
            curr_fm = np.diff(features[:n_f])
            prev_fm = np.diff(self._prev_features[:n_f])
            if len(curr_fm) > 0 and len(prev_fm) > 0:
                n = min(len(curr_fm), len(prev_fm))
                fm_coherence = float(np.clip(
                    np.corrcoef(curr_fm[:n], prev_fm[:n])[0, 1]
                    if n > 2 else 0.5, 0.0, 1.0))
            else:
                fm_coherence = 0.5
        else:
            fm_coherence = 0.5

        return {
            'harmonicity': harmonicity,
            'common_onset': onset_sync,
            'spatial_coherence': spatial_coherence,
            'fm_coherence': fm_coherence,
        }

    def _assign_to_streams(self, features: np.ndarray,
                           spatial: np.ndarray,
                           asa_cues: dict):
        """基于ASA线索将特征分配到听觉流.

        每个流是一个持续的"声音对象"追踪器。
        新特征匹配已有流或创建新流。
        """
        # 计算特征与各流的相似度
        similarities = np.zeros(self.n_streams, dtype=np.float32)
        for i in range(self.n_streams):
            if self._stream_activity[i] > 0.01:
                # 余弦相似度
                dot = np.dot(features, self._streams[i])
                n1 = np.linalg.norm(features)
                n2 = np.linalg.norm(self._streams[i])
                similarities[i] = dot / (n1 * n2 + 1e-8)

        # 最佳匹配流
        best_match = int(np.argmax(similarities))
        best_sim = similarities[best_match]

        # ASA综合分组强度
        grouping_strength = (
            ASA_WEIGHTS['harmonicity'] * asa_cues['harmonicity'] +
            ASA_WEIGHTS['common_onset'] * asa_cues['common_onset'] +
            ASA_WEIGHTS['spatial_coherence'] * asa_cues['spatial_coherence'] +
            ASA_WEIGHTS['fm_coherence'] * asa_cues['fm_coherence']
        )

        if best_sim > 0.3 and self._stream_activity[best_match] > 0.01:
            # 匹配已有流 → 更新
            alpha = 0.3 * (0.5 + 0.5 * grouping_strength)
            self._streams[best_match] = (
                (1.0 - alpha) * self._streams[best_match] +
                alpha * features
            )
            self._stream_activity[best_match] = min(1.0,
                self._stream_activity[best_match] + 0.1)
        else:
            # 创建新流 (替换最不活跃的)
            min_idx = int(np.argmin(self._stream_activity))
            self._streams[min_idx] = features.copy()
            self._stream_spatial[min_idx] = spatial[:self.n_where] if len(
                spatial) >= self.n_where else np.pad(
                spatial, (0, max(0, self.n_where - len(spatial))))
            self._stream_activity[min_idx] = 0.3

        # 所有流活动衰减
        self._stream_activity *= 0.95
        # 最低活动度 → 流消失
        self._stream_activity = np.maximum(1e-4, self._stream_activity)

    def _compute_continuity(self, current_features: np.ndarray) -> float:
        """连续性检测: 判断当前声音是否是前一帧的连续.

        即使短暂中断，知觉上仍是连续的。
        Returns: 连续性掩码 [0, 1]
        """
        if self._prev_features is None:
            return 0.0

        n = min(len(current_features), len(self._prev_features))
        # 相似度高 → 连续性高
        similarity = float(np.dot(
            current_features[:n], self._prev_features[:n]) /
            (np.linalg.norm(current_features[:n]) *
             np.linalg.norm(self._prev_features[:n]) + 1e-8))

        # 即使当前特征弱 (中断), 只要与前一帧相似 → 连续性
        current_strength = float(np.mean(np.abs(current_features)))
        return float(np.clip(similarity * (0.3 + 0.7 * (1.0 - current_strength)),
                             0.0, 1.0))

    def predict_to_Belt(self) -> np.ndarray:
        """Parabelt → Belt 自上而下预测."""
        return self._what_stream.copy()

    def compute_prediction_error(self) -> np.ndarray:
        """计算Parabelt的预测误差."""
        if self._prediction is None:
            return np.zeros(self.n_what, dtype=np.float32)
        return (self._what_stream - self._prediction).astype(np.float32)


# ============================================================
# AuditoryCortex — 听皮层整合
# ============================================================

class AuditoryCortex:
    """听皮层 — 三层层级整合 (A1 → Belt → Parabelt).

    组装A1 (核心), Belt (带区), Parabelt (旁带区),
    实现预测编码全通路 + 双流分离 + 听觉场景分析.

    用法:
      ac = AuditoryCortex()
      output = ac.process(mgb_output, soc_spatial, ic_output)
      # output['what_stream']  → Wernicke区 (语义理解)
      # output['where_stream'] → 顶叶/FPN (空间注意)
      # output['scene']        → 听觉场景分析结果
      # output['F_accuracy']   → 汇入扣带回
    """

    def __init__(self):
        self.a1 = PrimaryAuditoryCortex(n_channels=N_A1_CHANNELS)
        self.belt = BeltArea(n_channels=N_BELT_CHANNELS)
        self.parabelt = ParabeltArea(n_what=N_PARABELT_WHAT,
                                     n_where=N_PARABELT_WHERE)

    def process(self, mgb_output: dict,
                soc_spatial: Optional[np.ndarray] = None,
                ic_output: Optional[dict] = None,
                arousal: float = 0.8) -> dict:
        """听皮层全流程处理.

        Args:
            mgb_output: MGB输出 dict
            soc_spatial: SOC空间信息 (可选)
            ic_output: IC输出 (用于FM+多感官, 可选)
            arousal: 唤醒度

        Returns:
            dict with:
              'what_stream':  声音识别特征 → Wernicke
              'where_stream': 空间定位特征 → 顶叶/FPN
              'a1_output':    A1层输出
              'belt_output':  Belt层输出
              'scene':        听觉场景分析
              'F_accuracy':   预测误差汇入扣带回
              'PE_total':     总预测误差
        """
        # ---- A1: 频率调谐 ----
        a1_out = self.a1.process(mgb_output['tonotopic'], arousal=arousal)

        # ---- Belt: 复杂特征 (FM + 谐波 + 音色) ----
        ic_fm = ic_output.get('fm_sweep', None) if ic_output else None
        belt_out = self.belt.process(a1_out, mgb_output['complex'],
                                     ic_fm_sweep=ic_fm)

        # ---- Parabelt: 听觉场景分析 + 双流分离 ----
        ic_ms = ic_output.get('multisensory', None) if ic_output else None
        scene_out = self.parabelt.process(belt_out, a1_out,
                                          soc_spatial=soc_spatial,
                                          ic_multisensory=ic_ms)

        # ---- 预测编码: 自上而下 ----
        # Parabelt → Belt 预测
        para_pred = self.parabelt.predict_to_Belt()
        self.belt.receive_feedback(para_pred, lr=0.08)

        # Belt → A1 预测
        belt_pred = self.belt.predict_to_A1()
        self.a1.receive_feedback(belt_pred, lr=0.06)

        # ---- 预测误差: 各层PE汇总 ----
        pe_a1 = self.a1.compute_prediction_error()
        pe_belt = self.belt.compute_prediction_error()
        pe_para = self.parabelt.compute_prediction_error()

        # F_accuracy: 听皮层各层预测误差加权和
        F_accuracy = (
            float(np.mean(np.abs(pe_a1))) * 0.3 +
            float(np.mean(np.abs(pe_belt))) * 0.5 +
            float(np.mean(np.abs(pe_para))) * 0.7
        )
        PE_total = float(np.mean(np.abs(pe_a1)) + np.mean(np.abs(pe_belt)) +
                         np.mean(np.abs(pe_para)))

        # ---- 构建听皮层综合输出 ----
        return {
            'what_stream': scene_out['what_stream'],
            'where_stream': scene_out['where_stream'],
            'a1_output': a1_out,
            'belt_output': belt_out,
            'scene': scene_out,
            'F_accuracy': F_accuracy,
            'PE_total': PE_total,
            'pe_a1': pe_a1,
            'pe_belt': pe_belt,
            'pe_para': pe_para,
        }
