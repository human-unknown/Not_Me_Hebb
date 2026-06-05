"""
layer1_free_energy.py —— L1 自由能计算 + 注意力调制
自由能原理智能体 — M1 单智能体生存

F = D_KL[q(z) || p(z)] - E_q[ln p(s|z)]

分解为三个域:
- F_body:      生理稳态偏离 → 驱动生存行为
- F_social:    社会预测误差 → M1 中为 0
- F_cognitive: 模型复杂度 → 驱动学习与探索
"""

import numpy as np
from cns.data_types import D, H, Theta, FreeEnergy
from cerebrum.limbic_system.hippocampus import predict_sensations, ClusterNetwork
from cns.utils import exp_moving_average


# ============================================================
# 自由能分量
# ============================================================

def compute_F_body(pred_err: np.ndarray, theta: Theta,
                   body=None) -> float:
    """躯体域自由能：身体稳态偏离 (v2 FEP-native)

    如果 body 可用: F_body = w_body * Σ((b - setpoint)/sigma_z)²
    否则回退到:   F_body = 0.5 * ||s - s_pred||² / sigma_x² (v1 兼容)
    """
    if body is not None:
        dev = body.compute_deviation()
        return float(theta.w_body * dev / (theta.sigma_x ** 2 + 1e-8))
    return float(0.5 * np.sum(pred_err ** 2) / (theta.sigma_x ** 2 + 1e-8))


# ---- 社会上下文 (Stage 6 人类对话) ----
class SocialContext:
    """追踪人类对话中的社会情感状态。

    不是"情绪识别器"——是 Agent 对互动质量的预测模型。
    Agent 用这个来预测"人类下一句话的情感大概是什么样"，
    预测误差就是 F_social。
    """

    def __init__(self, tau: float = 20.0):
        self.tau = tau                    # EMA 时间常数
        self.expected_valence: float = 0.0  # 预测的人类情感效价 [-1, 1]
        self.expected_arousal: float = 0.2  # 预测的人类唤醒度 [0, 1]
        self.trust_level: float = 0.5       # 对人类的信任 [0, 1]
        self.n_interactions: int = 0        # 互动次数
        self.last_valence: float = 0.0
        self.steps_since_input: int = 0     # 距离上次人类输入的步数

    def update(self, valence: float, arousal: float):
        """用最新人类输入的情感更新社会预测"""
        alpha = 1.0 / self.tau
        self.expected_valence += alpha * (valence - self.expected_valence)
        self.expected_arousal += alpha * (arousal - self.expected_arousal)
        self.n_interactions += 1
        self.last_valence = valence
        self.steps_since_input = 0

        # 信任更新: 基于社会预测精度 (FEP 驱动)
        # 预测准确 → 信任上升; 预测误差大 → 信任下降
        # 这模拟了: 信任 = 对社会预测模型的置信度
        predicted = self.expected_valence
        prediction_error = abs(predicted - valence)
        # 低误差 (<0.3) → 正信任增量, 高误差 → 负信任增量
        trust_delta = 0.1 * (1.0 - 2.0 * prediction_error)
        self.trust_level = float(np.clip(self.trust_level + trust_delta, 0.0, 1.0))

    def tick(self):
        """每步调用——无人类输入时社会信号衰减"""
        self.steps_since_input += 1
        # 长期无互动 → expected_valence 缓慢回归中性
        if self.steps_since_input > 50:
            decay = 0.995
            self.expected_valence *= decay
            self.expected_arousal = max(0.05, self.expected_arousal * decay)


def compute_F_social(s: np.ndarray, beliefs, theta: Theta,
                     social_ctx: SocialContext = None) -> float:
    """社会域自由能：预测误差 (M3 多智能体 + Stage 6 人类对话)

    Stage 6 人类对话:
      s[80:88] 携带情感信号 (见 sentiment.sentiment_to_social_signal)
      s[80] = valence [0,1], s[86] = raw_valence [-1,1]
      s[81] = arousal, s[85] = human_active flag

      F_social = w_social × (predicted_valence - observed_valence)² / σ²
      负效价偏离预测 → F_social 高 → 身体感到不好
      正效价符合/超出预测 → F_social 低 → 身体感到好

    M3 多智能体:
      s[80:88] = 其他 agent 位置信号
    """
    # ---- Stage 6: 人类社会信号 ----
    if social_ctx is not None:
        signal = s[80:88]
        human_active = signal[5] > 0.5  # s[85] = is_human_active
        if human_active:
            raw_valence = float(signal[6])    # [-1, 1]
            arousal = float(signal[1])         # [0, 1]
            predicted = social_ctx.expected_valence
            # 预测误差
            error = predicted - raw_valence
            # 消极偏差: 负面信号惩罚更重 (theta.negativity_bias, 可学习)
            if raw_valence < 0:
                weight = float(theta.negativity_bias)
            else:
                weight = 1.0
            # 唤醒放大
            arousal_factor = 1.0 + arousal
            F = theta.w_social * weight * arousal_factor * (error ** 2) / (
                theta.sigma_x ** 2 + 1e-8)
            # v3: 信任调节 — 信任越高, F_social 越低 (预期好的互动)
            # trust_modulation = theta.w_F_signal 驱动 (信任 = 对社会预测模型的置信度)
            trust_mod = theta.w_F_signal * 3.0  # normalize: 0.1→0.3, 1.0→3.0
            F *= max(0.1, 1.0 - trust_mod * social_ctx.trust_level)
            return float(F)

    # ---- M3: 多智能体信号 ----
    if beliefs is None or not beliefs.other_positions:
        signal = s[80:88]
        if np.sum(np.abs(signal)) > 0.01:
            return 0.5 * np.sum(signal ** 2) / (theta.sigma_x ** 2 + 1e-8)
        return 0.0

    total = 0.0
    obs_idx = 0
    for aid in sorted(beliefs.other_positions.keys()):
        base = 80 + obs_idx * 8
        if base + 1 >= len(s):
            break
        sig_len = len(beliefs.other_positions.get(aid, [0.0, 0.0]))
        observed = s[base:base + sig_len]
        predicted = np.array(beliefs.other_positions[aid])
        total += 0.5 * np.sum((observed - predicted) ** 2) / (
            theta.sigma_x ** 2 + 1e-8)
        obs_idx += 1

    return float(total)


def compute_F_cognitive(net: ClusterNetwork, theta: Theta,
                       s: np.ndarray = None) -> float:
    """认知域自由能：模型复杂度代价 + 信息缺口 (M2)

    F_cognitive ∝ log(簇数 + 1)
    簇越多 → 模型越复杂 → 认知代价越高
    驱动系统在"精确"和"简洁"之间平衡

    M2 增强: 新颖状态降低 F_cognitive（奖励探索）
    """
    complexity = float(theta.w_cognitive * np.log(len(net.clusters) + 1) * 0.1)

    # M2: 信息缺口 — 新颖状态降低认知代价
    if s is not None and net.n_clusters > 0:
        best_sim = net.best_similarity(s)
        if best_sim < 0.5:
            # 低相似度 → 新颖状态 → 微小负贡献
            complexity = max(0.0, complexity - theta.exploration_bonus * 0.01)

    return complexity


# ============================================================
# Valence & Arousal
# ============================================================

def compute_valence_arousal(F_body: float, theta: Theta,
                           F_total: float = None,
                           F_baseline: float = None) -> tuple[float, float]:
    """从自由能计算 Valence（效价）和 Arousal（唤醒）

    v2: Valence 基于 F_total 相对 baseline 的变化
      - F_total < baseline → 自由能在下降 → 正效价（在变好）
      - F_total > baseline → 自由能在上升 → 负效价（在变差）
      - 回退: 无 baseline 时用 F_body 的绝对值

    arousal = tanh(eta_arousal * |F_total|)
      - F 偏离越大 → 唤醒越高
    """
    if F_total is not None and F_baseline is not None:
        # 相对变化: 正 = 在变好, 负 = 在变差
        delta_F = F_baseline - F_total  # F 下降 = 正
        valence = np.tanh(theta.eta_valence * delta_F)
    else:
        # 回退: 绝对值
        valence = np.tanh(-theta.eta_valence * F_body)

    # Arousal: 总是基于 F_total 的规模
    F_for_arousal = F_total if F_total is not None else F_body
    arousal = np.tanh(theta.eta_arousal * abs(F_for_arousal))
    return float(valence), float(arousal)


# ============================================================
# 习惯化追踪器
# ============================================================

class HabituationTracker:
    """追踪感觉通道的累积暴露，实现习惯化

    习惯化：同一刺激反复出现且无后果 → 响应递减
    去习惯化：单次意外（novelty）→ 重置累积暴露
    """

    def __init__(self, tau: float = 10.0):
        self.tau = tau            # 时间常数
        self.running_F: float = 0.0  # EMA 平滑后的 F
        self.n: int = 0           # 更新次数

    def update(self, F: float):
        """更新习惯化状态
        使用 EMA，tau 越大惯性越大（habituation 越慢）
        """
        self.n += 1
        alpha = 1.0 / self.tau
        self.running_F = exp_moving_average(F, self.running_F, alpha)

    def reset(self):
        """去习惯化：重置累积暴露"""
        self.running_F = 0.0
        self.n = 0


# ============================================================
# 动态注意力精度
# ============================================================

def compute_attention_precision(
    valence: float,
    arousal: float,
    theta: Theta,
    hab: HabituationTracker,
) -> float:
    """四因素调制注意力精度

    precision = f(novelty, goal_relevance, temporal_coherence, habituation)

    1. novelty:    效价绝对值低时（意外），新颖性高
    2. arousal:    唤醒度高 → 注意力集中
    3. habituation: 习惯化 → 注意力节省
    4. theta:      w_body 主导时注意力集中于生理域

    Returns:
        注意力精度 [0, 1]
    """
    # 新颖性：效价接近 0（预测误差小但不完全可预测）→ novelty 高
    # v3: eta_valence 驱动 novelty 敏感度 (默认 0.5 → 与原来等价)
    # eta_valence 高 → 效价变化对 novelty 影响更大 → 更快失去兴趣
    novelty = max(0.0, 1.0 - abs(valence) * theta.eta_valence)

    # 习惯化抑制：累积 F 越大 → hab_factor 越小
    hab_factor = float(np.exp(-hab.running_F * 0.1))

    # 目标相关性：w_body 在其总权重中的占比
    total_w = theta.w_body + theta.w_social + theta.w_cognitive + 1e-8
    goal_relevance = theta.w_body / total_w

    # 综合精度
    precision = arousal * novelty * hab_factor * goal_relevance

    return float(np.clip(precision, 0.0, 1.0))


# ============================================================
# 总入口：compute_free_energy
# ============================================================

def compute_free_energy(
    z: np.ndarray,
    s: np.ndarray,
    net: ClusterNetwork,
    theta: Theta,
    hab: HabituationTracker,
    beliefs = None,  # AgentBelief (M3+)
    body = None,     # BodyVector (v2)
    social_ctx = None,  # SocialContext (Stage 6 人类对话)
    visual_pred_error: float = 0.0,  # Module D: 视觉预测编码
) -> FreeEnergy:
    """计算当前时刻的总自由能及其分解

    流程:
    1. z → s_pred（L0 预测）
    2. s - s_pred → 预测误差
    3. 分别计算 F_body, F_social, F_cognitive
    4. 加权合成 F_total
    5. 从 F_body 计算 valence, arousal
    6. 计算注意力精度

    Args:
        z: 当前隐状态 (H,)
        s: 当前感觉输入 (D,)
        net: 簇记忆网络
        theta: 参数配置
        hab: 习惯化追踪器
        beliefs: AgentBelief (M3+, 可选)

    Returns:
        FreeEnergy 完整分解
    """
    # ---- Phase 1+3: s_pred from cluster (no z dependency) ----
    c = net.recall(s) if net.n_clusters > 0 else None
    if c is not None:
        s_pred = c.centroid
    else:
        s_pred = np.zeros_like(s)
        c = None

    # F_body: 身体稳态偏离 — 纯数值，来自 BodyVector.compute_deviation()
    # 注意: 这是 F_body (身体域), 不是 F_cognitive (认知复杂度, 见 compute_F_cognitive)
    if body is not None:
        dev = body.compute_deviation()
        F_body_deviation = theta.w_body * dev / (theta.sigma_x ** 2 + 1e-8)
    else:
        # 回退: 无 body 时用 s-s_pred 近似
        F_body_deviation = 0.5 * np.sum((s - s_pred) ** 2) / (theta.sigma_x ** 2 + 1e-8)

    # F_accuracy: 集群预测误差 — 从集群读出 (v3: Theta 参数化)
    if c is not None:
        from cns.data_types import S_CORE
        residual = np.sum((s[:S_CORE] - s_pred[:S_CORE]) ** 2)
        F_accuracy = (theta.w_accuracy * residual / (theta.sigma_x ** 2 + 1e-8)
                      + theta.w_F_signal * c.F_signal)
    else:
        # 冷启动: 基于输入熵的自适应 F_accuracy
        from cns.data_types import S_CORE
        input_norm = float(np.sum(s[:S_CORE] ** 2))
        F_accuracy = theta.w_accuracy * input_norm / (theta.sigma_x ** 2 + 1e-8)

    # F_social / F_cognitive 已通过 Theta 参数化
    F_social = compute_F_social(s, beliefs, theta, social_ctx)
    F_cognitive = compute_F_cognitive(net, theta, s)

    # 合成: F = F_body + F_social + F_cognitive + F_accuracy + F_visual_pred
    # Module D: 视觉预测编码 — 层级预测误差惩罚
    # 高层级预测误差 → 视觉世界不稳定 → F 增加 → 驱动注意/学习
    F_total = (F_body_deviation + F_social + F_cognitive + F_accuracy
               + theta.w_accuracy * 0.5 * visual_pred_error)

    # Valence & Arousal — 基于 F_total 相对 habituation baseline 的变化
    # F_total < running_F → 自由能下降 → 正效价（在变好）
    # F_total > running_F → 自由能上升 → 负效价（在变差）
    F_baseline = hab.running_F if hab.n > 0 else F_total
    valence, arousal = compute_valence_arousal(
        F_total, theta, F_total=F_total, F_baseline=F_baseline)

    # 注意力精度
    attention_precision = compute_attention_precision(
        valence, arousal, theta, hab
    )

    return FreeEnergy(
        total=F_total,
        body=F_body_deviation,
        social=F_social,
        cognitive=F_cognitive,
        accuracy=F_accuracy,
        valence=valence,
        arousal=arousal,
        attention_precision=attention_precision,
    )


def decompose_F(F: FreeEnergy) -> dict:
    """将 FreeEnergy 分解为字典，便于判断哪个域主导"""
    return {
        "body": F.body,
        "social": F.social,
        "cognitive": F.cognitive,
        "accuracy": F.accuracy,
    }
