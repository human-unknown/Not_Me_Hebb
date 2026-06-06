"""
data_types.py —— 全部 struct/dataclass，20 个 Theta 参数
自由能原理智能体 — M1 单智能体生存
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ============================================================
# 全局常量
# ============================================================
D = 372         # v5.1: 感知维度 = D_V5 (text[64] + visual[308])
S_CORE = 362    # v5.1: 感觉核心 = D - 10 (F_context[5] + action_onehot[5])
H = 16          # 隐藏状态维度
K = 256         # 最大簇数 (multimodal 330-dim)
A = 5           # 行动维度 (N,S,W,E,REST)
N_AGENTS = 3    # 最大智能体数（M3+）


# ============================================================
# v5.0 视觉通路布局 (按通路 × 层级组织)
# ============================================================
# 文本段
TEXT_V5_WIDTH = 64
TEXT_V5_START, TEXT_V5_END = 0, 64

# M 通路 (运动/空间): M-V1, M-V2, MT, MST
M_V1_WIDTH = 32
M_V2_WIDTH = 16
MT_WIDTH = 32
MST_WIDTH = 16
M_PATHWAY_WIDTH = M_V1_WIDTH + M_V2_WIDTH + MT_WIDTH + MST_WIDTH  # 96

M_V1_START, M_V1_END = TEXT_V5_END, TEXT_V5_END + M_V1_WIDTH
M_V2_START, M_V2_END = M_V1_END, M_V1_END + M_V2_WIDTH
MT_START, MT_END = M_V2_END, M_V2_END + MT_WIDTH
MST_START, MST_END = MT_END, MT_END + MST_WIDTH

# P 通路 (形状/细节): P-V1, P-V2, V4-shape
P_V1_WIDTH = 48
P_V2_WIDTH = 32
V4_SHAPE_WIDTH = 32
P_PATHWAY_WIDTH = P_V1_WIDTH + P_V2_WIDTH + V4_SHAPE_WIDTH  # 112

P_V1_START, P_V1_END = MST_END, MST_END + P_V1_WIDTH
P_V2_START, P_V2_END = P_V1_END, P_V1_END + P_V2_WIDTH
V4_SHAPE_START, V4_SHAPE_END = P_V2_END, P_V2_END + V4_SHAPE_WIDTH

# K 通路 (颜色): K-V1, K-V2, V4-color
K_V1_WIDTH = 16
K_V2_WIDTH = 16
V4_COLOR_WIDTH = 16
K_PATHWAY_WIDTH = K_V1_WIDTH + K_V2_WIDTH + V4_COLOR_WIDTH  # 48

K_V1_START, K_V1_END = V4_SHAPE_END, V4_SHAPE_END + K_V1_WIDTH
K_V2_START, K_V2_END = K_V1_END, K_V1_END + K_V2_WIDTH
V4_COLOR_START, V4_COLOR_END = K_V2_END, K_V2_END + V4_COLOR_WIDTH

# IT 物体
IT_WIDTH = 16
IT_START, IT_END = V4_COLOR_END, V4_COLOR_END + IT_WIDTH

# 快速通路 (SC + Pulvinar)
SC_WIDTH = 16
PULVINAR_WIDTH = 12
SC_START, SC_END = IT_END, IT_END + SC_WIDTH
PULVINAR_START, PULVINAR_END = SC_END, SC_END + PULVINAR_WIDTH

# 绑定信号
BINDING_WIDTH = 8
BINDING_START, BINDING_END = PULVINAR_END, PULVINAR_END + BINDING_WIDTH

# 总视觉维度 (v5.0)
D_VISUAL_V5 = (M_PATHWAY_WIDTH + P_PATHWAY_WIDTH + K_PATHWAY_WIDTH +
               IT_WIDTH + SC_WIDTH + PULVINAR_WIDTH + BINDING_WIDTH)

# 总感知维度 (v5.0)
D_V5 = TEXT_V5_WIDTH + D_VISUAL_V5  # 64 + 308 = 372

# ============================================================
# v5.2 听觉通路布局
# ============================================================

# 耳蜗核输出: tonotopic spectrum (mel-spaced channels)
CN_WIDTH = 32
CN_START, CN_END = D_V5, D_V5 + CN_WIDTH              # s[372:404]

# 上橄榄复合体输出: binaural spatial (ITD + ILD)
SOC_WIDTH = 24
SOC_START, SOC_END = CN_END, CN_END + SOC_WIDTH        # s[404:428]

# 下丘输出: integrated frequency×space×time features
IC_WIDTH = 24
IC_START, IC_END = SOC_END, SOC_END + IC_WIDTH         # s[428:452]

# 听皮层输出: auditory objects/scene features
AC_WIDTH = 16
AC_START, AC_END = IC_END, IC_END + AC_WIDTH           # s[452:468]

# 总听觉维度
D_AUDIO = CN_WIDTH + SOC_WIDTH + IC_WIDTH + AC_WIDTH   # 96
D_V52 = D_V5 + D_AUDIO                                 # 372 + 96 = 468

# ============================================================
# v5.4 痛觉通路布局
# ============================================================

# 脊髓背角输出: 闸门控制后的痛觉信号
PAIN_DH_WIDTH = 16
PAIN_DH_START, PAIN_DH_END = D_V52, D_V52 + PAIN_DH_WIDTH     # s[468:484]

# 外侧脊髓丘脑束: 感觉-辨别 (快痛Aδ → VPL → S1/S2)
PAIN_LATERAL_WIDTH = 12
PAIN_LATERAL_START, PAIN_LATERAL_END = PAIN_DH_END, PAIN_DH_END + PAIN_LATERAL_WIDTH  # s[484:496]

# 内侧脊髓丘脑束: 情感-动机 (慢痛C → CM-Pf/MD → ACC/岛叶)
PAIN_MEDIAL_WIDTH = 12
PAIN_MEDIAL_START, PAIN_MEDIAL_END = PAIN_LATERAL_END, PAIN_LATERAL_END + PAIN_MEDIAL_WIDTH  # s[496:508]

# 丘脑痛觉中继: VPL + CM-Pf + MD + Po 整合
PAIN_THALAMIC_WIDTH = 8
PAIN_THALAMIC_START, PAIN_THALAMIC_END = PAIN_MEDIAL_END, PAIN_MEDIAL_END + PAIN_THALAMIC_WIDTH  # s[508:516]

# 总痛觉维度
D_PAIN = PAIN_DH_WIDTH + PAIN_LATERAL_WIDTH + PAIN_MEDIAL_WIDTH + PAIN_THALAMIC_WIDTH  # 48
D_V54 = D_V52 + D_PAIN                                           # 468 + 48 = 516

# 更新全局感知维度 (v5.4)
D = D_V54
S_CORE = D - 10                                        # 感觉核心


# ============================================================
# 核心数据结构
# ============================================================

@dataclass
class Theta:
    """26 个可学习参数 —— 唯一参数集 (v5.6: +2 语言PE权重)
    L0 (6): 生成模型
    L1 (11): 自由能权重 + 情感偏差 + 精度系数 + 语言PE权重
    L2 (5): 策略推理
    L3 (4): 元学习
    """
    # L0 (6): 生成模型
    sigma_z: float = 0.1             # 状态噪声
    sigma_x: float = 1.0             # 感知噪声
    decay_rate: float = 0.01         # 簇衰减率
    cluster_threshold: float = 0.70  # 簇匹配阈值
    learn_rate_l0: float = 0.05      # 簇学习率
    pe_lr_scale: float = 0.0         # 预测误差驱动学习率缩放 (0=off)

    # L1 (11): 自由能权重 + 语言PE (v5.6: +2)
    w_body: float = 1.0              # 躯体域权重
    w_social: float = 1.0            # 社会域权重
    w_cognitive: float = 1.0         # 认知域权重
    eta_valence: float = 0.5         # 效价敏感度
    eta_arousal: float = 0.5         # 唤醒敏感度
    habituation_tau: float = 10.0    # 习惯化时间常数
    negativity_bias: float = 1.5     # 负面信号权重 (v2: 可学习, 非硬编码)
    w_accuracy: float = 0.5          # v3: 预测残差在 F_accuracy 中的权重
    w_F_signal: float = 0.1          # v3: 集群历史 F_signal 在 F_accuracy 中的权重
    w_semantic: float = 0.5          # v5.6: 语义PE (N400) 在 F_language 中的权重
    w_syntactic: float = 0.3         # v5.6: 句法PE (P600) 在 F_language 中的权重

    # L2 (5): 策略推理
    gamma: float = 0.95              # 时间折扣
    exploration_bonus: float = 0.1   # 探索奖励
    temperature: float = 1.0         # softmax 温度
    n_policy_samples: int = 16       # 策略采样数
    urgency_weight: float = 0.3      # 紧急度权重

    # L3 (4): 元学习
    meta_lr: float = 0.01            # 元学习率
    grad_epsilon: float = 0.001      # 有限差分 epsilon
    plasticity_decay: float = 0.999  # 可塑性衰减
    critical_window: int = 1000      # 关键期步数

    # L4 (6): v6.0 记忆系统 — 语义 + 程序性
    semantic_threshold: float = 0.45     # 语义簇匹配阈值
    semantic_learn_rate: float = 0.01    # 语义学习率
    semantic_decay_rate: float = 0.003   # 语义衰减率
    habit_threshold: float = 0.3         # 习惯形成的最低重复比例
    habit_automation_rate: float = 0.05  # 习惯自动化速度
    d1_d2_balance: float = 0.5           # D1(Go)/D2(No-Go) 平衡

    def to_dict(self) -> dict:
        """转为字典，便于参数扫描"""
        return {
            'sigma_z': self.sigma_z, 'sigma_x': self.sigma_x,
            'decay_rate': self.decay_rate, 'cluster_threshold': self.cluster_threshold,
            'learn_rate_l0': self.learn_rate_l0,
            'pe_lr_scale': self.pe_lr_scale,
            'w_body': self.w_body, 'w_social': self.w_social, 'w_cognitive': self.w_cognitive,
            'eta_valence': self.eta_valence, 'eta_arousal': self.eta_arousal,
            'habituation_tau': self.habituation_tau,
            'negativity_bias': self.negativity_bias,
            'w_accuracy': self.w_accuracy, 'w_F_signal': self.w_F_signal,
            'w_semantic': self.w_semantic, 'w_syntactic': self.w_syntactic,
            'gamma': self.gamma, 'exploration_bonus': self.exploration_bonus,
            'temperature': self.temperature, 'n_policy_samples': self.n_policy_samples,
            'urgency_weight': self.urgency_weight,
            'meta_lr': self.meta_lr, 'grad_epsilon': self.grad_epsilon,
            'plasticity_decay': self.plasticity_decay, 'critical_window': self.critical_window,
            'semantic_threshold': self.semantic_threshold,
            'semantic_learn_rate': self.semantic_learn_rate,
            'semantic_decay_rate': self.semantic_decay_rate,
            'habit_threshold': self.habit_threshold,
            'habit_automation_rate': self.habit_automation_rate,
            'd1_d2_balance': self.d1_d2_balance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Theta':
        """从字典构造，用于参数扫描"""
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class SensoryVector:
    """感觉输入向量"""
    data: np.ndarray          # shape (D,)
    timestamp: int = 0

    def __post_init__(self):
        if self.data.shape != (D,):
            raise ValueError(f"SensoryVector data must have shape ({D},), got {self.data.shape}")


@dataclass
class Cluster:
    """Hebb 细胞集群 —— 记忆的基本单元 (v2: 含行动经验)"""
    centroid: np.ndarray      # shape (D,)，簇中心（原型模式）
    count: int = 1            # 被激活次数
    age: int = 0              # 年龄（时间步计数）
    activation: float = 0.0   # 近期激活频率 [0, 1]
    G_ema: float = 0.0        # 该模式的期望 G 值 EMA (v2: GHistory → Cluster)
    F_signal: float = 0.0     # 历史上 F 的 EMA (v2: 用于 F_accuracy)

    def __post_init__(self):
        if self.centroid.shape != (D,):
            raise ValueError(f"Cluster centroid must have shape ({D},), got {self.centroid.shape}")


@dataclass
class FreeEnergy:
    """自由能分解 —— Layer 1 输出

    F = F_body + F_social + F_cognitive + F_accuracy
    """
    total: float
    body: float                 # 身体稳态偏离
    social: float               # 社会预测误差
    cognitive: float            # 模型复杂度
    accuracy: float             # 集群预测精度 (v2: 新增字段)
    valence: float              # [-1, 1]  效价
    arousal: float              # [0, 1]   唤醒
    attention_precision: float  # [0, 1]   注意力精度


@dataclass
class Action:
    """行动选择结果 —— Layer 2 输出"""
    index: int                 # 0..3 (N,S,W,E)
    expected_F: float          # 期望自由能
    expected_G: float          # 期望 EFE
    confidence: float          # MoE 置信度


@dataclass
class AgentBelief:
    """社会信念 —— 多智能体建模（M3+）"""
    other_positions: dict = field(default_factory=dict)   # agent_id -> np.ndarray(2,)
    trust_levels: dict = field(default_factory=dict)       # agent_id -> float [0,1]
    second_order: dict = field(default_factory=dict)       # agent_i -> {agent_j -> belief}


@dataclass
class SeedPackage:
    """种子包 —— 冷启动所需的最小先验"""
    theta: Theta
    hidden: np.ndarray        # (H,) 初始隐状态
    clusters: list            # list[Cluster]
    rng_state: tuple
    step_count: int = 0


# ============================================================
# BodyVector — 身体稳态 (v2: FEP-native homeostasis)
# ============================================================

@dataclass
class BodyVector:
    """M 维身体向量 — 自由能原理的内稳态基板

    每步按 ODE 漂移，行动触发恢复。F_body = 偏离设定点的代价。
    无语义标签：b_i 只是标量，含义从与环境的互动中涌现。
    """
    M: int = 5
    mode: str = 'grid'
    b: np.ndarray = None
    setpoints: np.ndarray = None
    decays: np.ndarray = None

    def __post_init__(self):
        if self.mode == 'text':
            self.M = 9  # v5.4: +1 痛觉维度
            if self.b is None:
                self.b = np.array([0.7, 0.7, 0.0, 0.0, 0.3, 0.3, 0.3, 0.5, 0.0], dtype=float)
            elif len(self.b) < 9:
                self.b = np.concatenate([self.b, np.zeros(9 - len(self.b))])
            if self.setpoints is None:
                self.setpoints = np.array([0.7, 0.7, 0.0, 0.0, 0.3, 0.3, 0.3, 0.5, 0.0], dtype=float)
            elif len(self.setpoints) < 9:
                self.setpoints = np.concatenate([self.setpoints, np.array([0.0])])
            if self.decays is None:
                # v5.7: 加强漂移速率 → 更多自由能动态
                self.decays = np.array([-0.006, 0.0, 0.004, 0.0, 0.0,
                                        -0.005, -0.005, 0.002, -0.001], dtype=float)
                # b[8] 痛觉/组织完整性: 缓慢自愈 (正漂移=恢复至1.0)
                # 实际痛觉信号由 nociception_hierarchy 基于 b[8] 偏离计算
        else:
            if self.b is None:
                self.b = np.array([0.7, 0.7, 0.0, 0.0, 0.3], dtype=float)
            if self.setpoints is None:
                self.setpoints = np.array([0.7, 0.7, 0.0, 0.0, 0.3], dtype=float)
            if self.decays is None:
                self.decays = np.array([-0.003, -0.002, 0.004, 0.0, 0.001], dtype=float)

    def step(self, action_type: int = -1, env_field: float = 0.0):
        """单步 ODE 更新

        Args:
            action_type: 执行的动作类型 (-1 = 无行动)
            env_field: 当前位置的环境场值 (用于 b₃ 漂移)
        """
        # 防御: 确保 decays/setpoints 与 b 维度一致 (修复持久化恢复时的不一致)
        if self.decays is None or len(self.decays) != len(self.b):
            if len(self.b) == 9:
                self.decays = np.array([-0.003, 0.0, 0.002, 0.0, 0.0,
                                        -0.003, -0.003, 0.001, -0.001], dtype=float)
            else:
                self.decays = np.array([-0.003, -0.002, 0.004, 0.0, 0.001], dtype=float)
        if self.setpoints is None or len(self.setpoints) != len(self.b):
            if len(self.b) == 9:
                self.setpoints = np.array([0.7, 0.7, 0.0, 0.0, 0.3, 0.3, 0.3, 0.5, 0.0], dtype=float)
            else:
                self.setpoints = np.array([0.7, 0.7, 0.0, 0.0, 0.3], dtype=float)
        self.M = len(self.b)

        # 基础漂移
        self.b = self.b + self.decays
        self.b[3] += 0.01 * (env_field - self.b[3])

        if self.mode == 'text':
            if action_type == 3:
                self.b[0] = min(1.0, self.b[0] + 0.03)
            elif self.b[0] < 0.3:
                self.b[0] += 0.005  # 临界区强制自愈: 低于 0.3 时无论如何回升
            if action_type == 4:
                self.b[2] = max(0.0, self.b[2] - 0.03)
            elif action_type >= 0:
                self.b[2] = min(1.0, self.b[2] + 0.005)
            # v5.4: b[8] 组织完整性 — 自愈趋势 (向setpoint回归)
            if len(self.b) >= 9:
                self.b[8] += 0.001 * (self.setpoints[8] - self.b[8])  # 缓慢自愈
        else:
            if action_type == 0 or action_type == 2:
                if env_field > 0.1:
                    self.b[0] += 0.3
                    self.b[1] += 0.4
            if action_type == 4:
                self.b[2] = max(0.0, self.b[2] - 0.02)
                self.b[4] = max(0.0, self.b[4] - 0.02)
            elif action_type >= 0:
                self.b[2] = min(1.0, self.b[2] + 0.01)
                self.b[4] = min(1.0, self.b[4] + 0.005)

        # 裁剪到 [0, 1]
        np.clip(self.b, 0.0, 1.0, out=self.b)

    def to_sensory(self) -> np.ndarray:
        """将身体状态编码为感知向量的 body 通道"""
        return self.b.copy()

    def compute_deviation(self) -> float:
        """计算偏离设定点的加权平方和 → F_body 的基础"""
        dev = (self.b - self.setpoints)
        return float(np.sum(dev ** 2))


# ============================================================
# 参数验证
# ============================================================

def validate_theta(theta: Theta) -> bool:
    """验证 Theta 参数数量 = 32 (v6.0)"""
    n = len(theta.to_dict())
    if n != 32:
        raise ValueError(f"Theta 必须有 32 个参数，当前有 {n}")
    return True


# 行动映射
ACTION_NAMES = {0: 'N', 1: 'S', 2: 'W', 3: 'E', 4: 'R'}
ACTION_DIRECTIONS = np.array([[-1,0],[1,0],[0,-1],[0,1],[0,0]], dtype=float)
