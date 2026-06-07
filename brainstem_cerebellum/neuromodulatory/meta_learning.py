"""
layer3_meta.py —— L3 元参数管理 (M5 激活)
自由能原理智能体

功能:
- create_default_theta(): 参数默认配置
- MetaLearner: 在线元学习（M5 真实有限差分梯度下降）

v6.1 发育机制:
- GluN2B→GluN2A 连续轨迹 (指数衰减, 半衰期 ~5000 步)
- 4 阶段发育年龄系统 (婴儿→儿童→青少年→成人)
- 发育因子调制 (learn_rate, threshold, PNN rate, silent synapse, pruning)
- 可塑性衰减: plasticity_decay 逐步降低更新幅度
- 创伤模拟: apply_trauma() 永久修改社会参数
"""

import numpy as np
from cns.data_types import Theta, FreeEnergy, DevelopmentalStage


# ============================================================
# 默认参数创建
# ============================================================

def create_default_theta() -> Theta:
    """返回 56 个参数的默认配置 (v6.3)"""
    return Theta(
        sigma_z=0.1, sigma_x=1.0, decay_rate=0.01,
        cluster_threshold=0.85, learn_rate_l0=0.05,
        w_body=1.0, w_social=1.0, w_cognitive=1.0,
        eta_valence=0.5, eta_arousal=0.5, habituation_tau=10.0,
        negativity_bias=1.5, w_accuracy=0.5, w_F_signal=0.1,
        gamma=0.95, exploration_bonus=0.1, temperature=1.0,
        n_policy_samples=16, urgency_weight=0.3,
        meta_lr=0.01, grad_epsilon=0.001,
        plasticity_decay=0.999, critical_window=1000,
        # v6.1: 发育优化新参数
        stdp_lr=0.02, stdp_window=3, stdp_weight=0.3,
        glun2b_ratio=0.9, pnn_formation_rate=0.001,
        developmental_stage=1, protection_decay=0.995,
        candidate_max=64,
        # v6.2: 记忆巩固优化
        tag_window=30, tag_decay_rate=0.05,
        tag_capture_strength=0.3,
        persistence_decay_rate=0.1,
        persistence_threshold_boost=0.2,
        persistence_lr_boost=0.5,
        consolidation_lock_factor=0.5,
        consolidation_lock_max=10,
        # v6.3: 睡眠与时间维度
        circa_tau=24.0, circa_light_sensitivity=0.3,
        sleep_pressure_threshold=0.65,
        nrem_duration_ratio=0.65,
        synaptic_downscale_rate=0.03,
        alpha_gating_strength=0.4,
        glymphatic_clear_rate=0.005,
        rem_emotional_processing=0.3,
    )


# ============================================================
# 种子包创建
# ============================================================

def create_seed_package(theta=None, rng=None, bootstrap_sensory=None):
    from cns.data_types import SeedPackage
    from cerebrum.limbic_system.hippocampus import bootstrap_clusters
    if rng is None: rng = np.random.default_rng()
    if theta is None: theta = create_default_theta()
    hidden = rng.normal(0, 0.1, 16)  # H=16 初始隐状态
    if bootstrap_sensory:
        net = bootstrap_clusters(bootstrap_sensory, theta)
        clusters = net.clusters
    else:
        clusters = []
    return SeedPackage(theta=theta, hidden=hidden, clusters=clusters,
                       rng_state=rng.bit_generator.state, step_count=0)


# ============================================================
# M5 元学习参数子集（8 个高影响力参数）
# ============================================================

META_PARAMS_M5 = [
    'sigma_z', 'sigma_x',
    'w_body', 'w_social', 'w_cognitive',
    'gamma', 'exploration_bonus', 'temperature',
]

# 参数边界（非负约束 + 合理上限）
PARAM_CLIP = {
    'sigma_z': (0.01, 2.0),
    'sigma_x': (0.1, 5.0),
    'w_body': (0.01, 5.0),
    'w_social': (0.01, 5.0),
    'w_cognitive': (0.01, 5.0),
    'gamma': (0.1, 0.99),
    'exploration_bonus': (0.001, 2.0),
    'temperature': (0.1, 10.0),
}


# ============================================================
# MetaLearner — M5 激活
# ============================================================

class MetaLearner:
    """元学习器：在线有限差分梯度下降 (M5)

    M5 机制:
    - 每 META_INTERVAL 步: 对 8 个参数做有限差分 → 梯度下降
    - 可塑性衰减: update *= plasticity_decay^step
    - 创伤: apply_trauma() → 社会参数骤降

    v6.1 发育机制:
    - GluN2B→GluN2A 指数衰减 (半衰期 ~5000 步)
    - 4 阶段发育年龄 (婴儿→儿童→青少年→成人)
    - get_developmental_factors() 返回当前阶段调制因子
    """

    META_INTERVAL = 50  # 元更新间隔

    def __init__(self, theta: Theta):
        self.theta = theta
        self.history: list[tuple[Theta, float]] = []
        self.step_count: int = 0
        self.is_critical: bool = True   # v6.1: 保留向后兼容，由 GluN2B 推导
        self.trauma_applied: bool = False
        # v6.1: 发育追踪
        self.developmental_stage: int = 1
        self.stage_name: str = "婴儿期 (Infant)"
        self._glun2b_half_life: float = 5000.0  # GluN2B 转换半衰期 (步)

    # ================================================================
    # 真实有限差分梯度估计 (M5)
    # ================================================================

    def _estimate_gradient_real(self, z: np.ndarray, s: np.ndarray,
                                 net, hab, beliefs=None) -> dict:
        """真实有限差分: 对每个参数扰动 → 重新计算 F → 中心差分

        grad_i = [F(θ_i + ε) - F(θ_i - ε)] / (2ε)
        """
        from cerebrum.limbic_system.cingulate import compute_free_energy

        grads = {}
        eps = self.theta.grad_epsilon

        for param_name in META_PARAMS_M5:
            orig = float(getattr(self.theta, param_name))

            # F(θ + ε)
            setattr(self.theta, param_name, orig + eps)
            F_plus = compute_free_energy(
                z, s, net, self.theta, hab, beliefs
            ).total

            # F(θ - ε)
            setattr(self.theta, param_name, orig - eps)
            F_minus = compute_free_energy(
                z, s, net, self.theta, hab, beliefs
            ).total

            # 中心差分
            grads[param_name] = (F_plus - F_minus) / (2.0 * eps)

            # 恢复原值
            setattr(self.theta, param_name, orig)

        return grads

    # ================================================================
    # 梯度应用 (M5)
    # ================================================================

    def _apply_gradients(self, grads: dict):
        """应用梯度下降: θ -= lr × grad × plasticity

        v6.1: 使用 GluN2B 连续调制替代二元关键期。
        高 GluN2B → 高 meta_lr (发育早期可塑性高).
        """
        # 有效学习率: GluN2B 调制 × 可塑性衰减
        glun2b_mod = 0.5 + 0.5 * self.theta.glun2b_ratio  # [0.55, 1.0]
        effective_lr = self.theta.meta_lr * glun2b_mod
        effective_lr *= self.theta.plasticity_decay ** self.step_count

        for param_name, grad in grads.items():
            old_val = float(getattr(self.theta, param_name))
            new_val = old_val - effective_lr * grad

            # 边界裁剪
            lo, hi = PARAM_CLIP.get(param_name, (0.001, 10.0))
            new_val = np.clip(new_val, lo, hi)

            setattr(self.theta, param_name, float(new_val))

    # ================================================================
    # v6.1: 发育因子
    # ================================================================

    def get_developmental_factors(self) -> dict:
        """返回当前发育阶段的所有调制因子.

        Returns:
            dict with:
              stage, stage_name, glun2b_ratio,
              learn_rate_mult, threshold_mult, pnn_rate,
              silent_synapse_bonus, prune_aggressiveness,
              is_infant, is_child, is_adolescent, is_adult
        """
        stage_factors = DevelopmentalStage.get_factors(self.developmental_stage)
        return {
            'stage': self.developmental_stage,
            'stage_name': self.stage_name,
            'glun2b_ratio': self.theta.glun2b_ratio,
            'step_count': self.step_count,
            'learn_rate_mult': stage_factors['learn_rate_mult'],
            'threshold_mult': stage_factors['threshold_mult'],
            'pnn_rate': stage_factors['pnn_rate'],
            'silent_synapse_bonus': stage_factors['silent_synapse_bonus'],
            'prune_aggressiveness': stage_factors['prune_aggressiveness'],
            'is_infant': self.developmental_stage == 1,
            'is_child': self.developmental_stage == 2,
            'is_adolescent': self.developmental_stage == 3,
            'is_adult': self.developmental_stage == 4,
        }

    # ================================================================
    # 单步更新 (核心入口)
    # ================================================================

    def update(self, F: float, z: np.ndarray = None,
               s: np.ndarray = None, net=None, hab=None,
               beliefs=None):
        """单步元更新

        M1 行为 (z=None): 仅记录历史快照，不更新参数
        M5 行为 (z!=None): 每 META_INTERVAL 步执行有限差分梯度下降

        v6.1: GluN2B 连续轨迹 + 发育阶段判定

        Args:
            F: 当前自由能
            z: 隐状态（M5 需要）
            s: 感知向量（M5 需要）
            net: 簇网络（M5 需要）
            hab: 习惯化追踪器（M5 需要）
            beliefs: 社会信念（可选）
        """
        self.step_count += 1

        # ---- v6.1: GluN2B→GluN2A 指数衰减 (半衰期 ~5000 步) ----
        self.theta.glun2b_ratio = float(
            0.1 + 0.8 * np.exp(-self.step_count / self._glun2b_half_life))

        # ---- v6.1: 发育阶段判定 ----
        new_stage = DevelopmentalStage.get_stage(self.step_count)
        if new_stage != self.developmental_stage:
            self.developmental_stage = new_stage
            self.stage_name = DevelopmentalStage.get_name(new_stage)
        self.theta.developmental_stage = self.developmental_stage

        # 向后兼容: is_critical = 仍在婴儿/儿童期
        self.is_critical = (self.developmental_stage <= 2)

        # 记录快照（始终执行）
        snapshot = Theta(**self.theta.__dict__)
        self.history.append((snapshot, F))

        # M5: 真实梯度更新
        if (z is not None and s is not None and net is not None
                and hab is not None):
            if self.step_count % self.META_INTERVAL == 0:
                grads = self._estimate_gradient_real(
                    z, s, net, hab, beliefs)
                self._apply_gradients(grads)

    # ================================================================
    # 创伤模拟
    # ================================================================

    def apply_trauma(self):
        """模拟创伤事件: 永久修改社会参数"""
        self.theta.w_social = 0.1
        self.theta.eta_valence = 0.1
        self.trauma_applied = True

    # ================================================================
    # 发育轨迹
    # ================================================================

    def get_trajectory(self) -> dict:
        """返回所有快照的参数发育轨迹 (v6.1: +GluN2B + 发育阶段)"""
        if not self.history:
            return {}
        param_names = list(META_PARAMS_M5)
        trajectories = {p: [] for p in param_names}
        trajectories['F'] = []
        trajectories['step'] = []
        trajectories['is_critical'] = []
        # v6.1: 发育轨迹
        trajectories['glun2b_ratio'] = []
        trajectories['developmental_stage'] = []
        for i, (theta_snap, F_val) in enumerate(self.history):
            for p in param_names:
                trajectories[p].append(getattr(theta_snap, p))
            trajectories['F'].append(F_val)
            trajectories['step'].append(i)
            trajectories['is_critical'].append(
                i < self.theta.critical_window)
            trajectories['glun2b_ratio'].append(
                getattr(theta_snap, 'glun2b_ratio', 0.5))
            trajectories['developmental_stage'].append(
                getattr(theta_snap, 'developmental_stage', 1))
        return trajectories
