"""
layer3_meta.py —— L3 元参数管理 (M5 激活)
自由能原理智能体

功能:
- create_default_theta(): 20 参数默认配置
- MetaLearner: 在线元学习（M5 真实有限差分梯度下降）

发育机制:
- 关键期: critical_window 步内学习率 2×
- 可塑性衰减: plasticity_decay 逐步降低更新幅度
- 创伤模拟: apply_trauma() 永久修改社会参数
"""

import numpy as np
from data_types import Theta, FreeEnergy


# ============================================================
# 默认参数创建
# ============================================================

def create_default_theta() -> Theta:
    """返回 23 个参数的默认配置"""
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
    )


# ============================================================
# 种子包创建
# ============================================================

def create_seed_package(theta=None, rng=None, bootstrap_sensory=None):
    from data_types import SeedPackage
    from layer0_model import bootstrap_clusters
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
    - 关键期: step < critical_window → 学习率 2×
    - 可塑性衰减: update *= plasticity_decay^step
    - 创伤: apply_trauma() → 社会参数骤降
    """

    META_INTERVAL = 50  # 元更新间隔

    def __init__(self, theta: Theta):
        self.theta = theta
        self.history: list[tuple[Theta, float]] = []
        self.step_count: int = 0
        self.is_critical: bool = True
        self.trauma_applied: bool = False

    # ================================================================
    # 真实有限差分梯度估计 (M5)
    # ================================================================

    def _estimate_gradient_real(self, z: np.ndarray, s: np.ndarray,
                                 net, hab, beliefs=None) -> dict:
        """真实有限差分: 对每个参数扰动 → 重新计算 F → 中心差分

        grad_i = [F(θ_i + ε) - F(θ_i - ε)] / (2ε)
        """
        from layer1_free_energy import compute_free_energy

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
        """应用梯度下降: θ -= lr × grad × plasticity × critical_bonus

        约束:
        - 关键期学习率 2×
        - 可塑性衰减
        - 非负 + 边界裁剪
        """
        # 有效学习率
        effective_lr = self.theta.meta_lr
        if self.is_critical:
            effective_lr *= 2.0
        effective_lr *= self.theta.plasticity_decay ** self.step_count

        for param_name, grad in grads.items():
            old_val = float(getattr(self.theta, param_name))
            new_val = old_val - effective_lr * grad

            # 边界裁剪
            lo, hi = PARAM_CLIP.get(param_name, (0.001, 10.0))
            new_val = np.clip(new_val, lo, hi)

            setattr(self.theta, param_name, float(new_val))

    # ================================================================
    # 单步更新 (核心入口)
    # ================================================================

    def update(self, F: float, z: np.ndarray = None,
               s: np.ndarray = None, net=None, hab=None,
               beliefs=None):
        """单步元更新

        M1 行为 (z=None): 仅记录历史快照，不更新参数
        M5 行为 (z!=None): 每 META_INTERVAL 步执行有限差分梯度下降

        Args:
            F: 当前自由能
            z: 隐状态（M5 需要）
            s: 感知向量（M5 需要）
            net: 簇网络（M5 需要）
            hab: 习惯化追踪器（M5 需要）
            beliefs: 社会信念（可选）
        """
        self.step_count += 1

        # 关键期判断
        if self.step_count > self.theta.critical_window:
            self.is_critical = False

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
        """返回所有快照的参数发育轨迹"""
        if not self.history:
            return {}
        param_names = list(META_PARAMS_M5)
        trajectories = {p: [] for p in param_names}
        trajectories['F'] = []
        trajectories['step'] = []
        trajectories['is_critical'] = []
        for i, (theta_snap, F_val) in enumerate(self.history):
            for p in param_names:
                trajectories[p].append(getattr(theta_snap, p))
            trajectories['F'].append(F_val)
            trajectories['step'].append(i)
            trajectories['is_critical'].append(
                i < self.theta.critical_window)
        return trajectories
