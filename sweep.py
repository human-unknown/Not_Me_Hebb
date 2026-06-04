"""
sweep.py —— 参数扫描与批量运行 (M4)
自由能原理智能体

功能:
- latin_hypercube(): 纯 numpy 拉丁超立方采样
- run_with_theta(): 用指定 Theta 运行单次 episode
- run_sweep(): 批量参数扫描
"""

import sys
import time
import numpy as np
from data_types import Theta
from gridworld import GridWorld
from agent import Agent


# ============================================================
# 拉丁超立方采样
# ============================================================

# 20 个连续可微参数及其边界 (v3: 包含 negativity_bias, w_accuracy, w_F_signal)
SWEEP_PARAMS = {
    'sigma_z':           (0.01, 1.0),
    'sigma_x':           (0.1,  5.0),
    'decay_rate':        (0.001, 0.1),
    'cluster_threshold': (0.5,  0.99),
    'learn_rate_l0':     (0.001, 0.5),
    'w_body':            (0.1,  5.0),
    'w_social':          (0.1,  5.0),
    'w_cognitive':       (0.1,  5.0),
    'eta_valence':       (0.1,  2.0),
    'eta_arousal':       (0.1,  2.0),
    'habituation_tau':   (1.0,  100.0),
    'negativity_bias':   (0.5,  5.0),
    'w_accuracy':        (0.05, 2.0),
    'w_F_signal':        (0.01, 1.0),
    'gamma':             (0.1,  0.99),
    'exploration_bonus': (0.01, 1.0),
    'temperature':       (0.1,  10.0),
    'urgency_weight':    (0.05, 1.0),
    'meta_lr':           (0.001, 0.1),
    'plasticity_decay':  (0.99, 0.9999),
}


def latin_hypercube(n_samples: int, bounds: dict = None,
                    rng: np.random.Generator = None) -> list[Theta]:
    """纯 numpy 拉丁超立方采样

    算法:
    1. 将每个参数维度分成 n_samples 个等概率区间
    2. 每个区间内随机采样一个值
    3. 随机打乱各维度的顺序

    Args:
        n_samples: 采样数量
        bounds: 参数边界字典，默认 SWEEP_PARAMS
        rng: 随机数生成器

    Returns:
        list of Theta，长度 = n_samples
    """
    if bounds is None:
        bounds = SWEEP_PARAMS
    if rng is None:
        rng = np.random.default_rng()

    param_names = list(bounds.keys())
    n_params = len(param_names)
    thetas = []

    # 为每个参数生成 LHS 矩阵
    samples = np.zeros((n_samples, n_params))
    for j in range(n_params):
        lo, hi = bounds[param_names[j]]
        # 将 [0, 1] 分成 n_samples 等份，每份内均匀采样
        segment_size = 1.0 / n_samples
        for i in range(n_samples):
            segment_low = i * segment_size
            samples[i, j] = segment_low + rng.uniform(0, segment_size)
        # 随机打乱该维度
        rng.shuffle(samples[:, j])
        # 映射到实际范围
        samples[:, j] = lo + samples[:, j] * (hi - lo)

    # 构造 Theta 对象
    for i in range(n_samples):
        theta_dict = {}
        for j, name in enumerate(param_names):
            theta_dict[name] = float(samples[i, j])
        # 填充非扫描参数为默认值 (v3: 包含新增 FEP 参数)
        theta_dict.setdefault('n_policy_samples', 16)
        theta_dict.setdefault('critical_window', 1000)
        theta_dict.setdefault('grad_epsilon', 0.001)
        theta_dict.setdefault('negativity_bias', 1.5)
        theta_dict.setdefault('w_accuracy', 0.5)
        theta_dict.setdefault('w_F_signal', 0.1)
        thetas.append(Theta(**theta_dict))

    return thetas


# ============================================================
# 单次 Theta 运行
# ============================================================

def run_with_theta(theta: Theta, seed: int = 42, steps: int = 300,
                   use_vision: bool = False, verbose: bool = False,
                   log_interval: int = 1000) -> dict:
    """用指定 Theta 配置运行单次 episode（M1 单智能体模式）

    Args:
        theta: 参数配置
        seed: 随机种子
        steps: 运行步数
        use_vision: 是否启用视野遮蔽 (M2)
        verbose: 是否打印日志
        log_interval: 日志间隔

    Returns:
        episode result dict（同 main.run_episode 格式）
    """
    rng = np.random.default_rng(seed)
    world = GridWorld(size=10, n_agents=1, rng=rng)
    agent = Agent(rng=rng)
    agent.theta = theta  # 覆盖默认参数

    rewards = []
    actions = []
    pos_history = []

    for t in range(steps):
        if use_vision:
            s = world.get_visible_sensory(3.0, agent_id=0)
        else:
            s = world.get_sensory(agent_id=0)

        action = agent.step(s, t)
        reward = world.step(action.index, agent_id=0)
        agent.add_reward(reward)

        rewards.append(reward)
        actions.append(action.index)
        pos_history.append(world.agent_positions[0].copy())

    return {
        'rewards': rewards,
        'actions': actions,
        'F_history': agent.F_history,
        'F_body_history': agent.F_body_history,
        'F_social_history': agent.F_social_history,
        'F_cognitive_history': agent.F_cognitive_history,
        'valence_history': agent.valence_history,
        'arousal_history': agent.arousal_history,
        'attention_history': agent.attention_history,
        'n_clusters': agent.net.n_clusters,
        'pos_history': pos_history,
        'total_reward': world.total_reward,
        'elapsed': 0.0,
        'theta_snapshots': agent.theta_snapshots,
        'coverage': world.compute_coverage(pos_history),
    }


# ============================================================
# 批量扫描
# ============================================================

def run_sweep(n_samples: int = 50, steps: int = 200,
              use_vision: bool = False, seed: int = 42,
              verbose: bool = True) -> dict:
    """批量参数扫描

    Args:
        n_samples: LHS 采样数量
        steps: 每个 episode 的步数
        use_vision: 是否启用视野遮蔽
        seed: 基准随机种子
        verbose: 是否打印进度

    Returns:
        {thetas, results, features, embeddings, labels, n_clusters}
    """
    from features import extract_features, features_matrix
    from attractors import pca_reduce, cluster_features, track_attractors

    # 采样
    rng = np.random.default_rng(seed)
    thetas = latin_hypercube(n_samples, SWEEP_PARAMS, rng)

    # 批量运行
    results = []
    t_start = time.perf_counter()

    for i, theta in enumerate(thetas):
        if verbose and (i % max(1, n_samples // 10) == 0):
            print(f"  [{i}/{n_samples}] running...")

        result = run_with_theta(theta, seed=seed, steps=steps,
                               use_vision=use_vision, verbose=False)
        results.append(result)

    t_end = time.perf_counter()

    # 特征提取
    feats = features_matrix(results)

    # PCA 降维
    emb = pca_reduce(feats)

    # 聚类
    labels = cluster_features(emb)

    # 追踪
    att = track_attractors(emb, labels)

    if verbose:
        print(f"\n  Sweep complete: {n_samples} samples × {steps} steps "
              f"in {t_end - t_start:.1f}s")
        print(f"  Attractors found: {att['n_clusters']}")
        print(f"  Cluster sizes: {att['sizes']}")

    return {
        'thetas': thetas,
        'results': results,
        'features': feats,
        'embeddings': emb,
        'labels': labels,
        'n_clusters': att['n_clusters'],
        'cluster_sizes': att['sizes'],
        'elapsed': t_end - t_start,
    }


# ============================================================
# 命令行测试
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  M4 Parameter Sweep Test")
    print("=" * 60)

    # 小规模测试
    print("  Generating 5 LHS samples...")
    thetas = latin_hypercube(5)
    for i, t in enumerate(thetas):
        print(f"    Theta[{i}]: sigma_z={t.sigma_z:.3f}, "
              f"w_body={t.w_body:.2f}, gamma={t.gamma:.2f}")

    print("\n  Running mini sweep (5 samples × 100 steps)...")
    result = run_sweep(n_samples=5, steps=100, verbose=True)
    print(f"\n  Done: {result['n_clusters']} attractors found")
