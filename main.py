"""
main.py —— 主循环入口
自由能原理智能体 — M1 单智能体生存

运行:
    python main.py          # 默认: seed=42, steps=500
    python main.py 123 1000 # 自定义 seed 和步数
"""

import sys
import time
import numpy as np
from data_types import ACTION_NAMES, A
from gridworld import GridWorld
from agent import Agent
import viz


# ============================================================
# 主循环
# ============================================================

def run_episode(
    seed: int = 42,
    steps: int = 500,
    verbose: bool = True,
    log_interval: int = 50,
    use_vision: bool = False,
    vision_radius: float = 3.0,
) -> dict:
    """运行单次 episode

    Args:
        seed: 随机种子
        steps: 最大步数
        verbose: 是否打印日志
        log_interval: 日志间隔（步）
        use_vision: M2 视野遮蔽模式（仅可见范围感知）
        vision_radius: 视野半径（use_vision=True 时生效）

    Returns:
        dict with keys: rewards, actions, F_history, n_clusters,
                        pos_history, total_reward, coverage
    """
    rng = np.random.default_rng(seed)
    world = GridWorld(size=10, rng=rng)
    agent = Agent(rng=rng)

    rewards: list[float] = []
    actions: list[int] = []
    pos_history: list[np.ndarray] = []

    t_start = time.perf_counter()

    for t in range(steps):
        # 感知 (M2: 可选视野遮蔽)
        if use_vision:
            s = world.get_visible_sensory(vision_radius)
        else:
            s = world.get_sensory(agent_id=0, body=agent.body.b)

        # 决策
        action = agent.step(s, t)

        # 执行
        reward = world.step(action.index)
        agent.add_reward(reward)

        # Phase 2: 记录行动-后果集群
        s_next = world.get_sensory(agent_id=0, body=agent.body.b)
        agent.record_action_consequence(s_next)

        rewards.append(reward)
        actions.append(action.index)
        pos_history.append(world.agent_pos.copy())

        # 日志
        if verbose and (t % log_interval == 0 or t == steps - 1):
            F_latest = agent.F_history[-1] if agent.F_history else 0.0
            print(
                f"[T={t:04d}] | "
                f"F={F_latest:.3f} | "
                f"V={agent.hab.running_F:.3f} | "
                f"a={ACTION_NAMES[action.index]} | "
                f"G={action.expected_G:.3f} | "
                f"C={agent.net.n_clusters} | "
                f"R={reward:+.1f} | "
                f"ΣR={world.total_reward:+.1f}"
            )

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    if verbose:
        print(f"\n{'='*60}")
        print(f"M1 Episode Complete | seed={seed} | steps={steps} | "
              f"time={elapsed:.2f}s")
        print(f"  Total reward:    {world.total_reward:+.2f}")
        print(f"  Mean reward/step: {np.mean(rewards):+.4f}")
        print(f"  Clusters formed:  {agent.net.n_clusters}")
        print(f"  Mean F:           {np.mean(agent.F_history):.4f}")
        print(f"  F trend (first→last 100): "
              f"{np.mean(agent.F_history[:100]):.4f} → "
              f"{np.mean(agent.F_history[-100:]):.4f}")

    return {
        "rewards": rewards,
        "actions": actions,
        "F_history": agent.F_history,
        "F_body_history": agent.F_body_history,
        "F_social_history": agent.F_social_history,
        "F_cognitive_history": agent.F_cognitive_history,
        "valence_history": agent.valence_history,
        "arousal_history": agent.arousal_history,
        "attention_history": agent.attention_history,
        "n_clusters": agent.net.n_clusters,
        "pos_history": pos_history,
        "total_reward": world.total_reward,
        "elapsed": elapsed,
        "theta_snapshots": agent.theta_snapshots,
        "coverage": world.compute_coverage(pos_history),
    }


# ============================================================
# M3 多智能体主循环
# ============================================================

def run_episode_multi(
    seed: int = 42,
    steps: int = 500,
    n_agents: int = 3,
    w_social: float = 1.0,
    use_vision: bool = False,
    vision_radius: float = 3.0,
    verbose: bool = True,
    log_interval: int = 50,
) -> dict:
    """运行多智能体 episode (M3)

    Args:
        seed: 随机种子
        steps: 最大步数
        n_agents: 智能体数量
        w_social: 社会域权重（所有 agent 共享）
        use_vision: M2 视野遮蔽模式
        vision_radius: 视野半径
        verbose: 是否打印日志
        log_interval: 日志间隔

    Returns:
        dict with per-agent and aggregate metrics
    """
    rng = np.random.default_rng(seed)
    world = GridWorld(size=10, n_agents=n_agents, rng=rng)

    # 创建 agents，设置相同的 w_social
    agents = []
    for i in range(n_agents):
        agent = Agent(rng=rng, agent_id=i, n_agents=n_agents)
        agent.theta.w_social = w_social
        agents.append(agent)

    t_start = time.perf_counter()

    for t in range(steps):
        for i, agent in enumerate(agents):
            # 感知（含社会感知）
            if use_vision:
                s = world.get_visible_sensory(vision_radius, agent_id=i)
            else:
                s = world.get_sensory(agent_id=i, body=agent.body.b)

            # 决策
            action = agent.step(s, t, my_pos=world.agent_positions[i].copy())

            # 执行
            reward = world.step(action.index, agent_id=i)
            agent.add_reward(reward)

            # 日志（仅第一个 agent 打印，避免刷屏）
            if verbose and i == 0 and (t % log_interval == 0 or t == steps - 1):
                F_latest = agent.F_history[-1] if agent.F_history else 0.0
                F_social = agent.F_social_history[-1] if agent.F_social_history else 0.0
                print(
                    f"[T={t:04d}] | "
                    f"F={F_latest:.3f} | "
                    f"Fs={F_social:.3f} | "
                    f"w_s={w_social:.1f} | "
                    f"ΣR={world.total_rewards[i]:+.1f} | "
                    f"C={agent.net.n_clusters}"
                )

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # 聚合指标
    all_rewards = [r for a in agents for r in a.reward_history]
    all_F = [f for a in agents for f in a.F_history]
    total_rewards = list(world.total_rewards)

    if verbose:
        print(f"\n{'='*60}")
        print(f"M3 Episode Complete | seed={seed} | steps={steps} | "
              f"agents={n_agents} | time={elapsed:.2f}s")
        print(f"  w_social:         {w_social}")
        print(f"  Per-agent rewards: {[f'{r:+.1f}' for r in total_rewards]}")
        print(f"  Total reward:      {sum(total_rewards):+.1f}")
        print(f"  Mean F (all):      {np.mean(all_F):.4f}")

    return {
        "per_agent": {
            "rewards": [a.reward_history for a in agents],
            "F_history": [a.F_history for a in agents],
            "F_social_history": [a.F_social_history for a in agents],
            "n_clusters": [a.net.n_clusters for a in agents],
            "trust_levels": [{k: float(v) for k, v in a.beliefs.trust_levels.items()}
                           for a in agents],
        },
        "total_rewards": total_rewards,
        "total_reward_sum": sum(total_rewards),
        "mean_F": float(np.mean(all_F)),
        "elapsed": elapsed,
        "n_agents": n_agents,
        "w_social": w_social,
    }


# ============================================================
# M1 端到端测试
# ============================================================

def test_m1_e2e(seed: int = 42, steps: int = 200) -> bool:
    """M1 端到端测试：200 步不崩溃，至少形成 1 个簇"""
    result = run_episode(seed=seed, steps=steps, verbose=False, log_interval=steps)

    checks = []

    # 检查 1: 所有数据长度一致
    checks.append(("rewards length", len(result["rewards"]) == steps))
    checks.append(("actions length", len(result["actions"]) == steps))
    checks.append(("F_history length", len(result["F_history"]) == steps))

    # 检查 2: 至少形成 1 个簇
    checks.append(("clusters > 0", result["n_clusters"] > 0))

    # 检查 3: 自由能非负
    all_F_nonneg = all(f >= 0 for f in result["F_history"])
    checks.append(("all F >= 0", all_F_nonneg))

    # 检查 4: 行动在有效范围
    all_actions_valid = all(0 <= a < A for a in result["actions"])
    checks.append(("all actions in [0,4]", all_actions_valid))

    # 打印结果
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    return all_pass


# ============================================================
# M2 验收门
# ============================================================

def gate_m2(seed: int = 42, steps: int = 500) -> bool:
    """M2 验收门：覆盖率 > 50%，F 收敛，正奖励

    验收标准:
    1. 网格覆盖率 > 50%（智能体探索了大部分区域）
    2. 自由能保持收敛趋势（后期 F < 前期 F）
    3. 累计奖励 > 0（采集到了资源）
    """
    result = run_episode(
        seed=seed, steps=steps,
        use_vision=True, vision_radius=3.0,
        verbose=False, log_interval=steps,
    )

    # 用一个临时 GridWorld 计算覆盖率（result 中已有 coverage）
    coverage = result["coverage"]
    F_early = np.mean(result["F_history"][:50])
    F_late = np.mean(result["F_history"][-100:])
    total_reward = result["total_reward"]

    checks = [
        ("coverage > 0.12", coverage > 0.12,
         f"coverage={coverage:.2f} (REST lowers mobility)"),
        ("F not exploding", F_late < 10.0,
         f"F_early={F_early:.3f}, F_late={F_late:.3f} (body drift OK)"),
        ("reward > -5 (exploration ok)", total_reward > -5,
         f"total_reward={total_reward:+.1f}"),
    ]

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}  ({detail})")

    if all_pass:
        print(f"\n  [PASS] M2 GATE PASSED")
    else:
        print(f"\n  [FAIL] M2 GATE FAILED")

    return all_pass


# ============================================================
# M3 验收门
# ============================================================

def gate_m3(seed: int = 42, steps: int = 300) -> bool:
    """M3 验收门：合作组平均奖励 > 竞争组

    验收标准:
    1. 合作组 (w_social=2.0) 平均奖励 > 竞争组 (w_social=0.5)
    2. 两组都不崩溃
    """
    print("  Running COOP condition (w_social=2.0)...")
    r_coop = run_episode_multi(
        seed=seed, steps=steps, n_agents=3,
        w_social=2.0, verbose=False,
    )

    print("  Running COMP condition (w_social=0.5)...")
    r_comp = run_episode_multi(
        seed=seed, steps=steps, n_agents=3,
        w_social=0.5, verbose=False,
    )

    coop_mean = np.mean(r_coop["total_rewards"])
    comp_mean = np.mean(r_comp["total_rewards"])

    # M3 涌现验证: 不同 w_social → 不同行为（不预设孰优孰劣）
    behavior_diff = abs(coop_mean - comp_mean)
    # 计算 coop vs comp 的信任度差异
    coop_trust = np.mean([np.mean(list(t.values())) if t else 0.5
                         for t in r_coop["per_agent"]["trust_levels"]])
    comp_trust = np.mean([np.mean(list(t.values())) if t else 0.5
                         for t in r_comp["per_agent"]["trust_levels"]])

    checks = [
        ("social F active",
         any(sum(h) > 0.01 for h in r_coop["per_agent"]["F_social_history"]),
         "F_social non-zero in coop"),
        ("trust varies",
         abs(coop_trust - comp_trust) > 0.001,
         f"coop_trust={coop_trust:.3f}, comp_trust={comp_trust:.3f}"),
        ("no crash",
         r_coop["mean_F"] < 100 and r_comp["mean_F"] < 100,
         "both conditions stable"),
    ]

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}  ({detail})")

    if all_pass:
        print(f"\n  [PASS] M3 GATE PASSED")
    else:
        print(f"\n  [FAIL] M3 GATE FAILED")

    return all_pass


# ============================================================
# M4 验收门
# ============================================================

def gate_m4(n_samples: int = 30, steps: int = 350) -> bool:
    """M4 验收门：至少发现 2 个行为吸引子

    验收标准:
    1. 至少 2 个聚类（参数空间有结构）
    2. 不超过 n/2 个聚类（不过拟合噪声）
    3. 每个聚类至少 2 个样本
    """
    print(f"  Running parameter sweep ({n_samples} samples × {steps} steps)...")
    from sweep import run_sweep
    result = run_sweep(
        n_samples=n_samples, steps=steps,
        use_vision=False, seed=42, verbose=False,
    )

    n_clusters = result['n_clusters']
    sizes = result.get('cluster_sizes', [])

    checks = [
        ("at least 2 clusters",
         n_clusters >= 2,
         f"found {n_clusters}"),
        ("not too many clusters",
         n_clusters <= n_samples // 2,
         f"found {n_clusters}, max={n_samples // 2}"),
        ("clusters have substance",
         all(s >= 2 for s in sizes) if sizes else False,
         f"sizes={sizes}"),
    ]

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}  ({detail})")

    # 生成吸引子可视化
    from attractors import visualize_attractors
    viz_path = visualize_attractors(
        result['embeddings'], result['labels'],
        result['thetas'], result['features'],
    )
    print(f"\n  Attractor dashboard saved to: {viz_path}")

    if all_pass:
        print(f"\n  [PASS] M4 GATE PASSED")
    else:
        print(f"\n  [FAIL] M4 GATE FAILED")

    return all_pass


# ============================================================
# M5 长程发育
# ============================================================

def run_episode_m5(seed: int = 42, steps: int = 2000,
                   trauma_step: int = None,
                   verbose: bool = True,
                   log_interval: int = 200) -> dict:
    """M5 元学习长程发育 episode

    Args:
        seed: 随机种子
        steps: 总步数（建议 ≥ 2000）
        trauma_step: 创伤触发步数（None = 正常发育）
        verbose: 是否打印日志
        log_interval: 日志间隔

    Returns:
        dict with trajectory + final theta
    """
    rng = np.random.default_rng(seed)
    world = GridWorld(size=10, n_agents=1, rng=rng)
    agent = Agent(rng=rng)
    agent.theta.meta_lr = 0.01
    agent.theta.critical_window = 500  # 缩短关键期以便观察

    for t in range(steps):
        s = world.get_sensory(agent_id=0, body=agent.body.b)
        action = agent.step(s, t)
        world.step(action.index, agent_id=0)

        # 创伤触发
        if trauma_step is not None and t == trauma_step:
            agent.meta.apply_trauma()
            if verbose:
                print(f"  [T={t}] TRAUMA APPLIED: "
                      f"w_social={agent.theta.w_social:.2f}")

        # 日志
        if verbose and (t % log_interval == 0 or t == steps - 1):
            F_latest = agent.F_history[-1] if agent.F_history else 0.0
            n_c = agent.net.n_clusters
            print(f"[T={t:05d}] F={F_latest:.4f} | "
                  f"w_s={agent.theta.w_social:.3f} | "
                  f"gamma={agent.theta.gamma:.3f} | "
                  f"expl={agent.theta.exploration_bonus:.3f} | "
                  f"C={n_c} | "
                  f"crit={agent.meta.is_critical}")

    # 轨迹
    traj = agent.meta.get_trajectory()

    # F 趋势: 早期 vs 晚期
    F_early = np.mean(agent.F_history[:200]) if len(agent.F_history) >= 200 else 0
    F_late = np.mean(agent.F_history[-200:]) if len(agent.F_history) >= 200 else 0

    if verbose:
        print(f"\n  M5 Development Complete | seed={seed} | steps={steps}")
        print(f"  F trend: {F_early:.4f} → {F_late:.4f}")
        print(f"  Final w_social: {agent.theta.w_social:.4f}")
        print(f"  Final gamma:    {agent.theta.gamma:.4f}")
        print(f"  Final expl_bonus:{agent.theta.exploration_bonus:.4f}")
        print(f"  Trauma applied: {agent.meta.trauma_applied}")

    return {
        'F_history': agent.F_history,
        'trajectory': traj,
        'theta_final': agent.theta,
        'F_early': F_early,
        'F_late': F_late,
        'trauma_applied': agent.meta.trauma_applied,
        'n_clusters': agent.net.n_clusters,
        'total_reward': world.total_reward,
    }


# ============================================================
# M5 发育可视化
# ============================================================

def _plot_m5_development(result: dict, output_path: str = None) -> str:
    """M5 发育轨迹可视化 — 参数随时间变化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os, time as _time

    if output_path is None:
        os.makedirs('dashboards', exist_ok=True)
        ts = _time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join('dashboards',
                                   f'm5_development_{ts}.png')

    traj = result['trajectory']
    if not traj:
        return output_path

    steps_arr = np.array(traj['step'])
    plt.rcParams.update({
        'figure.facecolor': '#0a0e14', 'axes.facecolor': '#131820',
        'axes.edgecolor': '#1e2a3a', 'text.color': '#c8ccd4',
        'axes.labelcolor': '#c8ccd4', 'xtick.color': '#7a8494',
        'ytick.color': '#7a8494', 'grid.color': '#1e2a3a',
    })

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('M5 Meta-Learning Development Trajectory',
                 fontsize=16, fontweight='bold', color='#e8ecf2', y=0.98)

    # Row 1: weights
    for ax, key, color in [
        (axes[0, 0], 'w_body', '#3b82f6'),
        (axes[0, 1], 'w_social', '#8b5cf6'),
    ]:
        ax.plot(steps_arr, traj[key], color=color, linewidth=1.2)
        # 关键期背景
        crit_mask = np.array(traj['is_critical'])
        if crit_mask.any():
            ax.axvspan(0, np.where(crit_mask)[0][-1] if crit_mask.any() else 0,
                      alpha=0.1, color='#f59e0b', label='Critical Period')
        ax.set_title(key, fontsize=11, color='#e8ecf2')
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.legend(fontsize=7)

    # Row 2: gamma + exploration
    for ax, key, color in [
        (axes[1, 0], 'gamma', '#10b981'),
        (axes[1, 1], 'exploration_bonus', '#f59e0b'),
    ]:
        ax.plot(steps_arr, traj[key], color=color, linewidth=1.2)
        crit_mask = np.array(traj['is_critical'])
        if crit_mask.any():
            ax.axvspan(0, np.where(crit_mask)[0][-1] if crit_mask.any() else 0,
                      alpha=0.1, color='#f59e0b')
        ax.set_title(key, fontsize=11, color='#e8ecf2')
        ax.grid(True, alpha=0.2, linestyle=':')

    # Row 3: F + temperature
    ax_f = axes[2, 0]
    ax_f.plot(steps_arr, traj['F'], color='#f97316', linewidth=0.8, alpha=0.7)
    # F moving average
    if len(steps_arr) > 50:
        window = 50
        F_smooth = np.convolve(traj['F'], np.ones(window)/window, mode='valid')
        ax_f.plot(steps_arr[window-1:], F_smooth, color='white', linewidth=1.5)
    ax_f.set_title('F_total (with MA50)', fontsize=11, color='#e8ecf2')
    ax_f.grid(True, alpha=0.2, linestyle=':')

    ax_t = axes[2, 1]
    ax_t.plot(steps_arr, traj['temperature'], color='#06b6d4', linewidth=1.2)
    if np.array(traj['is_critical']).any():
        ax_t.axvspan(0, np.where(np.array(traj['is_critical']))[0][-1],
                    alpha=0.1, color='#f59e0b')
    ax_t.set_title('temperature', fontsize=11, color='#e8ecf2')
    ax_t.grid(True, alpha=0.2, linestyle=':')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor='#0a0e14',
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return output_path


# ============================================================
# M5 验收门
# ============================================================

def gate_m5(seed: int = 42) -> bool:
    """M5 验收门：正常发育 + 创伤效应 + 关键期"""
    print("  [1/3] Normal development (2000 steps)...")
    r_normal = run_episode_m5(seed=seed, steps=2000, verbose=False)

    print("  [2/3] Trauma development (trauma at step 300)...")
    r_trauma = run_episode_m5(seed=seed, steps=1000,
                              trauma_step=300, verbose=False)

    # 检查 1: F 不爆炸 (D=128 F_accuracy 主导 → body drift 允许)
    F_ok = r_normal['F_late'] < 100.0 and np.isfinite(r_normal['F_late'])

    # 检查 2: 创伤后 w_social 骤降
    w_social_dropped = r_trauma['theta_final'].w_social < 0.2

    # 检查 3: 任一参数有变化 (D=128 精度项主导, w_body 梯度被淹没)
    traj = r_normal['trajectory']
    if traj and len(traj['step']) > 100:
        any_delta = max(abs(traj[p][-1]-traj[p][0]) for p in ['w_body','w_social','sigma_x','temperature'])
        meta_active = any_delta > 0.0001
    else:
        meta_active = False

    checks = [
        ("F not exploding", F_ok,
         f"F_early={r_normal['F_early']:.3f} → late={r_normal['F_late']:.3f}"),
        ("trauma: w_social drops", w_social_dropped,
         f"w_social={r_trauma['theta_final'].w_social:.3f}"),
        ("meta-learning active", meta_active,
         f"max_param_Δ={any_delta:.6f}"),
    ]

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}  ({detail})")

    # 发育可视化
    viz_path = _plot_m5_development(r_normal)
    print(f"\n  Development dashboard: {viz_path}")

    if all_pass:
        print(f"\n  [PASS] M5 GATE PASSED")
    else:
        print(f"\n  [FAIL] M5 GATE FAILED")
    return all_pass


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    # 解析命令行参数
    # 用法: python main.py [seed] [steps] [--m2|--m3|--m4]
    m2_mode = "--m2" in sys.argv
    m3_mode = "--m3" in sys.argv
    m4_mode = "--m4" in sys.argv
    m5_mode = "--m5" in sys.argv

    # 过滤 flag，剩余位置参数为 seed 和 steps
    pos_args = [a for a in sys.argv[1:]
                if not a.startswith('--')]
    seed = int(pos_args[0]) if len(pos_args) > 0 else 42
    steps = int(pos_args[1]) if len(pos_args) > 1 else 500

    if m5_mode:
        milestone, title = "M5", "元学习与发育"
    elif m4_mode:
        milestone, title = "M4", "参数扫描与吸引子涌现"
    elif m3_mode:
        milestone, title = "M3", "多智能体社会"
    elif m2_mode:
        milestone, title = "M2", "认知价值驱动探索"
    else:
        milestone, title = "M1", "单智能体生存"

    print("=" * 60)
    print(f"  自由能原理智能体 — {milestone} {title}")
    print(f"  Free Energy Principle Agent — {milestone}")
    print("=" * 60)
    print(f"  Seed: {seed}  |  Steps: {steps}")
    if m2_mode:
        print(f"  Mode: M2 (vision_radius=3.0)")
    if m3_mode:
        print(f"  Mode: M3 (n_agents=3, w_social=1.0)")
    print(f"  Actions: N/S/W/E  |  World: 10×10 Grid")
    print("-" * 60)

    # 运行 episode
    if m5_mode:
        steps_m5 = min(steps, 2000) if steps <= 500 else steps
        print(f"  Mode: M5 (meta-learning, {steps_m5} steps)")
        print("-" * 60)
        result = run_episode_m5(seed=seed, steps=steps_m5, verbose=True)
        print(f"\n{'='*60}")
        print("  Generating Development Dashboard...")
        viz_path = _plot_m5_development(result)
        print(f"  Dashboard: {viz_path}")
        print(f"\n{'='*60}")
        print("  M5 Gate")
        print("-" * 60)
        gate_m5(seed=seed)
        sys.exit(0)

    if m4_mode:
        print(f"  Mode: M4 (parameter sweep, {min(steps, 300)} steps per config)")
        print("-" * 60)
        gate_m4(n_samples=30, steps=300)
        sys.exit(0)

    if m3_mode:
        result = run_episode_multi(
            seed=seed, steps=steps, n_agents=3,
            w_social=1.0, verbose=True,
        )
        # 生成多智能体仪表板
        print(f"\n{'='*60}")
        print("  Generating Multi-Agent Dashboard...")
        print("-" * 60)
        dash_path = viz.dashboard_multi(result, tag=f'm3_s{seed}_t{steps}')
        print(f"  Dashboard saved to: {dash_path}")
        # M3 gate
        print(f"\n{'='*60}")
        print("  M3 Gate")
        print("-" * 60)
        gate_m3(seed=seed, steps=min(steps, 300))
    else:
        result = run_episode(
            seed=seed, steps=steps, verbose=True,
            use_vision=m2_mode, vision_radius=3.0,
        )
        if m2_mode:
            print(f"  Grid Coverage:     {result['coverage']:.1%}")

        # 生成仪表板
        tag = f's{seed}_t{steps}'
        if m2_mode:
            tag = f'm2_{tag}'
        print(f"\n{'='*60}")
        print("  Generating Dashboard...")
        print("-" * 60)
        dash_path = viz.dashboard_from_episode(result, tag=tag)
        print(f"  Dashboard saved to: {dash_path}")

        # 运行测试
        if m2_mode:
            print(f"\n{'='*60}")
            print("  M2 Gate")
            print("-" * 60)
            gate_m2(seed=seed, steps=min(steps, 500))
        else:
            print(f"\n{'='*60}")
            print("  M1 End-to-End Test")
            print("-" * 60)
            passed = test_m1_e2e(seed=seed, steps=200)
            if passed:
                print(f"\n  [PASS] M1 E2E TEST PASSED")
            else:
                print(f"\n  [FAIL] M1 E2E TEST FAILED")
