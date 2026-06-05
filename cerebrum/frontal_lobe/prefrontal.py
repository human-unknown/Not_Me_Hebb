"""
layer2_inference.py —— L2 主动推理 + 行动选择
自由能原理智能体 — v2: Cluster-driven experience (no GHistory)

核心流程:
1. predict_next_state():  模拟 z' = z + action_effect + noise
2. compute_G():           期望自由能 = 务实价值 + 认知价值
3. select_action():       五阶段集群驱动算法 (v2)
   - Phase 1: 宏观域筛选
   - Phase 2: 紧急度 → 时间预算
   - Phase 3: 集群经验排序 → recall(s_pred).G_ema
   - Phase 4: MoE 加权评估
   - Phase 5: 事后更新 → 写回 Cluster.G_ema/F_signal
"""

import numpy as np
from cns.data_types import (
    D, H, A, Theta, Action, AgentBelief,
    ACTION_DIRECTIONS,
)
from cerebrum.limbic_system.hippocampus import predict_sensations, ClusterNetwork
from cerebrum.basal_ganglia.action_gating import MoEGate


# ============================================================
# 社会信念更新 (M3)
# ============================================================

def update_social_beliefs(
    s: np.ndarray,
    beliefs: AgentBelief,
    my_pos: np.ndarray,
    theta: Theta,
    lr: float = 0.1,
) -> None:
    """更新社会信念：其他 agent 位置预测 + 信任度 (M3)"""
    obs_idx = 0
    other_ids = sorted(beliefs.other_positions.keys())
    if not other_ids:
        return

    for aid in other_ids:
        base = 80 + obs_idx * 8
        if base + 1 >= len(s):
            break

        sig_len = 2  # M3 gridworld: 2-dim position; Stage 4 overrides in main
        observed = s[base:base + sig_len]
        if aid in beliefs.other_positions:
            old = np.array(beliefs.other_positions[aid])
            if len(old) < sig_len:
                old = np.pad(old, (0, sig_len - len(old)))
            elif len(old) > sig_len:
                old = old[:sig_len]
            beliefs.other_positions[aid] = (1 - lr) * old + lr * observed
        else:
            beliefs.other_positions[aid] = observed.copy()

        pred_err = np.linalg.norm(observed - beliefs.other_positions[aid])
        trust_delta = np.exp(-pred_err * 5.0)
        current_trust = beliefs.trust_levels.get(aid, 0.5)
        beliefs.trust_levels[aid] = 0.9 * current_trust + 0.1 * trust_delta
        beliefs.second_order[aid] = my_pos.copy()
        obs_idx += 1


# ============================================================
# 状态转移预测
# ============================================================

def predict_next_state(
    z: np.ndarray,
    a: int,
    theta: Theta,
    rng: np.random.Generator = None,
    F_context: np.ndarray = None,
    net: ClusterNetwork = None,
) -> np.ndarray:
    """状态转移预测 (Phase 2: 集群模式补全)

    1. 尝试 recall([F_context | action_type]) → 补全 s_next
    2. 回退: z + 行动方向偏移 + 噪声 (手写物理)

    Args:
        z: 当前隐状态 (backward compat)
        a: 行动索引
        theta: 参数
        rng: 随机数
        F_context: [F_body, F_social, F_cognitive, valence, arousal] (5,)
        net: 集群网络
    """
    if rng is None:
        rng = np.random.default_rng()

    # Phase 2: 尝试集群模式补全
    if F_context is not None and net is not None and net.n_clusters > 0:
        onehot = np.zeros(5)
        onehot[a] = 1.0
        # Query: [zeros(48) | F_context(5) | action_onehot(5)] → 匹配 centroid[:48]=s_next
        from cns.data_types import S_CORE
        query = np.zeros(D)
        query[S_CORE:S_CORE+5] = F_context
        query[S_CORE+5:S_CORE+10] = onehot
        c = net.recall(query)
        if c is not None:
            # centroid[:S_CORE] = s_next 补全的下一步感觉核心
            s_next = c.centroid[:S_CORE]
            z_next = z.copy()
            n_dims = min(len(z), S_CORE)
            z_next[:n_dims] = (1.0-0.1)*z[:n_dims] + 0.1*s_next[:n_dims]
            return z_next

    # 回退: 手写物理模拟
    delta = np.zeros(H)
    direction = ACTION_DIRECTIONS[a] * 0.01
    delta[:2] = direction
    noise = theta.sigma_z * 0.01 * rng.normal(0, 1, H)
    return z + delta + noise


# ============================================================
# 期望自由能 G(a)
# ============================================================

def compute_G(
    z: np.ndarray,
    a: int,
    net: ClusterNetwork,
    theta: Theta,
    rng: np.random.Generator = None,
    F_context: np.ndarray = None,
    z_next: np.ndarray = None,
    s_pred: np.ndarray = None,
) -> float:
    """计算行动 a 的期望自由能

    G(a) = pragmatic_value - epistemic_value

    可选 z_next/s_pred: 外部预计算 → 避免 select_action 中重复随机模拟
    """
    if rng is None:
        rng = np.random.default_rng()

    if z_next is None:
        z_next = predict_next_state(z, a, theta, rng, F_context, net)
    if s_pred is None:
        s_pred = predict_sensations(z_next, theta)

    # 实用价值: 基于当前身体偏离 (F_context[0] = F_body)
    # F_body 越高 → 越需要行动来降低自由能
    F_body_now = F_context[0] if F_context is not None else 0.0
    pragmatic = F_body_now * theta.gamma
    # 认知价值: 信息增益 (新颖性驱动探索)
    best_sim = net.best_similarity(s_pred)
    info_gain = -np.log(max(best_sim, 0.01) + 1e-8)
    return pragmatic - theta.exploration_bonus * info_gain


# ============================================================
# L2-L4 递归多层次预期自由能 (Stage 1.5)
# ============================================================

def compute_G_recursive(
    z: np.ndarray, a: int, net: ClusterNetwork, theta: Theta,
    rng: np.random.Generator, F_context: np.ndarray,
    depth: int = 1, beta: float = 0.3,
    z_next_cache: np.ndarray = None, s_pred_cache: np.ndarray = None,
) -> float:
    """递归多层 G(a)。depth=1 → 单步, depth=2 → 序列, depth=3 → 情节。

    G(a) = G₁(a) + β·min_{a'}(G(a' after a))

    depth>1 时在预测的下一步状态上递归，衰减系数 β。
    """
    G1 = compute_G(z, a, net, theta, rng, F_context,
                   z_next=z_next_cache, s_pred=s_pred_cache)

    if depth <= 1:
        return G1

    # 预测下一状态
    z_next = (z_next_cache if z_next_cache is not None
              else predict_next_state(z, a, theta, rng, F_context, net))
    s_next = predict_sensations(z_next, theta)

    # 从集群估计下一步的 F_context
    c = net.recall(s_next) if net.n_clusters > 0 else None
    if c is not None:
        F_next = np.array([c.F_signal * 0.7, 0.0, 0.0, 0.0, 0.0])
    else:
        F_next = F_context.copy()

    # 递归找下一步最优
    best_next = float('inf')
    for next_a in range(A):
        Gn = compute_G_recursive(z_next, next_a, net, theta, rng, F_next,
                                 depth=depth - 1, beta=beta)
        best_next = min(best_next, Gn)

    return G1 + beta * best_next


# ============================================================
# 行动选择 —— 五阶段集群驱动算法 (v2: 无 GHistory)
# ============================================================

def select_action(
    z: np.ndarray,
    net: ClusterNetwork,
    theta: Theta,
    moe_gate: MoEGate,
    beliefs: AgentBelief,
    step: int,
    recent_F: float = 0.0,
    F_context: np.ndarray = None,
    rng: np.random.Generator = None,
    human_active: bool = True,
    dialogue_mode: bool = False,
) -> Action:
    """五阶段集群驱动行动选择 (v2)

    Args:
        human_active: 当前帧是否有人类输入 (仅 dialogue_mode=True 时生效)
        dialogue_mode: 是否为对话模式。True → A₃=表达, 启用耦合偏置。
                       False (默认) → gridworld 模式, A₃=东, 不应用偏置。
    """
    if rng is None:
        rng = np.random.default_rng()
    if F_context is None:
        F_context = np.zeros(5)

    # ---- Phase 1: 域筛选 (v3: Theta 驱动阈值, 不用硬编码 1.5) ----
    # 阈值由 w_body / w_social 比率动态决定:
    #   w_body >> w_social → 身体主导的阈值降低 (更容易切换到身体模式)
    #   w_social >> w_body → 社会主导的阈值降低 (更容易切换到社会模式)
    # 默认平衡时 (w_body=w_social=1.0) → threshold=1.5 (与原来行为一致)
    total_w = theta.w_body + theta.w_social + 1e-8
    dominance_ratio = theta.w_body / total_w  # [0, 1], 默认 0.5
    body_threshold = 0.5 + (1.0 - dominance_ratio) * 2.0   # [0.5, 2.5]
    social_threshold = 0.5 + dominance_ratio * 2.0          # [0.5, 2.5]

    F_body_dom = F_context[0]
    F_social_dom = F_context[1]
    if F_body_dom > F_social_dom * body_threshold:
        candidates = [0, 1, 2, 3]       # 身体主导 → 移动
    elif F_social_dom > F_body_dom * social_threshold:
        candidates = [0, 1, 2, 3, 4]    # 社会主导 → 含 REST 观察
    else:
        candidates = list(range(A))      # 均衡
    # A₄ 总是候选 — 通用恢复行动，不占域权重
    if 4 not in candidates:
        candidates.append(4)

    # ---- Phase 2: 紧急度 ----
    urgency = theta.urgency_weight * (1.0 if recent_F > 1.0 else 0.0)
    tau = 1.0 + urgency
    # 低紧急 → 深度递归 (depth>1), 高紧急 → 单步
    g_depth = max(1, int(tau))

    # ---- Phase 3+4 合并: 单遍计算 G + 集群查询 ----
    weights = moe_gate.compute_weights(z, beliefs, step)
    confidence = moe_gate.get_confidence(weights)

    # 无经验时随机打乱 → 避免 argmin 总选 A₀
    rng.shuffle(candidates)

    scored = []
    for a in candidates:
        z_next = predict_next_state(z, a, theta, rng, F_context, net)
        s_pred = predict_sensations(z_next, theta)
        c = net.recall(s_pred)
        # 传递预计算的 z_next/s_pred，避免重复随机模拟
        G = compute_G_recursive(z, a, net, theta, rng, F_context,
                                depth=g_depth, beta=theta.gamma,
                                z_next_cache=z_next, s_pred_cache=s_pred)
        priority = c.G_ema if c is not None else 0.0
        scored.append((priority, a, G * tau, G, c))

    scored.sort(key=lambda x: x[0])  # 按 G_ema 排序
    G_vals = [s[2] for s in scored]
    G_raw = [s[3] for s in scored]
    matched_clusters = [s[4] for s in scored]
    candidates = [s[1] for s in scored]

    # ---- Phase 4b: 社会上下文偏置 (M3) ----
    # v3: 信任阈值用 theta 参数, 不用硬编码 0.6/0.4
    social_bias = np.zeros(len(candidates))
    if beliefs.other_positions and beliefs.trust_levels:
        avg_trust = float(np.mean(list(beliefs.trust_levels.values())))
        avg_other = np.zeros(2)
        n_others = 0
        for aid in beliefs.other_positions:
            avg_other += np.array(beliefs.other_positions[aid])
            n_others += 1
        if n_others > 0:
            avg_other /= n_others
            # 信任阈值: 高 > 0.6 = 趋近, 低 < 0.4 = 回避, 中间 = 中性
            # theta.negativity_bias 驱动信任敏感度 (越高=越容易不信任)
            trust_high = 0.5 + 0.1 * theta.negativity_bias  # ~0.65
            trust_low = 0.5 - 0.1 * theta.negativity_bias   # ~0.35
            for i, a in enumerate(candidates):
                direction = ACTION_DIRECTIONS[a]
                toward_score = np.dot(direction, avg_other)
                if avg_trust > trust_high:
                    social_bias[i] = -theta.w_social * 0.5 * toward_score
                elif avg_trust < trust_low:
                    social_bias[i] = theta.w_social * 0.5 * toward_score

    # ---- Phase 4c: 表达耦合偏置 (仅对话模式, A₃=表达) ----
    # v3: FEP 推导 — 表达动机来自预期 F_social 降低
    # 有人等回应 (human_active) → 不回应 = 社会预测误差↑ → F_social↑
    # 表达降低 F_social ≈ θ_w_social × (1.0 - trust) 的预测误差
    # 偏置 = 不回应带来的预期自由能增量
    #
    # gridworld 模式 (dialogue_mode=False) 不应用——A₃ 是移动动作
    expr_bias = np.zeros(len(candidates))
    if dialogue_mode:
        for i, a in enumerate(candidates):
            if a == 3:  # A₃ expression
                if human_active:
                    # 预期 F_social 降低 ← 回应满足社会预测
                    # 偏置 = -w_social * (预测误差降幅)
                    expected_F_reduction = theta.w_social * 1.5
                    expr_bias[i] = -expected_F_reduction
                else:
                    # 没人时: 自言自语无社会预测误差降低价值
                    # F_social 越高 (孤独感) → 越倾向表达 (寻求连接)
                    social_deprivation = max(0.0, F_context[1])
                    expr_bias[i] = theta.w_social * (2.5 - social_deprivation * 2.0)

    G_weighted = [g * confidence + g * (1 - confidence) + sb + eb
                  for g, sb, eb in zip(G_vals, social_bias, expr_bias)]

    best_idx = int(np.argmin(G_weighted))
    best_action = candidates[best_idx]

    # ---- Phase 5: 集群经验写回 (v2: 无 GHistory) ----
    moe_gate.update_budgets(best_action, G_weighted[best_idx])
    moe_gate.fatigue_step()

    # 将 G_ema + F_signal 写回匹配的 cluster
    best_G = G_weighted[best_idx]
    c_best = matched_clusters[best_idx]
    if c_best is not None:
        c_best.G_ema = 0.9 * c_best.G_ema + 0.1 * best_G
        c_best.F_signal = 0.9 * c_best.F_signal + 0.1 * recent_F

    return Action(
        index=best_action,
        expected_F=G_vals[best_idx],
        expected_G=best_G,
        confidence=confidence,
    )
