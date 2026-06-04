"""
features.py —— 行为特征提取 (M4)
自由能原理智能体

从 episode 结果中提取 13 维行为特征向量，
用于参数空间分析和吸引子发现。
"""

import numpy as np

# 特征名称（按索引顺序）
FEATURE_NAMES = [
    'mean_F',           # 0  平均自由能
    'F_trend',          # 1  F 收敛斜率
    'F_std',            # 2  F 标准差
    'mean_valence',     # 3  平均效价
    'mean_arousal',     # 4  平均唤醒度
    'mean_attention',   # 5  平均注意力精度
    'coverage',         # 6  网格覆盖率
    'total_reward',     # 7  累计奖励
    'action_entropy',   # 8  行动多样性
    'n_clusters',       # 9  知识积累（簇数）
    'F_body_ratio',     # 10 生理域主导度
    'F_cognitive_mean', # 11 认知负担
    'reward_trend',     # 12 奖励改善趋势
    'rest_ratio',       # 13 REST 行动占比 (v2)
    'body_deviation',   # 14 身体偏离设定点均值 (v2)
]


def extract_features(result: dict) -> np.ndarray:
    """从单次 episode 结果提取 13 维行为特征

    Args:
        result: run_episode() 或 run_with_theta() 的返回值

    Returns:
        np.ndarray shape (13,) — 各维度已做基本归一化
    """
    F_hist = np.array(result['F_history'])
    F_body = np.array(result.get('F_body_history', [0]))
    F_cog = np.array(result.get('F_cognitive_history', [0]))
    valence = np.array(result.get('valence_history', [0]))
    arousal = np.array(result.get('arousal_history', [0]))
    attention = np.array(result.get('attention_history', [0]))
    rewards = np.array(result['rewards'])
    actions = np.array(result['actions'])
    steps = len(rewards)

    # 1. 平均自由能
    mean_F = float(np.mean(F_hist))

    # 2. F 收敛趋势: (后期 - 前期) / steps
    split = max(steps // 5, 10)
    F_trend = float((np.mean(F_hist[-split:]) - np.mean(F_hist[:split]))
                    / max(steps, 1))

    # 3. F 标准差（稳定性）
    F_std = float(np.std(F_hist))

    # 4. 平均效价
    mean_valence = float(np.mean(valence))

    # 5. 平均唤醒度
    mean_arousal = float(np.mean(arousal))

    # 6. 平均注意力
    mean_attention = float(np.mean(attention))

    # 7. 覆盖率
    coverage = float(result.get('coverage', 0.0))

    # 8. 累计奖励
    total_reward = float(result.get('total_reward', np.sum(rewards)))

    # 9. 行动熵（行为多样性）
    counts = np.bincount(actions, minlength=4)
    probs = counts / (counts.sum() + 1e-8)
    probs = probs[probs > 0]
    action_entropy = float(-np.sum(probs * np.log2(probs)))

    # 10. 簇数
    n_clusters = float(result.get('n_clusters', 0))

    # 11. F_body 占比
    F_body_ratio = float(np.mean(F_body) / (mean_F + 1e-8))

    # 12. 认知自由能均值
    F_cognitive_mean = float(np.mean(F_cog))

    # 13. 奖励趋势
    r_split = max(steps // 5, 10)
    reward_trend = float(np.mean(rewards[-r_split:])
                         - np.mean(rewards[:r_split]))

    # 14. REST 行动占比 (v2)
    rest_ratio = float(np.sum(actions == 4) / max(steps, 1))

    # 15. 身体偏离均值 (v2) — 从 F_body 近似
    body_dev = float(np.mean(F_body))

    features = np.array([
        mean_F, F_trend, F_std,
        mean_valence, mean_arousal, mean_attention,
        coverage, total_reward, action_entropy,
        n_clusters, F_body_ratio, F_cognitive_mean,
        reward_trend, rest_ratio, body_dev,
    ], dtype=float)

    return features


def features_matrix(results: list[dict]) -> np.ndarray:
    """批量提取特征 → (n_samples, 13) 矩阵

    自动做 Z-score 归一化以便 PCA 聚类。
    """
    feats = np.array([extract_features(r) for r in results])

    # Z-score 归一化（按列）
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    feats_norm = (feats - mean) / std

    return feats_norm


# ============================================================
# 命令行测试
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  M4 Feature Extraction Test")
    print("=" * 60)

    # 合成数据测试
    rng = np.random.default_rng(42)
    steps = 200

    demo = {
        'F_history': (0.5 + 0.1 * rng.normal(0, 1, steps)).tolist(),
        'F_body_history': (0.3 + 0.05 * rng.normal(0, 1, steps)).tolist(),
        'F_social_history': np.zeros(steps).tolist(),
        'F_cognitive_history': (0.1 + 0.02 * rng.normal(0, 1, steps)).tolist(),
        'valence_history': np.tanh(-0.5 * np.array(
            [0.5] * steps)).tolist(),
        'arousal_history': (0.3 + 0.1 * rng.random(steps)).tolist(),
        'attention_history': (0.3 + 0.1 * rng.random(steps)).tolist(),
        'rewards': rng.choice([-0.5, 0.0, 1.0], steps).tolist(),
        'actions': rng.integers(0, 4, steps).tolist(),
        'n_clusters': 5,
        'pos_history': np.cumsum(rng.normal(0, 0.3, (steps, 2)), axis=0).tolist(),
        'total_reward': 2.5,
        'coverage': 0.45,
    }

    feats = extract_features(demo)
    print("  Feature vector (13 dims):")
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, feats)):
        print(f"    [{i:2d}] {name:20s} = {val:+.4f}")

    # 批量测试
    demos = [demo] * 3
    mat = features_matrix(demos)
    print(f"\n  Feature matrix shape: {mat.shape}")
