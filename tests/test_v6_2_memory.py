"""
test_v6_2_memory.py — v6.2 记忆巩固优化 单元测试

测试:
  1. 突触标签 — 标签设置与衰减
  2. 标签捕获 — 高唤醒触发带标签簇的额外学习
  3. 激活持续性 — CaMKII 样窗口调制阈值/LR
  4. 巩固锁定 — 多轮睡眠后 decay 降低
  5. STC 时序链 — 端到端: A标签→B强学习→A被捕获
  6. 持久化 roundtrip — 新字段正确序列化/反序列化
"""

import sys
import os
import numpy as np

# 确保可以导入项目模块 — 从项目根目录运行
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from cns.data_types import (
    D, H, K, Cluster, CandidateCluster, Theta,
    DevelopmentalStage, FreeEnergy,
)


def make_theta(**overrides) -> Theta:
    """创建测试用 Theta (v6.2: 48 params)."""
    t = Theta()
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def make_random_s(text_bias: float = 0.5) -> np.ndarray:
    """生成随机感知向量 (文本段有信号)."""
    s = np.zeros(D, dtype=np.float32)
    s[:64] = np.random.randn(64).astype(np.float32) * text_bias
    return s


def make_deterministic_s(seed: int, text_signal: float = 1.0) -> np.ndarray:
    """生成确定性的感知向量 (文本段, 前16维有特征)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(D, dtype=np.float32)
    s[:16] = rng.normal(0, text_signal, 16).astype(np.float32)
    s[16:64] = rng.normal(0, 0.1, 48).astype(np.float32)
    return s


# ================================================================
# Test 1: 突触标签 — 设置与衰减
# ================================================================

def test_synaptic_tagging():
    """验证: recall 匹配设置标签 + 标签随时间衰减 + 过期清除."""
    from cerebrum.limbic_system.hippocampus import ClusterNetwork

    t = make_theta(cluster_threshold=0.5, learn_rate_l0=0.1,
                   tag_window=10, tag_decay_rate=0.2)
    net = ClusterNetwork(t)

    # 学习模式A
    s1 = make_deterministic_s(42, text_signal=1.5)
    c1 = net.learn(s1)

    # recall 触发标签设置
    s_query = s1 + np.random.randn(D).astype(np.float32) * 0.01
    recalled = net.recall(s_query)
    assert recalled is not None, "Should recall the learned cluster"

    # 标签应该被设置 (recall 时 tag = activation * 0.5)
    assert c1.tag > 0.0, f"Tag should be set on recall match, got {c1.tag:.4f}"
    assert c1.tag_age == 0, f"Tag age should be 0 on set, got {c1.tag_age}"

    initial_tag = c1.tag

    # 衰减几步
    for _ in range(5):
        net.decay()

    # 标签应衰减但未过期
    assert c1.tag < initial_tag, f"Tag should decay: {c1.tag:.4f} < {initial_tag:.4f}"
    assert c1.tag > 0.0, f"Tag should be positive after 5 decays, got {c1.tag:.4f}"
    assert c1.tag_age == 5, f"Tag age should advance"

    # 衰减更久 → 标签清除 (超过 tag_window=10)
    for _ in range(10):
        net.decay()

    # 过期或被衰减归零
    assert c1.tag < 0.01 or c1.tag_age > t.tag_window, \
        f"Tag should be expired: tag={c1.tag:.6f}, age={c1.tag_age}"

    print(f"  [PASS] test_synaptic_tagging: tag set={initial_tag:.4f}, "
          f"decayed to {c1.tag:.6f}")


# ================================================================
# Test 2: 标签捕获 — 高唤醒触发带标签簇的额外学习
# ================================================================

def test_tag_capture():
    """验证: 高唤醒时 .capture_tags() 增强带标签簇的激活."""
    from cerebrum.limbic_system.hippocampus import ClusterNetwork

    t = make_theta(cluster_threshold=0.5, learn_rate_l0=0.1,
                   tag_window=30, tag_decay_rate=0.01,
                   tag_capture_strength=0.3)
    net = ClusterNetwork(t)

    # 创建3个簇，给其中2个设置标签
    s_a = make_deterministic_s(10, 1.5)
    s_b = make_deterministic_s(20, 1.5)
    s_c = make_deterministic_s(30, 1.5)

    ca = net.learn(s_a)
    cb = net.learn(s_b)
    cc = net.learn(s_c)

    # 手动设置标签 (模拟低唤醒弱激活)
    ca.tag = 0.5
    ca.tag_age = 1
    cb.tag = 0.3
    cb.tag_age = 1
    # cc 无标签

    act_before_ca = ca.activation
    act_before_cb = cb.activation
    act_before_cc = cc.activation

    # 高唤醒事件 → 触发标签捕获
    net.capture_tags(arousal=0.8, F_body_delta=0.3)

    # ca (标签最高) 应该获得最大激活提升
    assert ca.activation > act_before_ca, \
        f"Tagged cluster A should get activation boost: {act_before_ca:.4f} → {ca.activation:.4f}"
    assert cb.activation > act_before_cb, \
        f"Tagged cluster B should get activation boost: {act_before_cb:.4f} → {cb.activation:.4f}"
    # cc (无标签) 不应改变
    assert abs(cc.activation - act_before_cc) < 0.001, \
        f"Untagged cluster C should not change: {act_before_cc:.4f} → {cc.activation:.4f}"

    # 标签应在捕获后被消耗 (减半)
    assert ca.tag < 0.5, f"Tag should be consumed after capture: {ca.tag:.4f}"
    assert cb.tag < 0.3, f"Tag should be consumed after capture: {cb.tag:.4f}"

    print(f"  [PASS] test_tag_capture: A boost {act_before_ca:.4f}→{ca.activation:.4f}, "
          f"B boost {act_before_cb:.4f}→{cb.activation:.4f}, "
          f"C unchanged {cc.activation:.4f}")


# ================================================================
# Test 3: 激活持续性 — CaMKII 窗口调制阈值/LR
# ================================================================

def test_activation_persistence():
    """验证: 激活持续性降低匹配阈值 + 提升学习率."""
    from cerebrum.limbic_system.hippocampus import ClusterNetwork

    t = make_theta(cluster_threshold=0.7, learn_rate_l0=0.1,
                   persistence_decay_rate=0.2,
                   persistence_threshold_boost=0.3,
                   persistence_lr_boost=0.8)
    net = ClusterNetwork(t)

    # 创建簇A
    s_a = make_deterministic_s(55, 1.5)
    ca = net.learn(s_a)

    # recall 触发 → persistence = 1.0
    net.recall(s_a)
    assert ca.activation_persistence > 0.9, \
        f"Persistence should be ~1.0 after recall, got {ca.activation_persistence:.4f}"

    # _persistence_factor: 高 persistence → 低阈值, 高LR
    thresh_f, lr_f = net._persistence_factor(ca)
    assert thresh_f < 1.0, f"Threshold factor should be <1.0 with persistence, got {thresh_f:.4f}"
    assert lr_f > 1.0, f"LR factor should be >1.0 with persistence, got {lr_f:.4f}"

    # 衰减 persistence
    for _ in range(10):
        net.decay()

    assert ca.activation_persistence < 0.5, \
        f"Persistence should decay: {ca.activation_persistence:.4f}"

    # 低 persistence → 接近默认因子
    thresh_f2, lr_f2 = net._persistence_factor(ca)
    assert thresh_f2 > thresh_f, f"Threshold factor should recover as persistence decays"
    assert lr_f2 < lr_f, f"LR factor should drop as persistence decays"

    print(f"  [PASS] test_activation_persistence: persist peak={1.0:.1f}→{ca.activation_persistence:.3f}, "
          f"thresh_f peak={thresh_f:.3f}→{thresh_f2:.3f}, "
          f"lr_f peak={lr_f:.3f}→{lr_f2:.3f}")


# ================================================================
# Test 4: 巩固锁定 — 多轮睡眠后 decay 降低
# ================================================================

def test_consolidation_lock():
    """验证: 多次睡眠巩固 → consolidation_count↑ → effective decay↓."""
    from cerebrum.limbic_system.hippocampus import ClusterNetwork, sleep_replay

    t = make_theta(cluster_threshold=0.3, learn_rate_l0=0.1,
                   decay_rate=0.05,
                   consolidation_lock_factor=1.0,
                   consolidation_lock_max=10)
    net = ClusterNetwork(t)

    # 创建一些簇
    for i in range(5):
        s = make_deterministic_s(100 + i, 1.5)
        net.learn(s)

    # 初始 consolidation_count = 0
    for c in net.clusters:
        assert c.consolidation_count == 0, "Initial consolidation_count should be 0"

    # sleep_replay 会递增 consolidation_count
    stats = sleep_replay(net, t)
    for c in net.clusters:
        assert c.consolidation_count == 1, \
            f"After 1st sleep, consolidation_count should be 1, got {c.consolidation_count}"

    # 再一轮睡眠
    stats2 = sleep_replay(net, t)
    for c in net.clusters:
        assert c.consolidation_count == 2, \
            f"After 2nd sleep, consolidation_count should be 2, got {c.consolidation_count}"

    # 验证 decay 统计
    assert 'n_locked_clusters' in stats2, "Stats should include n_locked_clusters"
    assert stats2['n_locked_clusters'] == net.n_clusters, \
        f"All clusters should be locked: {stats2['n_locked_clusters']}"
    assert 'mean_consolidation_lock' in stats2, "Stats should include mean_consolidation_lock"
    assert stats2['mean_consolidation_lock'] == 2.0, \
        f"Mean lock should be 2.0, got {stats2['mean_consolidation_lock']}"

    # 验证 decay 实际效果: 创建新簇, 经历sleep → decay变慢
    s_test = make_deterministic_s(200, 1.5)
    c_new = net.learn(s_test)
    c_new.activation = 1.0  # 设置高激活
    act_before = c_new.activation

    net.decay()  # 1次decay → consolidation_count=0 所以无锁保护
    drop_no_lock = act_before - c_new.activation
    assert drop_no_lock > 0, "Should have non-zero decay"

    # 给这个簇加锁
    c_new.consolidation_count = 5
    c_new.activation = 1.0
    net.decay()
    drop_with_lock = 1.0 - c_new.activation

    # 有锁的 decay 应该更小
    assert drop_with_lock < drop_no_lock, \
        f"Locked decay ({drop_with_lock:.6f}) should be < unlocked ({drop_no_lock:.6f})"

    print(f"  [PASS] test_consolidation_lock: stats={stats2['mean_consolidation_lock']:.1f}, "
          f"decay_unlocked={drop_no_lock:.4f}, decay_locked(×5)={drop_with_lock:.4f}")


# ================================================================
# Test 5: STC 时序链 — A标签→B强学习→A被B巩固
# ================================================================

def test_stc_temporal_chain():
    """端到端: 弱激活A→高唤醒B→A被标签捕获→AB时序关系被STDP增强.

    模拟: 你看到一只猫(A, 低唤醒) → 紧接着猫跳到你脸上(B, 高唤醒)
    → A的标签被捕获 → A获得额外巩固 → A→B的STDP链接增强。
    """
    from cerebrum.limbic_system.hippocampus import ClusterNetwork

    t = make_theta(cluster_threshold=0.5, learn_rate_l0=0.1,
                   tag_window=30, tag_decay_rate=0.02,
                   tag_capture_strength=0.4,
                   stdp_lr=0.05, stdp_window=5, stdp_weight=0.3)
    net = ClusterNetwork(t)

    # Phase 1: 弱事件A (低唤醒, 被标记)
    s_a = make_deterministic_s(300, 1.0)
    ca = net.learn(s_a)
    # recall 设置标签 (模拟A被激活)
    ca.tag = 0.6
    ca.tag_age = 0
    act_before_a = ca.activation

    # STDP: 记录A为上一个活跃簇
    net._last_activated_id = id(ca)
    net._last_activated_step = net._step_counter

    # Phase 2: 强事件B (高唤醒, 2步后)
    net._step_counter += 2
    s_b = make_deterministic_s(400, 1.5)
    cb = net.learn(s_b)
    act_after_learn_b = cb.activation

    # Phase 3: 高唤醒触发标签捕获 → A获得额外激活
    net.capture_tags(arousal=0.85, F_body_delta=0.5)

    # 验证: A的激活被标签捕获提升
    assert ca.activation > act_before_a, \
        f"Cluster A should be captured: {act_before_a:.4f} → {ca.activation:.4f}"

    # 验证: A→B存在STDP链接 (时序依赖)
    ca_id = id(ca)
    cb_id = id(cb)
    assert cb_id in ca.stdp_links, \
        f"STDP: A→B link should exist after temporal chain. Links: {ca.stdp_links}"
    stdp_weight = ca.stdp_links[cb_id]
    assert stdp_weight > 0, f"STDP weight should be positive, got {stdp_weight:.4f}"

    print(f"  [PASS] test_stc_temporal_chain: "
          f"A activ {act_before_a:.4f}→{ca.activation:.4f}, "
          f"STDP A→B weight={stdp_weight:.4f}, "
          f"tag consumed A={ca.tag:.4f}")


# ================================================================
# Test 6: 持久化 roundtrip — 新字段正确序列化
# ================================================================

def test_persistence_roundtrip():
    """验证: v6.2 新字段 (tag, tag_age, activation_persistence, consolidation_count)
    正确序列化到 pickle 数据并恢复."""
    from cerebrum.limbic_system.hippocampus import ClusterNetwork
    from cns.persistence import _save_cluster_network, _restore_cluster_network

    t = make_theta(cluster_threshold=0.3, learn_rate_l0=0.2)
    net = ClusterNetwork(t)

    # 创建带 v6.2 字段的簇
    from cns.data_types import Cluster
    created = []
    for i in range(4):
        s = np.zeros(D, dtype=np.float32)
        s[i * 12] = 5.0  # 每个簇在单一维度有强烈独特信号
        centroid = net.hash_features(s)
        c = Cluster(centroid=centroid)
        # v6.2 新字段
        c.tag = 0.1 * (i + 2)
        c.tag_age = i * 5
        c.activation_persistence = 0.3 * i
        c.consolidation_count = i + 1
        # v6.1 字段也设置 (确保兼容)
        c.protection_score = 0.05 * (i + 1)
        c.pnn_level = 0.03 * (i + 1)
        c.stdp_links = {999 + i: 0.2 * (i + 1)}
        net.clusters.append(c)
        key = net._hash_to_bucket(centroid)
        net.buckets.setdefault(key, []).append(c)
        created.append(c)

    n_expected = len(created)

    # 保存
    data = _save_cluster_network(net)

    # 验证保存数据包含新字段
    for cd in data['clusters']:
        assert 'tag' in cd, f"Save data should include 'tag'"
        assert 'tag_age' in cd, f"Save data should include 'tag_age'"
        assert 'activation_persistence' in cd, f"Save data should include 'activation_persistence'"
        assert 'consolidation_count' in cd, f"Save data should include 'consolidation_count'"

    # 创建新网络并恢复
    net2 = ClusterNetwork(t)
    _restore_cluster_network(net2, data)

    assert net2.n_clusters == n_expected, \
        f"Restored n_clusters mismatch: {net2.n_clusters} vs {n_expected}"

    # 验证新字段正确恢复
    for i in range(n_expected):
        c2 = net2.clusters[i]
        expected_tag = 0.1 * (i + 2)
        assert abs(c2.tag - expected_tag) < 0.001, \
            f"tag mismatch at {i}: {c2.tag:.4f} vs {expected_tag:.4f}"
        assert c2.tag_age == i * 5, \
            f"tag_age mismatch at {i}: {c2.tag_age} vs {i * 5}"
        expected_persist = 0.3 * i
        assert abs(c2.activation_persistence - expected_persist) < 0.001, \
            f"activation_persistence mismatch at {i}: {c2.activation_persistence:.4f} vs {expected_persist:.4f}"
        assert c2.consolidation_count == i + 1, \
            f"consolidation_count mismatch at {i}: {c2.consolidation_count} vs {i + 1}"

    print(f"  [PASS] test_persistence_roundtrip: all v6.2 fields saved/loaded correctly for {n_expected} clusters")


# ================================================================
# 运行所有测试
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("v6.2 记忆巩固优化 — 单元测试")
    print("=" * 60)

    tests = [
        ("突触标签", test_synaptic_tagging),
        ("标签捕获", test_tag_capture),
        ("激活持续性", test_activation_persistence),
        ("巩固锁定", test_consolidation_lock),
        ("STC 时序链", test_stc_temporal_chain),
        ("持久化 roundtrip", test_persistence_roundtrip),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, "
          f"{len(tests)} total")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("All v6.2 tests passed!")
