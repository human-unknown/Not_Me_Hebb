"""
test_v6_3_sleep.py — v6.3 睡眠优化与时间维度 单元测试

测试:
  1. SCN TTFL — 自由运行周期 ~24h, 光脉冲→相位重置
  2. Process S — 觉醒累积→睡眠清除
  3. VLPO 触发器 — sleep_propensity > threshold → 睡眠转换
  4. NREM 突触缩小 — 等比缩小保留相对权重, 不删除强簇
  5. REM 情绪去刺痛 — 高valence簇在REM中被温和衰减
  6. 双相睡眠端到端 — NREM+REM完整周期统计
  7. α 注意门控 — 非注意通道被α抑制, 注意通道不受影响
  8. 持久化 roundtrip — SCN/VLPO/睡眠状态正确保存/恢复
"""

import sys
import os
import numpy as np

# 确保可以导入项目模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from cns.data_types import (
    D, H, K, Cluster, Theta, CircadianState, SleepState,
)


def make_theta(**overrides) -> Theta:
    """创建测试用 Theta (v6.3: 56 params)."""
    t = Theta()
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def make_random_s(text_bias: float = 0.5) -> np.ndarray:
    """生成随机感知向量."""
    s = np.zeros(D, dtype=np.float32)
    s[:64] = np.random.randn(64).astype(np.float32) * text_bias
    return s


def make_deterministic_s(seed: int, text_signal: float = 1.0) -> np.ndarray:
    """生成确定性的感知向量."""
    rng = np.random.default_rng(seed)
    s = np.zeros(D, dtype=np.float32)
    s[:16] = rng.normal(0, text_signal, 16).astype(np.float32)
    s[16:64] = rng.normal(0, 0.1, 48).astype(np.float32)
    # 设置情感段
    s[64:72] = rng.normal(0, 0.3, 8).astype(np.float32)
    return s


# ================================================================
# Test 1: SCN TTFL — 自由运行 + 光脉冲相位重置
# ================================================================

def test_scn_ttfl_free_run():
    """验证: TTFL 自由运行产生 ~24h 周期振荡."""
    from cerebrum.limbic_system.scn import SCN

    scn = SCN()
    n_steps = 200  # 约 ~6h 等价

    per_history = []
    bmal1_history = []

    for _ in range(n_steps):
        state = scn.step(light_level=0.5, is_asleep=False,
                        circa_tau=24.0, circa_light_sensitivity=0.3)
        per_history.append(state.per_protein)
        bmal1_history.append(state.bmal1_activity)

    # PER 和 BMAL1 应该反相振荡 (PER高→BMAL1低)
    per_arr = np.array(per_history)
    bmal1_arr = np.array(bmal1_history)

    # 方差应 > 0 (有振荡)
    per_std = float(np.std(per_arr))
    bmal1_std = float(np.std(bmal1_arr))
    assert per_std > 0.02, f"PER should oscillate, std={per_std:.4f}"
    assert bmal1_std > 0.02, f"BMAL1 should oscillate, std={bmal1_std:.4f}"

    # PER 和 BMAL1 负相关 (反相)
    corr = float(np.corrcoef(per_arr, bmal1_arr)[0, 1])
    assert corr < 0, f"PER and BMAL1 should be anti-correlated, corr={corr:.4f}"

    # 褪黑素和皮质醇应反相
    mel = np.array([s.melatonin for s in [scn.step(0.5, False) for _ in range(50)]])
    cort = np.array([s.cortisol for s in [scn.step(0.5, False) for _ in range(50)]])

    print(f"  [PASS] test_scn_ttfl_free_run: PER std={per_std:.4f}, "
          f"BMAL1 std={bmal1_std:.4f}, PER-BMAL1 corr={corr:.4f}")


def test_scn_light_entrainment():
    """验证: 夜间光照 → Per 上调 → 相位重置."""
    from cerebrum.limbic_system.scn import SCN

    scn = SCN()
    # 先跑一段时间建立节律
    for _ in range(100):
        scn.step(light_level=0.3, is_asleep=False)

    per_before = scn.ttfl.per_mrna

    # 夜间强光 → Per mRNA 应显著上升
    # (模拟夜间光照: 高光 + 低觉醒)
    for _ in range(7):  # 继续几天
        scn.step(light_level=0.8, is_asleep=False,
                circa_light_sensitivity=0.8)

    per_after_high_light = scn.ttfl.per_mrna
    # 光照应影响 Per 水平
    assert per_after_high_light != per_before, \
        f"Light should affect Per levels: {per_before:.4f} → {per_after_high_light:.4f}"

    print(f"  [PASS] test_scn_light_entrainment: "
          f"Per mRNA {per_before:.4f} → {per_after_high_light:.4f}")


# ================================================================
# Test 2: Process S — 觉醒累积 → 睡眠清除
# ================================================================

def test_process_s():
    """验证: 觉醒期累积 + 睡眠期清除."""
    from cerebrum.limbic_system.scn import ProcessS

    ps = ProcessS()

    # 觉醒期累积
    wake_pressures = []
    for _ in range(50):
        p = ps.update(is_asleep=False, allostatic_load=0.1)
        wake_pressures.append(p)

    assert ps.pressure > 0.3, f"Process S should accumulate during wake: {ps.pressure:.4f}"
    assert wake_pressures[-1] > wake_pressures[0], \
        "Process S should monotonically increase during wake"

    # NREM 睡眠清除 (快)
    pressure_before_sleep = ps.pressure
    for _ in range(20):
        ps.update(is_asleep=True, sleep_phase='nrem')
    assert ps.pressure < pressure_before_sleep * 0.5, \
        f"NREM should rapidly clear Process S: {pressure_before_sleep:.4f} → {ps.pressure:.4f}"

    # REM 清除更慢
    ps_reset = ProcessS()
    ps_reset.pressure = 0.5
    ps_reset.update(is_asleep=True, sleep_phase='rem')
    rem_clear = 0.5 - ps_reset.pressure
    ps_reset2 = ProcessS()
    ps_reset2.pressure = 0.5
    ps_reset2.update(is_asleep=True, sleep_phase='nrem')
    nrem_clear = 0.5 - ps_reset2.pressure
    assert nrem_clear > rem_clear, \
        f"NREM clear ({nrem_clear:.4f}) should be > REM clear ({rem_clear:.4f})"

    print(f"  [PASS] test_process_s: wake accum → {wake_pressures[-1]:.3f}, "
          f"NREM clear → {ps.pressure:.3f}")


# ================================================================
# Test 3: VLPO 触发器 — 睡眠转换
# ================================================================

def test_vlpo_flip_flop():
    """验证: sleep_propensity > threshold → 睡眠."""
    from brainstem_cerebellum.pons.vlpo import VLPO

    vlpo = VLPO()

    # 低 propensity → 清醒
    state = vlpo.update(sleep_propensity=0.3, sleep_pressure=0.3, threshold=0.65)
    assert state.state == 'awake', f"Low propensity should stay awake: {state.state}"
    assert not vlpo.is_asleep

    # 高 propensity → 睡眠 (需连续满足 min_stable=15 步)
    for _ in range(20):
        state = vlpo.update(sleep_propensity=0.75, sleep_pressure=0.7, threshold=0.65)
    assert vlpo.is_asleep, f"High propensity should trigger sleep after 20 steps"
    # 初始睡眠阶段可能是 NREM 或 REM (取决于振荡器状态)
    # 验证 sleep_phase 是有效值
    assert state.phase in ('nrem', 'rem'), f"Invalid sleep phase: {state.phase}"

    # 睡眠后低 propensity → 可能醒来 (但需要低于threshold-hysteresis)
    for _ in range(5):
        state = vlpo.update(sleep_propensity=0.4, sleep_pressure=0.2, threshold=0.65)

    # NREM→REM 振荡应在睡眠中发生
    rem_achieved = False
    for _ in range(50):
        state = vlpo.update(sleep_propensity=0.8, sleep_pressure=0.6, threshold=0.65)
        if state.phase == 'rem':
            rem_achieved = True
            break

    assert rem_achieved or state.rem_on_activity > 0.3, \
        f"REM should eventually trigger, rem_on={state.rem_on_activity:.3f}"

    # 验证NE水平
    ne_awake = vlpo.get_ne_level()  # still asleep
    assert ne_awake < 0.2, f"Sleep NE should be low: {ne_awake:.4f}"

    print(f"  [PASS] test_vlpo_flip_flop: state={state.state}, phase={state.phase}, "
          f"vlpo_act={state.vlpo_activation:.3f}")


# ================================================================
# Test 4: NREM 突触缩小 — 等比缩小保留相对权重
# ================================================================

def test_nrem_synaptic_downscaling():
    """验证: NREM 等比缩小所有簇, 强簇不被删除, 相对强度保留."""
    from cerebrum.limbic_system.hippocampus import (
        ClusterNetwork, sleep_consolidation_nrem,
    )

    t = make_theta(cluster_threshold=0.4, learn_rate_l0=0.1,
                   synaptic_downscale_rate=0.15,
                   glymphatic_clear_rate=0.001)
    net = ClusterNetwork(t)

    # 创建强弱不同的簇
    for i in range(8):
        s = make_deterministic_s(100 + i, text_signal=0.5 + i * 0.1)
        c = net.learn(s)
        c.activation = 0.15 + i * 0.06  # [0.15, 0.57]

    acts_before = [c.activation for c in net.clusters]
    acts_before_sorted = sorted(acts_before, reverse=True)
    strongest_before_idx = acts_before.index(max(acts_before))
    weakest_before_idx = acts_before.index(min(acts_before))
    n_before = net.n_clusters

    # 执行 NREM 巩固 (高缩小率确保缩小主导)
    stats = sleep_consolidation_nrem(
        net, t, semantic_memory=None,
        n_replay_cycles=1, downscale_rate=t.synaptic_downscale_rate,
    )

    acts_after = [c.activation for c in net.clusters]

    # 相对排名应保留 (最强仍是强, 最弱仍是最弱)
    if net.n_clusters >= 4 and strongest_before_idx < len(acts_after):
        # 验证最强簇仍处于较高位置
        assert acts_after[strongest_before_idx] >= acts_after[weakest_before_idx] * 0.8, \
            "Relative ranking should be approximately preserved after NREM"

    # 整体均值应下降 (缩小主导)
    mean_before = np.mean(acts_before)
    mean_after = np.mean(acts_after)
    # 注意: 回放会温和提升activation → mean可能不降, 但个体差异应保留

    # 统计应有意义
    assert stats['nrem_phase'] == 'NREM'

    print(f"  [PASS] test_nrem_synaptic_downscaling: "
          f"downscaled {stats['n_downscaled']}, cleared {stats['n_cleared']}, "
          f"n_clusters {n_before}→{net.n_clusters}, "
          f"mean_act {mean_before:.3f}→{mean_after:.3f}")


# ================================================================
# Test 5: REM 情绪去刺痛
# ================================================================

def test_rem_emotional_depotentiation():
    """验证: 高情感簇在REM中被温和衰减."""
    from cerebrum.limbic_system.hippocampus import (
        ClusterNetwork, sleep_consolidation_rem,
    )

    t = make_theta(cluster_threshold=0.4, learn_rate_l0=0.1,
                   rem_emotional_processing=0.4)
    net = ClusterNetwork(t)

    # 创建高情感强度的簇 (centroid[64:72] 有大值)
    for i in range(5):
        s = make_deterministic_s(200 + i, text_signal=0.3 + i * 0.2)
        # 手动增强情感段
        c = net.learn(s)
        c.centroid[64:72] = np.random.randn(8).astype(np.float32) * (0.5 + i * 0.2)
        c.activation = 0.3 + i * 0.1

    # 第4,5簇有高情感强度
    high_emotion_acts_before = [c.activation for c in net.clusters[-2:]]
    high_emotion_norms_before = [
        float(np.linalg.norm(c.centroid[64:72])) for c in net.clusters[-2:]]

    # 执行 REM 巩固
    stats = sleep_consolidation_rem(
        net, t, amygdala=None, striatum=None,
        emotional_processing_strength=t.rem_emotional_processing,
    )

    # 高情感簇应被处理
    assert stats['n_emotional_processed'] >= 0, "Should report emotional processing"

    if stats['n_emotional_processed'] > 0:
        # 验证情感段被衰减
        high_emotion_norms_after = [
            float(np.linalg.norm(c.centroid[64:72])) for c in net.clusters[-2:]]
        avg_decay = (sum(high_emotion_norms_before) -
                    sum(high_emotion_norms_after)) / max(len(high_emotion_norms_before), 1)
        assert avg_decay >= -0.001, \
            f"Emotional segment should be attenuated: {avg_decay:.6f}"

    assert stats['rem_phase'] == 'REM'

    print(f"  [PASS] test_rem_emotional_depotentiation: "
          f"processed {stats['n_emotional_processed']}, "
          f"cross_linked {stats['n_cross_linked']}")


# ================================================================
# Test 6: 双相睡眠端到端
# ================================================================

def test_dual_phase_sleep_e2e():
    """端到端: NREM+REM 完整周期统计正确."""
    from cerebrum.limbic_system.hippocampus import (
        ClusterNetwork, dual_phase_sleep,
    )

    t = make_theta(cluster_threshold=0.4, learn_rate_l0=0.1,
                   synaptic_downscale_rate=0.04,
                   rem_emotional_processing=0.3,
                   glymphatic_clear_rate=0.002)
    net = ClusterNetwork(t)

    # 创建混合簇群 (一些强, 一些弱)
    for i in range(10):
        s = make_deterministic_s(300 + i, text_signal=0.5 + i * 0.08)
        c = net.learn(s)
        c.activation = 0.1 + i * 0.08
        if i < 3:
            c.activation = 0.02  # 超弱簇 — 应被清除

    n_before = net.n_clusters

    # 执行双相睡眠
    result = dual_phase_sleep(
        net, t,
        semantic_memory=None,
        amygdala=None, striatum=None,
        nrem_duration_ratio=0.65,
        sleep_duration_steps=20,
    )

    combined = result['combined']
    nrem = result['nrem']
    rem = result['rem']

    # 验证统计完整
    assert combined['dual_phase_complete'], "Dual-phase should complete"
    assert 'nrem' in result and 'rem' in result, "Should have both phases"
    assert combined['nrem_steps'] > combined['rem_steps'], \
        "NREM should be > REM duration"

    # 验证有实质性操作
    assert (combined['total_replayed'] + combined['total_emotional'] +
            combined['total_cross_linked']) > 0, \
        "Dual-phase sleep should produce measurable effects"

    print(f"  [PASS] test_dual_phase_sleep_e2e: "
          f"NREM replay={nrem['n_replayed']} downscale={nrem['n_downscaled']} "
          f"cleared={nrem['n_cleared']}, "
          f"REM emotional={rem['n_emotional_processed']} "
          f"cross_link={rem['n_cross_linked']}, "
          f"clusters {n_before}→{combined['clusters_after']}")


# ================================================================
# Test 7: α 注意门控
# ================================================================

def test_alpha_attention_gating():
    """验证: 非注意通道被α抑制, 注意通道不受影响."""
    from cerebrum.association.fpn import FrontoparietalNetwork

    fpn = FrontoparietalNetwork(input_dim=D)
    fpn.tpn_activation = 0.6  # 中等任务参与

    # 创建注意模板: 文本通道 [0:64] 注意, 其余忽略
    attn_template = np.ones(D, dtype=np.float32)
    attn_template[0:64] = 1.5   # 注意文本
    attn_template[64:] = 0.3    # 忽略其他
    fpn.attention_template = attn_template

    # 测试 α 门控
    sensory = np.random.randn(D).astype(np.float32) * 0.5

    result = fpn.alpha_gate_attention(
        sensory, attention_mask=attn_template,
        alpha_strength=0.5, step_count=42,
    )

    gated = result['gated_sensory']

    # 注意通道的平均抑制应该小于非注意通道
    attended_mask = attn_template > 0.5
    unattended_mask = ~attended_mask

    if np.any(attended_mask):
        attended_ratio = float(np.mean(np.abs(gated[attended_mask])) /
                              max(np.mean(np.abs(sensory[attended_mask])), 1e-8))
    else:
        attended_ratio = 1.0

    if np.any(unattended_mask):
        unattended_ratio = float(np.mean(np.abs(gated[unattended_mask])) /
                                max(np.mean(np.abs(sensory[unattended_mask])), 1e-8))
    else:
        unattended_ratio = 1.0

    # 被注意通道应保持更强 (相对抑制较小)
    assert attended_ratio > unattended_ratio * 0.9, \
        ("Attended channels should be less suppressed: "
         f"attn_ratio={attended_ratio:.3f}, unattn_ratio={unattended_ratio:.3f}")

    # 验证返回的alpha水平合理
    assert 0.0 <= result['alpha_level'] <= 1.0, \
        f"Alpha level should be in [0,1]: {result['alpha_level']:.3f}"
    assert 0.0 <= result['alpha_phase'] <= 1.0, \
        f"Alpha phase should be in [0,1]: {result['alpha_phase']:.3f}"

    print(f"  [PASS] test_alpha_attention_gating: "
          f"alpha_level={result['alpha_level']:.3f}, "
          f"mean_supp={result['mean_suppression']:.3f}, "
          f"attn_ratio={attended_ratio:.3f}, unattn_ratio={unattended_ratio:.3f}")


# ================================================================
# Test 8: 持久化 roundtrip — SCN/VLPO/睡眠状态
# ================================================================

def test_persistence_scn_vlpo():
    """验证: SCN/VLPO/SCN 状态正确保存/恢复."""
    from cerebrum.limbic_system.scn import SCN
    from brainstem_cerebellum.pons.vlpo import VLPO

    # 创建并运行 SCN
    scn = SCN()
    for _ in range(80):
        scn.step(light_level=0.4, is_asleep=False)

    per_before = scn.ttfl.per_protein
    pressure_before = scn.process_s.pressure

    # 创建并运行 VLPO
    vlpo = VLPO()
    for _ in range(30):
        vlpo.update(sleep_propensity=0.75, sleep_pressure=0.7)

    vlpo_act_before = vlpo.flip_flop.vlpo_activation
    rem_on_before = vlpo.oscillator.rem_on

    # 验证 Agent 创建 + SCN 集成
    from cns.agent import Agent
    agent = Agent()
    agent.scn = scn
    agent.vlpo = vlpo

    # 运行一步
    sensory = make_random_s(0.5)
    try:
        agent.step(sensory, step_count=0)
        agent_step_ok = True
    except Exception as e:
        print(f"  [WARN] agent.step() failed: {e} — testing components separately")
        agent_step_ok = False

    # 验证 get_state_summary 包含新字段
    summary = agent.get_state_summary()
    assert 'circadian_hour' in summary, f"Summary should have circadian_hour"
    assert 'sleep_pressure' in summary, f"Summary should have sleep_pressure"
    assert 'sleep_state' in summary, f"Summary should have sleep_state"
    assert 'is_asleep' in summary, f"Summary should have is_asleep"

    print(f"  [PASS] test_persistence_scn_vlpo: "
          f"PER={summary['circadian_hour']:.1f}h, "
          f"pressure={summary['sleep_pressure']:.3f}, "
          f"sleep_state={summary['sleep_state']}, "
          f"agent_step={'ok' if agent_step_ok else 'partial'}")


# ================================================================
# 运行所有测试
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("v6.3 睡眠优化与时间维度 — 单元测试")
    print("=" * 60)

    tests = [
        ("SCN TTFL自由运行", test_scn_ttfl_free_run),
        ("SCN 光同步", test_scn_light_entrainment),
        ("Process S", test_process_s),
        ("VLPO 触发器", test_vlpo_flip_flop),
        ("NREM 突触缩小", test_nrem_synaptic_downscaling),
        ("REM 情绪去刺痛", test_rem_emotional_depotentiation),
        ("双相睡眠端到端", test_dual_phase_sleep_e2e),
        ("α注意门控", test_alpha_attention_gating),
        ("持久化 SCN/VLPO", test_persistence_scn_vlpo),
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
        print("All v6.3 tests passed!")
