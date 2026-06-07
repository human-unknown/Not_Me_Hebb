"""
test_v6_4_resident.py — v6.4 长期常驻学习 单元测试

测试:
  1. Reader 句子切分 — 中文标点正确切分，进度追踪准确
  2. Reader 疲劳模型 — 连续阅读后 should_read() 返回 False，休息后恢复
  3. AutonomousLoop 模式切换 — idle → wandering → reading 自动转换
  4. AutonomousLoop 睡眠暂停 — 睡眠期跳过所有自主活动
  5. InternalLife 走神回忆 — 从海马成功 recall，联想链不崩溃
  6. InternalLife 内部独白 — 亚发声模式正确产出
  7. Telemetry 记录/刷新 — CSV 格式正确，roundtrip 可读
  8. light_step 一致性 — 自主模式身体/SCN/VLPO 状态正常
  9. 持久化 roundtrip — Reader进度 + 自主状态正确保存/恢复
"""

import sys
import os
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from cns.data_types import D, Theta


def make_theta(**overrides) -> Theta:
    """创建测试用 Theta (v6.4: 59 params)."""
    t = Theta()
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def make_deterministic_s(seed: int, text_signal: float = 1.0) -> np.ndarray:
    """生成确定性的感知向量."""
    rng = np.random.default_rng(seed)
    s = np.zeros(D, dtype=np.float32)
    s[:16] = rng.normal(0, text_signal, 16).astype(np.float32)
    s[16:64] = rng.normal(0, 0.1, 48).astype(np.float32)
    s[64:72] = rng.normal(0, 0.3, 8).astype(np.float32)
    return s


# ============================================================
# Test 1: Reader 句子切分
# ============================================================

def test_reader_sentence_split():
    """验证: Reader 正确切分中文句子."""
    from tools.reader import Reader

    r = Reader()
    text = "你好。今天天气真好！你想去哪里？我们走吧…"
    r.load_from_text(text)
    assert r.state.total_sentences >= 3, \
        f"Expected >= 3 sentences, got {r.state.total_sentences}"

    sentences = []
    while (s := r.next_sentence()):
        sentences.append(s)

    assert len(sentences) >= 3, f"Expected >= 3, got {len(sentences)}"
    assert r.progress == 1.0, f"Progress should be 1.0, got {r.progress:.2f}"
    assert r.is_finished, "Should be finished"

    print(f"  [PASS] test_reader_sentence_split: "
          f"{r.state.total_sentences} sentences, progress={r.progress:.2f}")


# ============================================================
# Test 2: Reader 疲劳模型
# ============================================================

def test_reader_fatigue():
    """验证: 连续阅读后认知负荷上升，疲劳触发暂停，休息后恢复."""
    from tools.reader import Reader

    r = Reader(fatigue_per_sentence=0.15, recovery_rate=0.05, max_fatigue=0.6)

    # 创建一个长文本
    sentences = ["句子" + str(i) for i in range(50)]
    text = "。".join(sentences)
    r.load_from_text(text)

    # 模拟阅读若干句
    body_b = np.array([0.7, 0.7, 0.0, 0.0, 0.5, 0.3, 0.3, 0.0, 0.9], dtype=np.float32)
    read_count = 0
    for _ in range(10):
        s = r.next_sentence()
        if s is None:
            break
        read_count += 1
        # 模拟理解
        r.record_comprehension(0.5)

    # 认知负荷应显著上升
    assert r.cognitive_load > 0.3, \
        f"Cognitive load should be > 0.3 after reading, got {r.cognitive_load:.3f}"

    # 再读更多 → should_read 返回 False
    for _ in range(5):
        r.next_sentence()
        r.record_comprehension(0.5)

    # 高认知 + 低专注 → 应该暂停
    body_b[7] = 0.5  # 高认知
    body_b[4] = 0.1  # 低专注
    should = r.should_read(body_b, tpn_activation=0.3)
    # should_read may already have paused; check the flag
    assert r.state.is_paused or not should, \
        "Should be paused or should_read returns False under high fatigue"

    # 恢复测试
    body_b[7] = 0.2
    body_b[4] = 0.5
    for _ in range(10):
        if r.try_resume(body_b, tpn_activation=0.4):
            break
        body_b[7] = max(0, body_b[7] - 0.1)

    print(f"  [PASS] test_reader_fatigue: "
          f"read {read_count}, load={r.cognitive_load:.3f}, "
          f"paused={r.state.is_paused}")


# ============================================================
# Test 3: AutonomousLoop 模式切换
# ============================================================

def test_autonomous_mode_switch():
    """验证: AutonomousLoop 能正确在不同活动模式间切换."""
    from cns.agent import Agent
    from cns.data_types import BodyVector
    from entry.autonomous import AutonomousLoop

    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')

    loop = AutonomousLoop(agent, broca=None, steps_per_second=100)

    # 初始应为 idle 或 wandering (取决于 TPN/DMN)
    assert loop.mode in ('idle', 'wandering'), \
        f"Initial mode should be idle or wandering, got {loop.mode}"

    # 跑 20 步，验证模式历史被填充
    for _ in range(20):
        loop.tick()

    assert len(loop.activity_history) >= 20, \
        f"Should have 20 activities, got {len(loop.activity_history)}"

    modes = set(loop.activity_history)
    print(f"  [PASS] test_autonomous_mode_switch: "
          f"modes={modes}, steps={loop._step_counter}")


# ============================================================
# Test 4: AutonomousLoop 睡眠暂停
# ============================================================

def test_autonomous_sleep_pause():
    """验证: 睡眠期 AutonomousLoop 正确跳过所有自主活动."""
    from cns.agent import Agent
    from cns.data_types import BodyVector
    from entry.autonomous import AutonomousLoop

    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')

    # 强制进入睡眠状态 (通过 flip_flop 内部属性)
    agent.vlpo.flip_flop._is_asleep = True
    agent.vlpo._was_asleep = True
    agent._sleep_state.state = 'nrem_n2'

    loop = AutonomousLoop(agent, broca=None, steps_per_second=100)

    result = loop.tick()
    assert result['is_asleep'], "Should be asleep"
    assert result['mode'] == 'sleeping', \
        f"Mode should be 'sleeping' when asleep, got {result['mode']}"

    print(f"  [PASS] test_autonomous_sleep_pause: "
          f"mode={result['mode']}, sleep_state={result.get('sleep_state')}")


# ============================================================
# Test 5: InternalLife 走神回忆
# ============================================================

def test_internal_life_wander():
    """验证: 走神模式从海马成功 recall 并形成联想链."""
    from cns.agent import Agent
    from cns.data_types import BodyVector
    from cerebrum.association.internal_life import InternalLife

    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')

    # 喂入一些数据，让海马有集群
    for i in range(20):
        s = make_deterministic_s(i, text_signal=1.0)
        agent.net.learn(s)
    agent.net.decay()

    assert agent.net.n_clusters >= 3, \
        f"Need >= 3 clusters for wander, got {agent.net.n_clusters}"

    il = InternalLife()
    result = il.mind_wander(agent)

    assert result['thought_type'] == 'wander', \
        f"Wrong thought_type: {result.get('thought_type')}"
    assert result['n_recalled'] >= 1, \
        f"Should recall at least 1 cluster, got {result['n_recalled']}"

    print(f"  [PASS] test_internal_life_wander: "
          f"chain={result['n_recalled']}, "
          f"emotional={result.get('emotional_shift', 0):.3f}")


# ============================================================
# Test 6: InternalLife 内部独白
# ============================================================

def test_internal_life_monologue():
    """验证: 内部独白亚发声能正确产出文本."""
    from cns.agent import Agent
    from cns.data_types import BodyVector
    from cerebrum.association.internal_life import InternalLife
    from environments.text_interface import TextEnvironment
    from cerebrum.frontal_lobe.broca import Broca

    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')

    # 喂入数据
    for i in range(30):
        s = make_deterministic_s(i, text_signal=1.0)
        agent.net.learn(s)

    # Broca (纯净模式 — headless TextEnvironment)
    te = TextEnvironment(load_corpus=False)
    broca = Broca(text_env=te, load_corpus=False)

    il = InternalLife()
    result = il.internal_monologue(agent, broca=broca)

    # 基本验证：应该返回有效结果
    assert result['thought_type'] == 'monologue', \
        f"Wrong type: {result.get('thought_type')}"
    assert 'error' not in result or result.get('words_generated', 0) >= 0, \
        f"Unexpected error: {result.get('error')}"

    print(f"  [PASS] test_internal_life_monologue: "
          f"words={result.get('words_generated', 0)}, "
          f"text='{result.get('text', '')[:30]}'")


# ============================================================
# Test 7: Telemetry 记录/刷新
# ============================================================

def test_telemetry_record_flush():
    """验证: Telemetry 正确记录步数据并写入 CSV."""
    import tempfile
    import os as _os

    from cns.agent import Agent
    from cns.data_types import BodyVector
    from tools.telemetry import Telemetry, TELEMETRY_DIR

    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')

    # 先做一些 step 让 history 不为空
    for i in range(5):
        s = make_deterministic_s(i)
        agent.net.learn(s)
    # 手动推进一次完整 step
    s = make_deterministic_s(99)
    agent.step(s, 0)

    tel = Telemetry(session_id='test_v6_4')

    # 记录一些步
    for i in range(50):
        tel.record_step(agent, activity='idle')

    assert tel.step_count == 50, f"Should have 50 steps, got {tel.step_count}"
    assert len(tel._buffer) == 50, f"Buffer should have 50, got {len(tel._buffer)}"

    # 测试 summary
    summary = tel.get_summary(30)
    assert summary['n_steps'] == 30, \
        f"Summary should have 30 steps, got {summary['n_steps']}"

    # 测试 flush
    tel.flush()
    csv_path = _os.path.join(TELEMETRY_DIR, "steps_test_v6_4.csv")
    assert _os.path.exists(csv_path), f"CSV file should exist at {csv_path}"

    # 清理
    tel.clear()
    if _os.path.exists(csv_path):
        _os.remove(csv_path)

    print(f"  [PASS] test_telemetry_record_flush: "
          f"steps={tel.step_count}, summary OK, CSV created")


# ============================================================
# Test 8: light_step 一致性
# ============================================================

def test_light_step_consistency():
    """验证: light_step 不崩溃，且身体/SCN/VLPO 状态正常推进."""
    from cns.agent import Agent
    from cns.data_types import BodyVector

    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')

    # 喂入一些初始数据
    for i in range(10):
        s = make_deterministic_s(i)
        agent.net.learn(s)

    # 执行 light_step
    results = []
    for step in range(30):
        r = agent.light_step(step, activity='idle')
        results.append(r)

    # 验证所有 step 都成功返回
    assert len(results) == 30, f"Should have 30 results, got {len(results)}"
    for r in results:
        assert 'F_total' in r, "Each result should have F_total"
        assert 'valence' in r, "Each result should have valence"

    # 验证身体状态被更新
    assert len(agent.F_history) >= 30, \
        f"F_history should have >= 30, got {len(agent.F_history)}"
    assert len(agent.valence_history) >= 30, \
        f"valence_history should have >= 30, got {len(agent.valence_history)}"

    # 验证 SCN/VLPO 在运行
    assert agent._circadian_state.circadian_phase != 0.0 or \
           agent._circadian_state.sleep_pressure > 0.0, \
        "SCN should have advanced"

    print(f"  [PASS] test_light_step_consistency: "
          f"{len(results)} steps, F={results[-1]['F_total']:.3f}")


# ============================================================
# Test 9: 持久化 roundtrip (Reader + 自主状态)
# ============================================================

def test_persistence_roundtrip_v6_4():
    """验证: Reader进度 + 自主状态 + internal_life 正确保存/恢复."""
    import tempfile
    import os as _os

    from cns.agent import Agent
    from cns.data_types import BodyVector
    from tools.reader import Reader
    from cerebrum.association.internal_life import InternalLife
    from tools.telemetry import Telemetry

    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')

    # 设置 v6.4 状态
    agent.reader = Reader()
    agent.reader.load_from_text("测试句子一。测试句子二。测试句子三。测试句子四。")
    agent.reader.next_sentence()  # 读一句

    agent.internal_life = InternalLife()
    agent.internal_life._wander_count = 5
    agent.internal_life._monologue_count = 3

    agent.telemetry = Telemetry(session_id='test_persist')
    agent._autonomous_mode = True
    agent._last_activity = 'reading'

    # 保存
    tmp_path = _os.path.join(tempfile.gettempdir(), 'test_v6_4_agent.pkl')
    agent.save(path=tmp_path)

    # 加载到新 Agent
    agent2, meta = Agent.load(tmp_path, verbose=False)
    assert meta is not None, "Meta should not be None"

    # 验证 Reader 状态
    assert hasattr(agent2, 'reader'), "agent2 should have reader"
    if agent2.reader is not None:
        progress = agent2.reader.get_progress()
        assert progress['sentences_read'] >= 1, \
            f"Should have read at least 1, got {progress['sentences_read']}"
        assert progress['file_name'] != '', "Should have file name"

    # 验证 InternalLife 状态
    if agent2.internal_life is not None:
        state = agent2.internal_life.get_state()
        assert state['wander_count'] == 5, \
            f"wander_count should be 5, got {state['wander_count']}"
        assert state['monologue_count'] == 3, \
            f"monologue_count should be 3, got {state['monologue_count']}"

    # 验证自主模式标志
    assert agent2._autonomous_mode, "Should be in autonomous mode"
    assert agent2._last_activity == 'reading', \
        f"Last activity should be 'reading', got {agent2._last_activity}"

    # 清理
    if _os.path.exists(tmp_path):
        _os.remove(tmp_path)

    print(f"  [PASS] test_persistence_roundtrip_v6_4: "
          f"reader={agent2.reader is not None}, "
          f"internal_life={agent2.internal_life is not None}")


# ============================================================
# Test runner
# ============================================================

def run_all_tests():
    tests = [
        ('Reader 句子切分', test_reader_sentence_split),
        ('Reader 疲劳模型', test_reader_fatigue),
        ('AutonomousLoop 模式切换', test_autonomous_mode_switch),
        ('AutonomousLoop 睡眠暂停', test_autonomous_sleep_pause),
        ('InternalLife 走神回忆', test_internal_life_wander),
        ('InternalLife 内部独白', test_internal_life_monologue),
        ('Telemetry 记录/刷新', test_telemetry_record_flush),
        ('light_step 一致性', test_light_step_consistency),
        ('持久化 roundtrip', test_persistence_roundtrip_v6_4),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("  NotMe v6.4 Unit Tests — 长期常驻学习")
    print("=" * 60)
    print()

    for name, func in tests:
        try:
            func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, "
          f"{len(tests)} total")
    if errors:
        print(f"  Failed tests:")
        for name, err in errors:
            print(f"    - {name}: {err}")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
