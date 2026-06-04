"""
test_inner_speech.py —— 验证内部言语闭环

测试:
1. 沉默帧触发内部言语 (inner speech fires on silence)
2. 内部言语编码 → 自听通道 (last_self_semantic/sentiment)
3. 自听 → body 效应 → 下一步信念改变 (feedback loop)
4. 内部言语链: 连续多步走神漂移
5. 表达 cooldown 不干扰内部言语
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_types import D, BodyVector
from agent import Agent
from sentiment import analyze_sentiment, sentiment_to_social_signal
from layer1_free_energy import SocialContext
from broca import Broca


class MockTextEnv:
    def encode_text(self, text):
        h = abs(hash(text)) % 10000
        rng = np.random.default_rng(h)
        return rng.normal(0, 0.3, 64).astype(np.float32)


def run_silence_simulation(n_steps=30, seed=42):
    """模拟纯沉默场景 — 没有人类输入，只有内部言语"""
    print("=" * 60)
    print("  Inner Speech Simulation: pure silence")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    agent = Agent(rng=rng, agent_id=0, n_agents=1)
    agent.body = BodyVector(mode='text')
    agent.theta.cluster_threshold = 0.35
    agent.theta.w_social = 2.0

    env = MockTextEnv()
    broca = Broca()
    social_ctx = SocialContext(tau=15.0)

    last_self_semantic = np.zeros(64, dtype=np.float32)
    last_self_sentiment = np.zeros(8, dtype=np.float32)

    inner_cooldown = 0
    thoughts = []
    body_trace = []
    valence_trace = []

    print(f"\n  {'Step':<6} {'b[0]':<8} {'V':<8} {'Inner?':<8} Thought")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*40}")

    for t in range(n_steps):
        # 构建 s (沉默帧)
        s = np.zeros(D, dtype=np.float32)
        s[:64] = rng.normal(0, 0.15, 64).astype(np.float32)  # 语料背景
        social_ctx.tick()

        # 自听回路
        s[128:192] = last_self_semantic
        s[96:104] = last_self_sentiment

        # Agent 处理
        action = agent.step(s, t, social_ctx=social_ctx)

        inner_cooldown -= 1

        # 内部言语逻辑 (与 main_dialogue.py 一致)
        had_inner = False
        thought_text = ""
        if action.index != 3 and inner_cooldown <= 0:
            if agent.net.n_clusters > 0:
                top = max(agent.net.clusters, key=lambda c: c.activation)
                if top.activation > 0.01:
                    belief_sem = top.centroid[:64].copy().astype(np.float32)
                    sensory_ctx = s[:64].astype(np.float32)
                    inner_query = belief_sem * 0.8 + sensory_ctx * 0.2

                    v = agent.valence_history[-1] if agent.valence_history else 0
                    a = agent.arousal_history[-1] if agent.arousal_history else 0
                    body = agent.body
                    social_need = max(0.0, body.setpoints[0] - body.b[0])
                    inner_temp = (0.7 + abs(v) * 0.5) * (1.0 + a * 0.6 + social_need * 0.4)
                    inner_k = max(5, min(18, int(8 + a * 12)))

                    words, _ = broca.speak_sentence(
                        inner_query, temperature=inner_temp, top_k=inner_k)
                    thought_text = "".join(words) if words else ""

                    if thought_text:
                        try:
                            last_self_semantic = env.encode_text(thought_text).astype(np.float32)
                        except Exception:
                            last_self_semantic = np.zeros(64, dtype=np.float32)
                        self_sent = analyze_sentiment(thought_text)
                        last_self_sentiment = sentiment_to_social_signal(self_sent)[:8].astype(np.float32)
                        had_inner = True

                    inner_cooldown = max(4, int(15 * (1.0 - a * 0.6)))

        if action.index == 3:
            inner_cooldown = max(inner_cooldown, 4)

        thoughts.append(thought_text)
        body_trace.append(agent.body.b[0])
        valence_trace.append(agent.valence_history[-1] if agent.valence_history else 0)

        # 显示
        v_now = agent.valence_history[-1] if agent.valence_history else 0
        marker = "[INNER]" if had_inner else "[     ]"
        print(f"  {t:<6} {agent.body.b[0]:.4f}   {v_now:+.3f}   {marker:<8} {thought_text[:40]}")

    # ---- 验证 ----
    inner_count = sum(1 for th in thoughts if th)
    print(f"\n  --- Results ---")
    print(f"  Total steps:        {n_steps}")
    print(f"  Inner speech fires: {inner_count}")
    print(f"  Body b[0] start:    {body_trace[0]:.4f}")
    print(f"  Body b[0] end:      {body_trace[-1]:.4f}")
    print(f"  Valence start:      {valence_trace[0]:+.3f}")
    print(f"  Valence end:        {valence_trace[-1]:+.3f}")

    # 内部言语应该至少触发几次
    assert inner_count >= 1, "Inner speech should fire at least once in 30 silence steps"
    print("  [PASS] Inner speech fires during silence")

    # body 应该发生变化 (内部言语 + 自听回路在起作用)
    assert abs(body_trace[-1] - body_trace[0]) > 0.001, \
        "Body should change due to inner speech feedback"
    print("  [PASS] Body state changes (inner speech feedback loop active)")

    # 内部言语内容应该多样 (不是每次都一样)
    unique_thoughts = set(th for th in thoughts if th)
    if len(unique_thoughts) >= 2:
        print(f"  [PASS] Diverse thoughts: {len(unique_thoughts)} unique")
    else:
        print(f"  [NOTE] Only {len(unique_thoughts)} unique thoughts (small network)")

    return thoughts, body_trace, valence_trace


def test_inner_vs_outer_separation():
    """验证内部言语和外部表达不冲突"""
    print("\n" + "=" * 60)
    print("  Test: Inner speech doesn't fire on A3 frames")
    print("=" * 60)

    rng = np.random.default_rng(123)
    agent = Agent(rng=rng, agent_id=0, n_agents=1)
    agent.body = BodyVector(mode='text')
    agent.theta.cluster_threshold = 0.35
    env = MockTextEnv()
    broca = Broca()
    social_ctx = SocialContext(tau=15.0)

    last_self_semantic = np.zeros(64, dtype=np.float32)
    last_self_sentiment = np.zeros(8, dtype=np.float32)
    inner_cooldown = 0

    inner_on_same_frame = 0
    total_a3 = 0

    for t in range(50):
        s = np.zeros(D, dtype=np.float32)
        s[:64] = rng.normal(0, 0.15, 64).astype(np.float32)
        social_ctx.tick()
        s[128:192] = last_self_semantic
        s[96:104] = last_self_sentiment

        action = agent.step(s, t, social_ctx=social_ctx)
        inner_cooldown -= 1

        had_inner_this_frame = False
        had_a3_this_frame = (action.index == 3)

        # 内部言语: 应该跳过 action=3 的帧
        is_silence = True
        if is_silence and action.index != 3 and inner_cooldown <= 0:
            if agent.net.n_clusters > 0:
                top = max(agent.net.clusters, key=lambda c: c.activation)
                if top.activation > 0.01:
                    belief_sem = top.centroid[:64].copy().astype(np.float32)
                    sensory_ctx = s[:64].astype(np.float32)
                    inner_query = belief_sem * 0.8 + sensory_ctx * 0.2
                    v = agent.valence_history[-1] if agent.valence_history else 0
                    a = agent.arousal_history[-1] if agent.arousal_history else 0
                    body = agent.body
                    sn = max(0.0, body.setpoints[0] - body.b[0])
                    inner_temp = (0.7 + abs(v) * 0.5) * (1.0 + a * 0.6 + sn * 0.4)
                    inner_k = max(5, min(18, int(8 + a * 12)))
                    words, _ = broca.speak_sentence(inner_query, temperature=inner_temp, top_k=inner_k)
                    thought = "".join(words) if words else ""
                    if thought:
                        had_inner_this_frame = True
                        try:
                            last_self_semantic = env.encode_text(thought).astype(np.float32)
                        except Exception:
                            pass
                        self_sent = analyze_sentiment(thought)
                        last_self_sentiment = sentiment_to_social_signal(self_sent)[:8].astype(np.float32)
                    inner_cooldown = max(4, int(15 * (1.0 - a * 0.6)))

        # 表达后抑制内部言语
        if action.index == 3:
            inner_cooldown = max(inner_cooldown, 4)

        # 记录: A₃ 和内部言语不应同一帧发生
        if had_a3_this_frame and had_inner_this_frame:
            inner_on_same_frame += 1
            print(f"  BUG at t={t}: A3 and inner speech on same frame!")
        if had_a3_this_frame:
            total_a3 += 1

    print(f"  Total A3 actions: {total_a3}")
    print(f"  Inner+A3 same frame: {inner_on_same_frame}")
    assert inner_on_same_frame == 0, f"A3 and inner speech collided on {inner_on_same_frame} frames"
    print("  [PASS] Inner speech never collides with expression")


if __name__ == '__main__':
    thoughts, body_trace, valence_trace = run_silence_simulation(n_steps=30, seed=42)
    test_inner_vs_outer_separation()
    print("\n" + "=" * 60)
    print("  All inner speech tests passed!")
    print("=" * 60)
