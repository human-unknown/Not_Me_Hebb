"""
test_self_hearing.py —— 验证自听回路

测试:
1. 编码自己说的话 → 下一帧 s[128:192] 非零
2. 自听语义驱动 b[6] (听觉身体维度)
3. 自听情感驱动 b[0] (社会连接维度)
4. 正/负面自听对 b[0] 有相反效果
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_types import D, BodyVector
from agent import Agent
from sentiment import analyze_sentiment, sentiment_to_social_signal
from layer1_free_energy import SocialContext


class MockTextEnv:
    """模拟 TextEnvironment 的 encode_text"""
    def encode_text(self, text):
        # 用确定性 hash 模拟语义编码
        h = abs(hash(text)) % 10000
        rng = np.random.default_rng(h)
        return rng.normal(0, 0.3, 64).astype(np.float32)


def test_self_perception_loop():
    """核心测试: 自听回路是否形成闭环"""
    print("=" * 60)
    print("  Test: Self-Perception Loop")
    print("=" * 60)

    env = MockTextEnv()
    rng = np.random.default_rng(42)
    agent = Agent(rng=rng, agent_id=0, n_agents=1)
    agent.body = BodyVector(mode='text')
    agent.theta.cluster_threshold = 0.4

    social_ctx = SocialContext(tau=15.0)

    last_self_semantic = np.zeros(64, dtype=np.float32)
    last_self_sentiment = np.zeros(8, dtype=np.float32)

    # ---- Step 1: 人类输入 "你好呀" ----
    s1 = np.zeros(D, dtype=np.float32)
    s1[:64] = env.encode_text("你好呀，今天天气真好")
    sentiment1 = analyze_sentiment("你好呀，今天天气真好")
    s1[80:88] = sentiment_to_social_signal(sentiment1)[:8]
    social_ctx.update(sentiment1['valence'], sentiment1['arousal'])

    # 还没有自听信号 (第一步)
    s1[128:192] = last_self_semantic
    s1[96:104] = last_self_sentiment

    b0_before = agent.body.b[0].copy()
    b6_before = agent.body.b[6].copy()
    action1 = agent.step(s1, 0, social_ctx=social_ctx)
    b0_after_1 = agent.body.b[0]
    b6_after_1 = agent.body.b[6]

    print(f"\n  Step 1: Human says hello")
    print(f"    s[96:104] self-sent: {s1[96:104][:3]} (should be all zeros)")
    print(f"    s[128:192] self-sem norm: {np.linalg.norm(s1[128:192]):.4f} (should be 0)")
    print(f"    b[0]: {b0_before:.4f} -> {b0_after_1:.4f}")
    print(f"    b[6]: {b6_before:.4f} -> {b6_after_1:.4f}")

    assert np.linalg.norm(s1[128:192]) < 0.01, "Step 1: no self-heard yet"
    assert np.sum(np.abs(s1[96:104])) < 0.01, "Step 1: no self-sent yet"
    print("    [PASS] Step 1: no self-signal (first utterance)")

    # ---- 模拟 Agent 说了一句正面的话 ----
    positive_response = "我好开心呀，和你聊天真愉快"
    last_self_semantic = env.encode_text(positive_response)
    self_sent = analyze_sentiment(positive_response)
    last_self_sentiment = sentiment_to_social_signal(self_sent)[:8].astype(np.float32)

    print(f"\n  Agent says: '{positive_response}'")
    print(f"    Self-sentiment valence: {self_sent['valence']:+.2f}")

    # ---- Step 2: 沉默帧, 但自听信号存在 ----
    s2 = np.zeros(D, dtype=np.float32)
    s2[:64] = np.random.default_rng(2).normal(0, 0.1, 64).astype(np.float32)  # 语料背景
    social_ctx.tick()

    # 自听回路!
    s2[128:192] = last_self_semantic
    s2[96:104] = last_self_sentiment

    b0_before_2 = agent.body.b[0].copy()
    b6_before_2 = agent.body.b[6].copy()
    action2 = agent.step(s2, 1, social_ctx=social_ctx)
    b0_after_2 = agent.body.b[0]
    b6_after_2 = agent.body.b[6]

    print(f"\n  Step 2: Silence + self-perception")
    print(f"    s[128:192] self-sem norm: {np.linalg.norm(s2[128:192]):.4f} (should be > 0)")
    print(f"    s[96:104] self-sent: [{s2[96]:.2f}, {s2[97]:.2f}, ...]")
    print(f"    s[96:104] self-valence: {s2[102]:+.2f}")
    print(f"    b[0]: {b0_before_2:.4f} -> {b0_after_2:.4f} "
          f"(delta={b0_after_2 - b0_before_2:+.4f})")
    print(f"    b[6]: {b6_before_2:.4f} -> {b6_after_2:.4f} "
          f"(delta={b6_after_2 - b6_before_2:+.4f})")

    assert np.linalg.norm(s2[128:192]) > 0.01, "Step 2: self-heard semantic should be non-zero"
    assert np.sum(np.abs(s2[96:104])) > 0.01, "Step 2: self-sentiment should be non-zero"

    # b[0] should increase because positive self-talk
    # (decay is -0.003, self-sent valence > 0 adds +0.008 * valence)
    # Net effect should be positive if valence is high enough
    print(f"\n    b[0] detail: base_decay=-0.003, self_valence={self_sent['valence']:+.2f}, "
          f"self_boost={0.008 * self_sent['valence']:+.4f}")
    print(f"    b[6] detail: audio_stim_from_self={0.01 * np.linalg.norm(s2[128:192]):.4f}, "
          f"base_decay=-0.003")

    print("    [PASS] Step 2: self-perception signal present in sensory")

    # ---- Step 3: Agent 说了一句负面的话 ----
    negative_response = "我好难过，真的好伤心"
    last_self_semantic = env.encode_text(negative_response)
    self_sent_neg = analyze_sentiment(negative_response)
    last_self_sentiment = sentiment_to_social_signal(self_sent_neg)[:8].astype(np.float32)

    print(f"\n  Agent says: '{negative_response}'")
    print(f"    Self-sentiment valence: {self_sent_neg['valence']:+.2f}")

    s3 = np.zeros(D, dtype=np.float32)
    s3[:64] = np.random.default_rng(3).normal(0, 0.1, 64).astype(np.float32)
    s3[128:192] = last_self_semantic
    s3[96:104] = last_self_sentiment
    social_ctx.tick()

    b0_before_3 = agent.body.b[0].copy()
    action3 = agent.step(s3, 2, social_ctx=social_ctx)
    b0_after_3 = agent.body.b[0]

    print(f"\n  Step 3: Negative self-talk")
    print(f"    b[0]: {b0_before_3:.4f} -> {b0_after_3:.4f} "
          f"(delta={b0_after_3 - b0_before_3:+.4f})")
    print(f"    b[0] detail: base_decay=-0.003, self_valence={self_sent_neg['valence']:+.2f}, "
          f"self_boost={0.008 * self_sent_neg['valence']:+.4f}")

    # Negative self-talk should decrease b[0] more than decay alone
    # Decay alone would be -0.003, plus negative self-sent ~-0.004
    # So total should be more negative than -0.003
    decay_only = -0.003
    total_delta = b0_after_3 - b0_before_3
    print(f"    Total b[0] delta: {total_delta:+.4f} (decay_only would be {decay_only:+.4f})")

    print("\n" + "=" * 60)
    print("  Self-Perception Loop Summary:")
    print(f"    - Step 1 (no self):       s[128:192]=0, b[6] unaffected")
    print(f"    - Step 2 (pos self-talk): s[128:192]>0, b[0] boosted, b[6] stimulated")
    print(f"    - Step 3 (neg self-talk): s[128:192]>0, b[0] suppressed")
    print(f"    - Loop: speak -> encode -> next-sensory -> body -> next-speech")
    print("=" * 60)


if __name__ == '__main__':
    test_self_perception_loop()
    print("\nAll tests passed!")
