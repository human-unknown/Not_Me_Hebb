"""
main_social.py —— Stage 4: 双 agent 社会对话
自由能原理智能体

Agent A 的 A3(表达) → Agent B 的 s[80:88] (社会通道)
Agent B 的 A3(表达) → Agent A 的 s[80:88]
F_social = 预测对方输出 vs 实际对方输出的误差
"""

import sys
import numpy as np
from data_types import D, BodyVector, ACTION_DIRECTIONS
from agent import Agent
from multimodal_interface import MultimodalReplay
from text_interface import TextEnvironment

A_NAMES = {0: 'NEXT', 1: 'PREV', 2: 'JUMP', 3: 'EXPR', 4: 'OBSV'}


def run_social_episode(steps: int = 500, verbose: bool = True,
                       log_interval: int = 50) -> dict:
    rng = np.random.default_rng(42)

    # 两个 agent，共享语料但独立游标
    agent_a = Agent(rng=rng, agent_id=0, n_agents=2)
    agent_b = Agent(rng=rng, agent_id=1, n_agents=2)
    agent_a.body = BodyVector(mode='text')
    agent_b.body = BodyVector(mode='text')
    agent_a.theta.cluster_threshold = 0.55
    agent_b.theta.cluster_threshold = 0.55
    agent_a.record_action_consequence = lambda s: None
    agent_b.record_action_consequence = lambda s: None
    # 文本模式: A3/A₄ 无位移
    ACTION_DIRECTIONS[3] = [0.0, 0.0]
    ACTION_DIRECTIONS[4] = [0.0, 0.0]

    text_env = TextEnvironment()
    text_emb = text_env.embeddings

    env_a = MultimodalReplay(text_embeddings=text_emb, n_frames=steps)
    env_b = MultimodalReplay(text_embeddings=text_emb, n_frames=steps)
    env_b.cursor = 10  # B 从不同位置开始

    # 社会信号缓冲区 (初始小随机值打破对称)
    social_a_to_b = np.random.default_rng(42).normal(0, 0.1, 8)
    social_b_to_a = np.random.default_rng(99).normal(0, 0.1, 8)

    for t in range(steps):
        # ===== Agent A =====
        s_a = env_a.get_sensory(body=agent_a.body)
        if np.sum(np.abs(s_a[0:64])) < 0.01:
            s_a[0:64] = text_emb[t % 5 if t < 100 else t % len(text_emb)]
        s_a[256:260] = 0.0
        s_a[80:88] = social_b_to_a  # B 的上一轮 A3 → A 的社会通道

        action_a = agent_a.step(s_a, t)

        if action_a.index == 3:  # A3 → 输出给 B
            c = agent_a.net.recall(s_a)
            if c is not None:
                text_out = agent_a.net.diffuse(c, steps=3)
                social_a_to_b = text_out[:8].copy()
        env_a.step(action_a.index)

        # ===== Agent B =====
        s_b = env_b.get_sensory(body=agent_b.body)
        if np.sum(np.abs(s_b[0:64])) < 0.01:
            s_b[0:64] = text_emb[(t + 5) % 5 if t < 100 else (t + 5) % len(text_emb)]
        s_b[256:260] = 0.0
        s_b[80:88] = social_a_to_b  # A 的 A3 → B 的社会通道

        action_b = agent_b.step(s_b, t)

        if action_b.index == 3:  # A3 → 输出给 A
            c = agent_b.net.recall(s_b)
            if c is not None:
                text_out = agent_b.net.diffuse(c, steps=3)
                social_b_to_a = text_out[:8].copy()
        env_b.step(action_b.index)

        # 日志
        if verbose and (t % log_interval == 0 or t == steps - 1):
            Fa = agent_a.F_history[-1] if agent_a.F_history else 0
            Fb = agent_b.F_history[-1] if agent_b.F_history else 0
            Fsa = agent_a.F_social_history[-1] if agent_a.F_social_history else 0
            Fsb = agent_b.F_social_history[-1] if agent_b.F_social_history else 0
            b_a = agent_a.body.b
            b_b = agent_b.body.b
            print(f"[T={t:04d}] "
                  f"A: F={Fa:.3f} Fs={Fsa:.3f} b0={b_a[0]:.2f} C={agent_a.net.n_clusters} | "
                  f"B: F={Fb:.3f} Fs={Fsb:.3f} b0={b_b[0]:.2f} C={agent_b.net.n_clusters}")

    Fa = np.mean(agent_a.F_social_history[-100:]) if agent_a.F_social_history else 0
    Fb = np.mean(agent_b.F_social_history[-100:]) if agent_b.F_social_history else 0
    print(f"\nSocial F: A={Fa:.3f} B={Fb:.3f}")
    print(f"A EXPR count: {sum(1 for a in agent_a.action_history if a==3)}")
    print(f"B EXPR count: {sum(1 for a in agent_b.action_history if a==3)}")

    return {
        'F_social_a': Fa, 'F_social_b': Fb,
        'clusters_a': agent_a.net.n_clusters,
        'clusters_b': agent_b.net.n_clusters,
    }


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print("=" * 60)
    print("  自由能原理智能体 — Stage 4: 双 agent 社会对话")
    print("  Free Energy Principle Agent — Social Dialogue")
    print("=" * 60)
    print(f"  Steps: {steps}")
    print(f"  A3(EXPR) → s[80:88] of other agent")
    print(f"  F_social = pred_err(what they said, what I expected)")
    print("-" * 60)

    run_social_episode(steps=steps, verbose=True)
