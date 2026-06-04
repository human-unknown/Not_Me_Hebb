"""
main_social3.py —— Stage 4: 三 agent 社会对话
自由能原理智能体

A 的 EXPR → B 的 s[80:88]
B 的 EXPR → C 的 s[80:88]
C 的 EXPR → A 的 s[80:88]
环形信号链
"""

import sys
import numpy as np
from data_types import D, BodyVector, ACTION_DIRECTIONS
from agent import Agent
from multimodal_interface import MultimodalReplay
from text_interface import TextEnvironment
from word_speech import get_speaker

A_NAMES = {0: 'NEXT', 1: 'PREV', 2: 'JUMP', 3: 'EXPR', 4: 'OBSV'}


def run_social_episode(steps=500, verbose=True, log_interval=50):
    rng = np.random.default_rng(42)

    agent_a = Agent(rng=rng, agent_id=0, n_agents=3)
    agent_b = Agent(rng=rng, agent_id=1, n_agents=3)
    agent_c = Agent(rng=rng, agent_id=2, n_agents=3)
    for a in [agent_a, agent_b, agent_c]:
        a.body = BodyVector(mode='text')
        a.theta.cluster_threshold = 0.55
        a.record_action_consequence = lambda s: None

    ACTION_DIRECTIONS[3] = [0.0, 0.0]
    ACTION_DIRECTIONS[4] = [0.0, 0.0]

    text_env = TextEnvironment()
    text_emb = text_env.embeddings
    sentences = text_env.chunks
    emb_64 = text_emb

    speaker = get_speaker()

    env_a = MultimodalReplay(text_embeddings=text_emb, n_frames=steps)
    env_b = MultimodalReplay(text_embeddings=text_emb, n_frames=steps)
    env_c = MultimodalReplay(text_embeddings=text_emb, n_frames=steps)
    env_b.cursor = 10
    env_c.cursor = 20

    # 环形信号: A→B→C→A
    sig_a = np.random.default_rng(42).normal(0, 0.1, 8)
    sig_b = np.random.default_rng(99).normal(0, 0.1, 8)
    sig_c = np.random.default_rng(77).normal(0, 0.1, 8)

    for t in range(steps):
        # Agent A — 接收 C 的信号
        s_a = env_a.get_sensory(body=agent_a.body)
        if np.sum(np.abs(s_a[0:64])) < 0.01:
            s_a[0:64] = text_emb[t % 5 if t < 100 else t % len(text_emb)]
        s_a[256:260] = 0.0
        s_a[80:88] = sig_c
        action_a = agent_a.step(s_a, t)
        if action_a.index == 3:
            c = agent_a.net.recall(s_a)
            if c is not None:
                path_centroids, path_indices = agent_a.net.diffuse(c, steps=3)
                sig_a = path_centroids[0][:8].copy()
                _, labels = speaker.speak_concept_path(path_centroids, output_path=f'audio_output/agent_A_t{t:04d}.wav')
                if verbose: print(f'  [A says]: {" ".join(labels)}')
        env_a.step(action_a.index)

        # Agent B — 接收 A 的信号
        s_b = env_b.get_sensory(body=agent_b.body)
        if np.sum(np.abs(s_b[0:64])) < 0.01:
            s_b[0:64] = text_emb[(t + 5) % 5 if t < 100 else (t + 5) % len(text_emb)]
        s_b[256:260] = 0.0
        s_b[80:88] = sig_a
        action_b = agent_b.step(s_b, t)
        if action_b.index == 3:
            c = agent_b.net.recall(s_b)
            if c is not None:
                path_centroids, path_indices = agent_b.net.diffuse(c, steps=3)
                sig_b = path_centroids[0][:8].copy()
                _, labels = speaker.speak_concept_path(path_centroids, output_path=f'audio_output/agent_B_t{t:04d}.wav')
                if verbose: print(f'  [B says]: {" ".join(labels)}')
        env_b.step(action_b.index)

        # Agent C — 接收 B 的信号
        s_c = env_c.get_sensory(body=agent_c.body)
        if np.sum(np.abs(s_c[0:64])) < 0.01:
            s_c[0:64] = text_emb[(t + 10) % 5 if t < 100 else (t + 10) % len(text_emb)]
        s_c[256:260] = 0.0
        s_c[80:88] = sig_b
        action_c = agent_c.step(s_c, t)
        if action_c.index == 3:
            c = agent_c.net.recall(s_c)
            if c is not None:
                path_centroids, path_indices = agent_c.net.diffuse(c, steps=3)
                sig_c = path_centroids[0][:8].copy()
                _, labels = speaker.speak_concept_path(path_centroids, output_path=f'audio_output/agent_C_t{t:04d}.wav')
                if verbose: print(f'  [C says]: {" ".join(labels)}')
        env_c.step(action_c.index)

        if verbose and (t % log_interval == 0 or t == steps - 1):
            Fa = agent_a.F_history[-1] if agent_a.F_history else 0
            Fb = agent_b.F_history[-1] if agent_b.F_history else 0
            Fc = agent_c.F_history[-1] if agent_c.F_history else 0
            print(f"[T={t:04d}] "
                  f"A: F={Fa:.3f} b0={agent_a.body.b[0]:.2f} C={agent_a.net.n_clusters} | "
                  f"B: F={Fb:.3f} b0={agent_b.body.b[0]:.2f} C={agent_b.net.n_clusters} | "
                  f"C: F={Fc:.3f} b0={agent_c.body.b[0]:.2f} C={agent_c.net.n_clusters}")

    print(f"\nEXPR: A={sum(1 for a in agent_a.action_history if a==3)} "
          f"B={sum(1 for a in agent_b.action_history if a==3)} "
          f"C={sum(1 for a in agent_c.action_history if a==3)}")


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print("=" * 60)
    print("  自由能原理智能体 — Stage 4: 三 agent 社会对话")
    print("=" * 60)
    print(f"  Steps: {steps} | A->B->C->A 信号环")
    print("-" * 60)
    run_social_episode(steps=steps)
