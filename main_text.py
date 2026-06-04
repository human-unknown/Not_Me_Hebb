"""
main_text.py —— Stage 2 文本环境主循环
自由能原理智能体

运行: python main_text.py [steps]
"""

import sys
import time
import numpy as np
from data_types import BodyVector
from text_interface import TextEnvironment
from agent import Agent

# 行动名称映射
A_NAMES = {0: 'NEXT', 1: 'PREV', 2: 'JUMP', 3: 'EXPR', 4: 'OBSV'}


def run_text_episode(steps: int = 500, verbose: bool = True,
                     log_interval: int = 50) -> dict:
    """运行文本环境 episode

    Agent 在语料中导航，通过 Hebb 集群学习文本模式。
    无外部奖励 — 纯 F 最小化驱动。
    """
    rng = np.random.default_rng(42)
    env = TextEnvironment()
    agent = Agent(rng=rng)
    agent.body = BodyVector(mode='text')

    rewards, actions, F_history = [], [], []

    t_start = time.perf_counter()

    for t in range(steps):
        # 感知
        s = env.get_sensory(body=agent.body.b)

        # 决策
        action = agent.step(s, t)
        agent.add_reward(0.0)

        # 执行 + 记录行动-后果
        env.step(action.index)
        s_next = env.get_sensory(body=agent.body.b)
        agent.record_action_consequence(s_next)

        rewards.append(0.0)
        actions.append(action.index)
        F_history.append(agent.F_history[-1] if agent.F_history else 0.0)

        if verbose and (t % log_interval == 0 or t == steps - 1):
            F_latest = agent.F_history[-1] if agent.F_history else 0.0
            body_str = f"b=[{agent.body.b[0]:.2f},{agent.body.b[1]:.2f},{agent.body.b[2]:.2f}]"
            text_preview = env.chunks[env.cursor][:40]

            print(f"[T={t:04d}] F={F_latest:.3f} | {body_str} | "
                  f"a={A_NAMES[action.index]} | C={agent.net.n_clusters} | "
                  f"text=\"{text_preview}...\"")

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n{'='*60}")
        print(f"Text Episode Complete | steps={steps} | {elapsed:.1f}s")
        print(f"  Clusters formed: {agent.net.n_clusters}")
        print(f"  Mean F:          {np.mean(agent.F_history):.3f}")
        print(f"  F trend:         {np.mean(F_history[:50]):.3f} -> "
              f"{np.mean(F_history[-50:]):.3f}")
        print(f"  Body final:      {agent.body.b}")

    return {
        'F_history': agent.F_history,
        'actions': actions,
        'n_clusters': agent.net.n_clusters,
        'elapsed': elapsed,
        'body_final': agent.body.b.copy(),
    }


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print("=" * 60)
    print("  自由能原理智能体 — Stage 2: 文本环境")
    print("  Free Energy Principle Agent — Text Navigation")
    print("=" * 60)
    print(f"  Steps: {steps}  |  Corpus: 50 sentences")
    print(f"  Embedding: sentence-transformer (384d → PCA 64d)")
    print(f"  Actions: NEXT/PREV/JUMP/EXPR/OBSV")
    print(f"  Reward: none (pure F-minimization)")
    print("-" * 60)

    result = run_text_episode(steps=steps)
