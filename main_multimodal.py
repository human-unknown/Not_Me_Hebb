"""
main_multimodal.py —— Stage 3A 多模态离线回放
自由能原理智能体

运行: python main_multimodal.py [steps]
"""

import sys
import numpy as np
from data_types import BodyVector, ACTION_DIRECTIONS
from agent import Agent
from multimodal_interface import MultimodalReplay

A_NAMES = {0: 'NEXT', 1: 'PREV', 2: 'JUMP', 3: 'EXPR', 4: 'OBSV'}


def run_multimodal_episode(steps: int = 500, verbose: bool = True,
                           log_interval: int = 50,
                           realtime: bool = False) -> dict:
    rng = np.random.default_rng(42)
    agent = Agent(rng=rng)
    agent.body = BodyVector(mode='text')
    agent.theta.cluster_threshold = 0.55
    # 文本模式: A₃(EXPR)和 A₄(OBSV)无空间位移
    ACTION_DIRECTIONS[3] = [0.0, 0.0]
    ACTION_DIRECTIONS[4] = [0.0, 0.0]

    # 复用 Stage 2 文本嵌入
    from text_interface import TextEnvironment
    text_env = TextEnvironment()
    text_emb = text_env.embeddings  # (N, 64)

    env = MultimodalReplay(text_embeddings=text_emb,
                           body=agent.body, n_frames=steps,
                           realtime=realtime)

    actions = []
    for t in range(steps):
        s = env.get_sensory(body=agent.body)
        # 如果 log 中文本通道为 0，从 TextEnvironment 覆盖
        if np.sum(np.abs(s[0:64])) < 0.01:
            if t < 100:
                s[0:64] = text_emb[(t // 3) % 5]   # 预热: 每 3 步换一句话
            else:
                s[0:64] = text_emb[(t // 2) % len(text_emb)]  # 每 2 步换一句
        # 时间相位归零: 避免跨帧相似值抬高余弦相似度
        s[256:260] = 0.0
        action = agent.step(s, t)
        agent.add_reward(0.0)
        env.step(action.index)
        actions.append(action.index)

        # A₃ 输出: 用集群 centroid 编码反馈
        if action.index == 3:
            c = agent.net.recall(s)
            if c is not None:
                env.last_output_embedding = c.centroid[:64]
                if verbose:
                    # 找 centroid[:64] 最接近的语料句子
                    sims = np.dot(text_emb, c.centroid[:64]) / (
                        np.linalg.norm(text_emb, axis=1) * np.linalg.norm(c.centroid[:64]) + 1e-8)
                    top = int(np.argmax(sims))
                    print(f"  A3 OUTPUT: '{text_env.chunks[top][:50]}...'")

        s_next = env.get_sensory(body=agent.body)
        agent.record_action_consequence(s_next)

        if verbose and (t % log_interval == 0 or t == steps - 1):
            F_latest = agent.F_history[-1] if agent.F_history else 0.0
            b = agent.body.b
            body_str = f"b=[b0:{b[0]:.2f} b2:{b[2]:.2f} b5:{b[5]:.2f} b6:{b[6]:.2f}]" if len(b) >= 7 else f"b=[{b[0]:.2f},{b[2]:.2f}]"
            print(f"[T={t:04d}] F={F_latest:.3f} | {body_str} | "
                  f"a={A_NAMES[action.index]} | C={agent.net.n_clusters}")

    # 跨通道集群统计
    cross = 0
    for c in agent.net.clusters:
        t_act = np.sum(np.abs(c.centroid[0:64])) > 0.1
        v_act = np.sum(np.abs(c.centroid[64:128])) > 0.1
        if t_act and v_act:
            cross += 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"Multimodal Episode Complete | steps={steps}")
        print(f"  Clusters:     {agent.net.n_clusters}")
        print(f"  Cross-modal:  {cross}/{agent.net.n_clusters} "
              f"({cross/max(1,agent.net.n_clusters):.0%})")
        print(f"  Mean F:       {np.mean(agent.F_history):.3f}")
        print(f"  Body final:   {agent.body.b}")

    return {
        'clusters': agent.net.n_clusters,
        'cross_modal': cross,
        'F_history': agent.F_history,
        'actions': actions,
    }


if __name__ == '__main__':
    realtime = '--realtime' in sys.argv
    pos_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    steps = int(pos_args[0]) if pos_args else 300

    stage = "3B: 实时多模态" if realtime else "3A: 多模态离线回放"
    print("=" * 60)
    print(f"  自由能原理智能体 — Stage {stage}")
    print(f"  Free Energy Principle Agent — Multimodal")
    print("=" * 60)
    print(f"  Steps: {steps} | D={330}")
    print(f"  Channels: text[64]|vision[64]|audio[64]|body[64]|meta[64]")
    import os
    if realtime:
        print(f"  Mode: REALTIME (camera + CLIP)")
    elif os.path.exists('multimodal_log.npy'):
        print(f"  Mode: full multimodal_log.npy")
    elif os.path.exists('clip_vision.npy'):
        print(f"  Vision: CLIP ViT-B/32 (real)")
    else:
        print(f"  Vision: pseudo")
    if not realtime:
        print(f"  Audio: pseudo (smooth drift)")
    print("-" * 60)

    result = run_multimodal_episode(steps=steps, realtime=realtime)
