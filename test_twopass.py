"""两遍离线回放验证 — 3A 第四条"""
import numpy as np
from agent import Agent
from multimodal_interface import MultimodalReplay
from text_interface import TextEnvironment
from data_types import BodyVector

text_env = TextEnvironment()
text_emb = text_env.embeddings

def run_pass(n_frames=200):
    agent = Agent(rng=np.random.default_rng(42))
    agent.body = BodyVector(mode='text')
    agent.theta.cluster_threshold = 0.55
    agent.record_action_consequence = lambda s: None
    env = MultimodalReplay(text_embeddings=text_emb, n_frames=n_frames)
    for t in range(n_frames):
        s = env.get_sensory(body=agent.body)
        if np.sum(np.abs(s[0:64])) < 0.01:
            s[0:64] = text_emb[t % len(text_emb)]
        s[256:260] = 0.0
        agent.step(s, t)
        env.step(0)
    return np.mean(agent.F_history[-50:]), agent.net.n_clusters

F1, c1 = run_pass(200)
F2, c2 = run_pass(200)

print(f"Pass 1: F_mean(last50)={F1:.3f}, clusters={c1}")
print(f"Pass 2: F_mean(last50)={F2:.3f}, clusters={c2}")
print(f"F lower on 2nd pass? {F2 < F1}")
print(f"(expected: F1≈F2 — clusters don't persist across passes)")
