"""
test_belief_anchor_v2.py —— 对照实验: 旧 vs 新查询路径检索对比

核心问题: 信念锚定是否改变了实际检索结果?
方法: 同一 Agent 状态, 同时计算旧查询和新查询, 对比 broca 检索结果。
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_types import Theta, BodyVector
from agent import Agent
from layer1_free_energy import SocialContext
from broca import Broca


def get_top_sentences(broca, query_vec, top_k=5):
    """返回 top-k 句子列表 (不采样, 确定性)"""
    broca._ensure_sent_embeddings()  # 确保懒加载完成
    q = query_vec[:64].astype(np.float32)
    q_norm = np.linalg.norm(q) + 1e-8
    s_norms = np.linalg.norm(broca._sent_embeddings, axis=1) + 1e-8
    sims = np.dot(broca._sent_embeddings, q) / (s_norms * q_norm)
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [broca.sentences[i] for i in top_indices], top_indices, sims[top_indices]


def compare_old_vs_new(agent, s, broca, label="", verbose=True):
    """对同一状态, 对比旧路径 (recall(s)) 和新路径 (belief) 的检索结果"""

    # ---- 旧路径: raw sensory recall ----
    c_recall = agent.net.recall(s.copy())
    old_query = c_recall.centroid[:64].copy() if c_recall else np.zeros(64, dtype=np.float32)

    # ---- 新路径: belief-anchored ----
    if agent.net.n_clusters > 0:
        top = max(agent.net.clusters, key=lambda c: c.activation)
        if top.activation > 0.01:
            belief_sem = top.centroid[:64].copy().astype(np.float32)
            new_query = belief_sem * 0.7 + s[:64].astype(np.float32) * 0.3
        else:
            new_query = np.zeros(64, dtype=np.float32)
    else:
        new_query = np.zeros(64, dtype=np.float32)

    # ---- 对比两个查询向量 ----
    cos_sim = float(np.dot(old_query, new_query) /
                    (np.linalg.norm(old_query) * np.linalg.norm(new_query) + 1e-8))

    # ---- 分别检索 top-3 句子 (确定性, 不采样) ----
    old_sents, old_idx, old_sim = get_top_sentences(broca, old_query, top_k=3)
    new_sents, new_idx, new_sim = get_top_sentences(broca, new_query, top_k=3)

    overlap = len(set(old_sents) & set(new_sents))

    if verbose:
        print(f"  [{label}]")
        print(f"    Query cosine similarity: {cos_sim:.4f} "
              f"{'(SAME)' if cos_sim > 0.999 else '(DIFFERENT)' if cos_sim < 0.95 else '(similar)'}")
        print(f"    Top-3 overlap: {overlap}/3")
        if overlap < 3:
            for i in range(3):
                marker_old = " ***" if old_sents[i] not in new_sents else ""
                marker_new = " ***" if new_sents[i] not in old_sents else ""
                if i < len(old_sents):
                    print(f"    OLD#{i+1} [sim={old_sim[i]:.3f}]{marker_old}: {old_sents[i][:90]}")
                if i < len(new_sents):
                    print(f"    NEW#{i+1} [sim={new_sim[i]:.3f}]{marker_new}: {new_sents[i][:90]}")

    return cos_sim, overlap, old_sents, new_sents


def main():
    print("=" * 70)
    print("  A/B Test: Old (recall(s)) vs New (belief-anchored) Retrieval")
    print("=" * 70)

    broca = Broca()

    # ---- Scenario 1: 学习多轮后对比 ----
    print("\n--- Scenario 1: After 30 varied inputs + high body need ---")
    rng = np.random.default_rng(99)
    agent = Agent(rng=rng, agent_id=0, n_agents=1)
    agent.body = BodyVector(mode='text')
    agent.theta.cluster_threshold = 0.4
    social_ctx = SocialContext(tau=15.0)

    for i in range(30):
        s = np.zeros(330, dtype=np.float32)
        s[:64] = np.random.default_rng(i * 7).normal(0, 0.5, 64).astype(np.float32)
        s[80] = 0.3 + 0.5 * (i % 3) / 3
        social_ctx.update(0.1 * (i % 5 - 2), 0.3)
        agent.step(s, i, social_ctx=social_ctx)

    s_test = np.zeros(330, dtype=np.float32)
    s_test[:64] = np.random.default_rng(15 * 7).normal(0, 0.5, 64).astype(np.float32)
    s_test[80] = 0.8
    agent.body.b[0] = 0.3   # 社交需求高
    agent.body.b[3] = 0.1   # 新颖需求高
    agent.step(s_test, 30, social_ctx=social_ctx)

    cos1, ov1, old1, new1 = compare_old_vs_new(agent, s_test, broca, "30 inputs + body need")

    # ---- Scenario 2: 同一输入, 不同 body 状态 (satisfied vs deprived) ----
    print("\n--- Scenario 2: Same input, body SATISFIED vs DEPRIVED ---")

    def make_agent_with_history(seed, n_steps=20):
        rng_a = np.random.default_rng(seed)
        ag = Agent(rng=rng_a, agent_id=0, n_agents=1)
        ag.body = BodyVector(mode='text')
        ag.theta.cluster_threshold = 0.4
        sc = SocialContext(tau=15.0)
        for i in range(n_steps):
            s = np.zeros(330, dtype=np.float32)
            s[:64] = np.random.default_rng(i * 3 + seed).normal(0, 0.3, 64).astype(np.float32)
            s[80] = 0.5
            sc.update(0.0, 0.2)
            ag.step(s, i, social_ctx=sc)
        return ag, sc

    s_same = np.zeros(330, dtype=np.float32)
    s_same[:64] = np.random.default_rng(999).normal(0, 0.3, 64).astype(np.float32)
    s_same[80] = 0.6

    # 2a: 满足状态
    ag_a, sc_a = make_agent_with_history(77)
    ag_a.body.b = ag_a.body.setpoints.copy()
    ag_a.body.b[0] = 0.72
    ag_a.body.b[3] = 0.55
    ag_a.step(s_same.copy(), 20, social_ctx=sc_a)
    compare_old_vs_new(ag_a, s_same, broca, "SATISFIED (b0=0.72, b3=0.55)")

    # 2b: 饥渴状态
    ag_b, sc_b = make_agent_with_history(77)
    ag_b.body.b[0] = 0.25
    ag_b.body.b[3] = 0.05
    ag_b.step(s_same.copy(), 20, social_ctx=sc_b)
    compare_old_vs_new(ag_b, s_same, broca, "DEPRIVED (b0=0.25, b3=0.05)")

    # ---- 温度 & top-k 对比 ----
    print("\n  --- Temp & top-k modulation ---")
    for label, ag in [("SATISFIED", ag_a), ("DEPRIVED", ag_b)]:
        v = ag.valence_history[-1] if ag.valence_history else 0
        a = ag.arousal_history[-1] if ag.arousal_history else 0
        temp_base = 0.5 + abs(v) * 0.8
        body = ag.body
        social_need = max(0.0, body.setpoints[0] - body.b[0])
        novelty_need = max(0.0, 0.5 - body.b[3])
        temp = temp_base * (1.0 + social_need * 0.6 + novelty_need * 0.4)
        k = max(3, min(12, int(5 + a * 8)))
        print(f"    {label:10s}: b0={body.b[0]:.2f} b3={body.b[3]:.2f} "
              f"need_soc={social_need:.2f} need_nov={novelty_need:.2f} "
              f"temp={temp:.2f} k={k}")

    # ---- Scenario 3: 100 次采样, 统计 old vs new 回应分布 ----
    print("\n--- Scenario 3: 100-sample distribution comparison ---")
    rng3 = np.random.default_rng(55)
    agent3 = Agent(rng=rng3, agent_id=0, n_agents=1)
    agent3.body = BodyVector(mode='text')
    agent3.theta.cluster_threshold = 0.4
    sc3 = SocialContext(tau=15.0)
    for i in range(25):
        s = np.zeros(330, dtype=np.float32)
        s[:64] = np.random.default_rng(i * 11).normal(0, 0.5, 64).astype(np.float32)
        s[80] = 0.4 + 0.3 * (i % 3 - 1)
        sc3.update(0.1, 0.2)
        agent3.step(s, i, social_ctx=sc3)

    agent3.body.b[0] = 0.3
    s_test3 = np.zeros(330, dtype=np.float32)
    s_test3[:64] = np.random.default_rng(999).normal(0, 0.5, 64).astype(np.float32)
    s_test3[80] = -0.3
    agent3.step(s_test3, 25, social_ctx=sc3)

    # 构造两个查询
    c_recall3 = agent3.net.recall(s_test3.copy())
    old_q = c_recall3.centroid[:64].copy() if c_recall3 else np.zeros(64, dtype=np.float32)
    top3 = max(agent3.net.clusters, key=lambda c: c.activation)
    belief_q = top3.centroid[:64].copy().astype(np.float32)
    new_q = belief_q * 0.7 + s_test3[:64].astype(np.float32) * 0.3

    # 参数
    v3 = agent3.valence_history[-1]
    a3 = agent3.arousal_history[-1]
    temp_base3 = 0.5 + abs(v3) * 0.8
    body3 = agent3.body
    sn3 = max(0.0, body3.setpoints[0] - body3.b[0])
    nn3 = max(0.0, 0.5 - body3.b[3])
    temp_new = temp_base3 * (1.0 + sn3 * 0.6 + nn3 * 0.4)
    k_new = max(3, min(12, int(5 + a3 * 8)))
    temp_old = 0.5 + abs(v3) * 0.8
    k_old = 5

    print(f"    Old: temp={temp_old:.2f} k={k_old}")
    print(f"    New: temp={temp_new:.2f} k={k_new}")

    old_set = set()
    new_set = set()
    for _ in range(100):
        ow, _ = broca.speak_sentence(old_q, temperature=temp_old, top_k=k_old)
        nw, _ = broca.speak_sentence(new_q, temperature=temp_new, top_k=k_new)
        old_set.add("".join(ow) if ow else "")
        new_set.add("".join(nw) if nw else "")

    shared = old_set & new_set
    print(f"    Old unique:  {len(old_set)}")
    print(f"    New unique:  {len(new_set)}")
    print(f"    Shared:      {len(shared)}")
    print(f"    Old-only:    {len(old_set - shared)}")
    print(f"    New-only:    {len(new_set - shared)}")

    if len(new_set - shared) > 0:
        print(f"    New-only examples:")
        for r in list(new_set - shared)[:2]:
            print(f"      -> {r[:100]}")
    if len(old_set - shared) > 0:
        print(f"    Old-only examples:")
        for r in list(old_set - shared)[:2]:
            print(f"      -> {r[:100]}")

    print("\n" + "=" * 70)
    print("  Key metric: Top-3 overlap < 3 means belief anchoring")
    print("  changes WHICH sentences are retrieved (not just sampling).")
    print("  Temperature/top-k modulation further shifts the distribution.")
    print("=" * 70)


if __name__ == '__main__':
    main()
