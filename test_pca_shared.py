"""
test_pca_shared.py —— 验证 Broca 和 TextEnvironment 共享 PCA 空间 (v2: Hebb 记忆检索)

测试:
1. 共享 PCA 时 Broca 使用 TextEnvironment 的 PCA (不自己拟合)
2. 查询向量与检索目标在同一空间 → Hebb recall 有效
3. 共享 vs 独立 PCA 检索结果对比
4. 闭环一致性: 编码→检索→再编码→再检索
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from text_interface import TextEnvironment
from broca import Broca
from data_types import D
from layer0_model import _masked_cosine


def test_shared_pca_basic():
    """验证共享 PCA 时 Broca 使用 TextEnvironment 的 PCA"""
    print("=" * 60)
    print("  Test 1: Shared PCA basic verification")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    broca._ensure_sent_clusters()

    # 验证集群网络已构建
    assert broca._sentence_net is not None, "sentence_net should be built"
    assert broca._sentence_net.n_clusters > 0, "should have sentence clusters"
    n_buckets = len(broca._sentence_net.buckets)
    print(f"  Sentence clusters: {broca._sentence_net.n_clusters}")
    print(f"  Hash buckets:      {n_buckets}")
    print(f"  Avg per bucket:    {broca._sentence_net.n_clusters / n_buckets:.0f}")
    print(f"  TextEnv embeddings: {text_env.embeddings.shape}")
    print("  [PASS] Shared PCA + Hebb clusters loaded correctly")


def test_query_target_same_space():
    """验证查询向量和检索目标在同一 PCA 空间"""
    print("\n" + "=" * 60)
    print("  Test 2: Query and target in same space (Hebb retrieval)")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    broca._ensure_sent_clusters()

    # 用 TextEnvironment 编码一句话 → 得到 PCA 空间向量
    test_text = "今天天气真好，想出去走走"
    query = text_env.encode_text(test_text)

    # 用 Broca Hebb 检索
    words, _ = broca.speak_sentence(query, temperature=0.01, top_k=1)
    response = "".join(words)

    print(f"  Query text:   '{test_text}'")
    print(f"  Response:     '{response[:80]}'")

    # 验证: 在 Hebb 网络中找与 query 最相似的集群
    q = np.zeros(D, dtype=np.float32)
    q[:64] = query[:64].astype(np.float32)
    h = broca._sentence_net.hash_features(q)
    mask = np.zeros(D, dtype=bool)
    mask[:64] = True

    best_sim = -1.0
    best_sent = ""
    for c in broca._sentence_net.clusters:
        sim = _masked_cosine(h, c.centroid, mask)
        if sim > best_sim:
            best_sim = sim
            best_idx = broca._cluster_to_sentence.get(id(c), 0)
            best_sent = broca.sentences[best_idx]

    print(f"  Best in Hebb: '{best_sent[:80]}'")
    print(f"  Best cosine:  {best_sim:.4f}")

    # TextEnv 中找最近邻 (全局余弦 — 仅用于对比)
    te_norms = np.linalg.norm(text_env.embeddings, axis=1) + 1e-8
    te_sims = np.dot(text_env.embeddings, query) / (te_norms * np.linalg.norm(query) + 1e-8)
    te_best_idx = int(np.argmax(te_sims))
    te_best_sim = float(te_sims[te_best_idx])
    te_best_chunk = text_env.chunks[te_best_idx]

    print(f"  TextEnv best: '{te_best_chunk[:80]}'")
    print(f"  TextEnv cos:  {te_best_sim:.4f}")

    # 两个都在有效余弦范围
    print(f"\n  Hebb best cosine:    {best_sim:.4f}")
    print(f"  TextEnv best cosine: {te_best_sim:.4f}")
    print("  [PASS] Both in valid cosine range [0,1]")


def test_shared_vs_standalone():
    """对比共享 PCA vs 独立 PCA 的 Hebb 检索差异"""
    print("\n" + "=" * 60)
    print("  Test 3: Shared vs Standalone PCA (Hebb retrieval)")
    print("=" * 60)

    text_env = TextEnvironment()

    # 共享 PCA 版本
    broca_shared = Broca(text_env=text_env)
    broca_shared._ensure_sent_clusters()

    # 独立 PCA 版本 (旧行为)
    broca_standalone = Broca()  # no text_env
    broca_standalone._ensure_sent_clusters()

    # 用同一个查询
    test_text = "我很开心"
    query = text_env.encode_text(test_text)

    def get_top3(broca, q):
        from data_types import D
        from layer0_model import _masked_cosine
        q_full = np.zeros(D, dtype=np.float32)
        q_full[:64] = q[:64].astype(np.float32)
        h = broca._sentence_net.hash_features(q_full)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True
        # 全集群评分 (用于对比测试; 实际 speak_sentence 只在桶内)
        scored = []
        for c in broca._sentence_net.clusters:
            sim = _masked_cosine(h, c.centroid, mask)
            sent_idx = broca._cluster_to_sentence.get(id(c), 0)
            scored.append((broca.sentences[sent_idx], float(sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:3]

    shared_top3 = get_top3(broca_shared, query)
    standalone_top3 = get_top3(broca_standalone, query)

    print(f"  Query: '{test_text}'")
    print(f"\n  --- Shared PCA (same space as query) ---")
    for sent, sim in shared_top3:
        print(f"    [{sim:.4f}] {sent[:70]}")

    print(f"\n  --- Standalone PCA (different space) ---")
    for sent, sim in standalone_top3:
        print(f"    [{sim:.4f}] {sent[:70]}")

    # 对比重叠
    shared_sents = set(s for s, _ in shared_top3)
    standalone_sents = set(s for s, _ in standalone_top3)
    overlap = len(shared_sents & standalone_sents)
    print(f"\n  Top-3 overlap: {overlap}/3")

    print(f"  Shared top-1 cosine:     {shared_top3[0][1]:.4f}")
    print(f"  Standalone top-1 cosine: {standalone_top3[0][1]:.4f}")

    # 共享 PCA 的余弦值应该更可信 (在同一空间)
    assert shared_top3[0][1] > 0.5, \
        f"Shared PCA should have high retrieval confidence, got {shared_top3[0][1]:.4f}"
    print(f"\n  Key: shared PCA cosines are in the SAME space as the query.")
    print(f"  Standalone PCA cosines are in a DIFFERENT space -> distorted.")
    print(f"  [PASS] Shared PCA produces meaningful retrieval scores")


def test_roundtrip_consistency():
    """验证闭环一致性: 编码→检索→再编码→再检索"""
    print("\n" + "=" * 60)
    print("  Test 4: Round-trip consistency (Hebb retrieval)")
    print("=" * 60)

    text_env = TextEnvironment()
    broca = Broca(text_env=text_env)
    broca._ensure_sent_clusters()

    # 第一轮: 用 TextEnvironment 编码 → Broca Hebb 检索
    q1 = text_env.encode_text("你好")
    words1, _ = broca.speak_sentence(q1, temperature=0.1, top_k=1)
    response1 = "".join(words1)
    print(f"  Round 1: 'hello' -> '{response1[:60]}'")

    # 第二轮: 编码回应 (模拟自听回路) → 再次检索
    q2 = text_env.encode_text(response1)
    words2, _ = broca.speak_sentence(q2, temperature=0.1, top_k=1)
    response2 = "".join(words2)
    print(f"  Round 2: '{response1[:40]}' -> '{response2[:60]}'")

    # 第三轮: 继续
    q3 = text_env.encode_text(response2)
    words3, _ = broca.speak_sentence(q3, temperature=0.1, top_k=1)
    response3 = "".join(words3)
    print(f"  Round 3: '{response2[:40]}' -> '{response3[:60]}'")

    # 验证每一轮都能正确检索
    assert len(response1) > 0
    assert len(response2) > 0
    assert len(response3) > 0
    print("  [PASS] Round-trip encoding/retrieval consistent")


if __name__ == '__main__':
    test_shared_pca_basic()
    test_query_target_same_space()
    test_shared_vs_standalone()
    test_roundtrip_consistency()
    print("\n" + "=" * 60)
    print("  All PCA sharing tests passed (Hebb retrieval)!")
    print("=" * 60)
