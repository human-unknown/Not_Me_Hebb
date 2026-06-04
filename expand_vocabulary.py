"""
expand_vocabulary.py —— 词表扩展：543 → ~8,000 中文词

v2: 共现向量 (co-occurrence) + 中文过滤
方法:
1. jieba 分词全语料 → 收集高频中文词 (freq ≥ 10, 纯中文)
2. 构建词-词共现矩阵 (同句出现 → 正权重)
3. TruncatedSVD → 64-dim 共现向量
4. Procrustes 对齐: 64-dim co-occ space → 64-dim audio space (用锚点词)
5. 生成 expanded word_spectrum_dataset.npy

为什么不用 transformer:
- 词序网络需要句法/序列信息, 不是语义信息
- "猫" 和 "狗" 语义相似, 但它们的上下文词序模式不同
- 共现向量捕获的是 "哪些词倾向于出现在相似的上下文中"
- 这对词序预测更有用
"""

import numpy as np
import os, sys, re
from collections import Counter
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD


def is_chinese_word(w: str) -> bool:
    """检查是否为有效中文词 (至少包含一个中文字符, 无标点/数字)"""
    if len(w) < 1:
        return False
    # 必须包含中文字符
    has_cjk = any('一' <= c <= '鿿' or '㐀' <= c <= '䶿' for c in w)
    if not has_cjk:
        return False
    # 不能是纯标点/数字/拉丁字符
    all_valid = all(
        '一' <= c <= '鿿' or
        '㐀' <= c <= '䶿' or
        '　' <= c <= '〿' or  # CJK punctuation
        c in '·'  # middle dot
        for c in w
    )
    # 去除纯标点的 (可能包含单个标点)
    pure_punct = all(not ('一' <= c <= '鿿') for c in w)
    if pure_punct:
        return False
    return all_valid


def extract_vocabulary(corpus_path: str, min_freq: int = 10,
                       max_words: int = 10000) -> list[tuple[str, int]]:
    """从语料提取高频中文词"""
    import jieba

    print(f"Tokenizing corpus: {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    sentences = [s.strip() for s in re.split(r'[。！？；\n]+', text)
                 if len(s.strip()) > 5]
    print(f"  {len(sentences)} sentences")

    wc = Counter()
    for i, sent in enumerate(sentences):
        words = [w for w in jieba.lcut(sent) if len(w.strip()) >= 1]
        for w in words:
            if is_chinese_word(w):
                wc[w] += 1
        if (i + 1) % 50000 == 0:
            print(f"  ... {i + 1}/{len(sentences)} sentences, "
                  f"{len(wc)} unique Chinese words")

    print(f"  {len(wc)} unique Chinese words total")

    # 过滤: freq >= min_freq
    vocab = [(w, c) for w, c in wc.items() if c >= min_freq]
    vocab.sort(key=lambda x: -x[1])
    if len(vocab) > max_words:
        vocab = vocab[:max_words]

    total_chinese = sum(wc.values())
    covered = sum(c for _, c in vocab)
    print(f"  {len(vocab)} Chinese words with freq >= {min_freq}")
    print(f"  Coverage: {covered}/{total_chinese} = "
          f"{covered / max(total_chinese, 1) * 100:.1f}%")

    return vocab


def build_cooccurrence_matrix(
    corpus_path: str,
    word_to_idx: dict[str, int],
    n_words: int,
) -> np.ndarray:
    """构建词-词共现矩阵 (句级共现)。

    两个词在同一句话中出现 → 共现计数 +1。
    使用稀疏矩阵处理大词汇量。
    """
    import jieba

    print(f"Building co-occurrence matrix: {n_words} x {n_words}...")
    cooc = lil_matrix((n_words, n_words), dtype=np.float32)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [s.strip() for s in re.split(r'[。！？；\n]+', text)
                 if len(s.strip()) > 5]

    for si, sent in enumerate(sentences):
        words = [w for w in jieba.lcut(sent) if len(w.strip()) >= 1]
        # 获取句子中出现的词索引
        indices = []
        for w in words:
            idx = word_to_idx.get(w)
            if idx is not None:
                indices.append(idx)
        indices = list(set(indices))  # 去重: 同句中多次出现只计1次

        # 所有词对共现
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                cooc[indices[i], indices[j]] += 1
                cooc[indices[j], indices[i]] += 1

        if (si + 1) % 50000 == 0:
            nnz = cooc.nnz
            print(f"  ... {si + 1}/{len(sentences)} sentences, "
                  f"{nnz} non-zero entries")

    # 转换为密集矩阵 (用 SVD 需要)
    print(f"  Final: {cooc.nnz} non-zero entries")
    # PMI-like 归一化: 用 log(1 + count)
    cooc_dense = np.log1p(cooc.toarray().astype(np.float32))
    return cooc_dense


def align_spaces(source_embeddings: np.ndarray,
                 target_vectors: np.ndarray,
                 ) -> dict:
    """Procrustes 对齐"""
    src_mean = source_embeddings.mean(axis=0, keepdims=True)
    tgt_mean = target_vectors.mean(axis=0, keepdims=True)
    src_centered = source_embeddings - src_mean
    tgt_centered = target_vectors - tgt_mean

    M = tgt_centered.T @ src_centered
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt

    src_var = np.var(source_embeddings, axis=0).sum()
    tgt_var = np.var(target_vectors, axis=0).sum()
    scale = np.sqrt(tgt_var / (src_var + 1e-8))

    transformed = (src_centered @ R.T) * scale + tgt_mean
    mse = np.mean((transformed - target_vectors) ** 2)
    print(f"  Procrustes alignment: MSE={mse:.6f}, scale={scale:.3f}")

    return {
        'rotation': R.astype(np.float32),
        'scale': float(scale),
        'src_mean': src_mean.astype(np.float32).flatten(),
        'tgt_mean': tgt_mean.astype(np.float32).flatten(),
        'mse': float(mse),
    }


def apply_alignment(embeddings: np.ndarray, transform: dict) -> np.ndarray:
    """应用 Procrustes 变换"""
    src_centered = embeddings - transform['src_mean'].reshape(1, -1)
    transformed = (src_centered @ transform['rotation'].T) * transform['scale'] \
                  + transform['tgt_mean'].reshape(1, -1)
    return transformed.astype(np.float32)


def build_expanded_dataset(
    corpus_path: str,
    output_path: str,
    min_freq: int = 10,
    max_words: int = 8000,
):
    """主流程: 构建扩展词表数据集 (共现向量版)"""
    base = os.path.dirname(__file__)

    # ---- 1. 加载现有词表 (543 锚点) ----
    old_ds_path = os.path.join(base, 'word_spectrum_dataset.npy')
    old_raw = np.load(old_ds_path, allow_pickle=True)
    old_words = [r[2] for r in old_raw]
    old_vecs = np.stack([r[0] for r in old_raw])
    old_mels = np.stack([r[1] for r in old_raw])
    print(f"Existing dataset: {len(old_words)} words")

    # ---- 2. 提取高频中文词 ----
    vocab = extract_vocabulary(corpus_path, min_freq=min_freq,
                               max_words=max_words)
    new_words_all = [w for w, c in vocab]
    print(f"Target vocabulary: {len(new_words_all)} Chinese words")

    # ---- 3. 确保旧锚点词也包含在内 ----
    old_chinese = [w for w in old_words if is_chinese_word(w)]
    for w in old_chinese:
        if w not in new_words_all:
            new_words_all.append(w)
    print(f"  After adding {len(old_chinese) - len(set(old_chinese) & set(new_words_all))} "
          f"anchor words: {len(new_words_all)} total")

    # 重新构建 word_to_idx
    word_to_idx = {w: i for i, w in enumerate(new_words_all)}
    n_words = len(new_words_all)

    # ---- 4. 构建共现矩阵 ----
    cooc_matrix = build_cooccurrence_matrix(corpus_path, word_to_idx, n_words)

    # ---- 5. SVD 降维: n_words → 64 ----
    print(f"SVD: {n_words} → 64...")
    svd = TruncatedSVD(n_components=64, random_state=42)
    cooc_64 = svd.fit_transform(cooc_matrix).astype(np.float32)
    print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    # 标准化每个分量到单位方差 (避免 Procrustes scale 过大)
    cooc_std = cooc_64.std(axis=0, keepdims=True) + 1e-8
    cooc_64 = cooc_64 / cooc_std
    print(f"  Normalized SVD components to unit variance")

    # ---- 6. 直接使用共现向量 (不强制对齐到 audio space) ----
    # 词序网络需要句法/共现信息，不需要音频特征。
    # 纯共现向量更适合预测 "下一个词是什么"。
    # 对于 WordSpeaker: vec_to_word 在共现空间中做余弦相似度检索。
    all_audio_64 = cooc_64.copy()
    transform = {'mse': 0.0, 'scale': 1.0, 'method': 'cooccurrence_only'}

    # 对于有音频特征的 543 个原始词，保留 mel (用于 TTS)
    # 对于新词，mel 置零 (WordSpeaker 懒生成 edge-tts 音频)

    # ---- 8. 构建扩展数据集 ----
    new_records = []
    old_word_to_mel = {old_words[i]: old_mels[i] for i in range(len(old_words))}

    for i, word in enumerate(new_words_all):
        vec = all_audio_64[i].astype(np.float32)
        if word in old_word_to_mel:
            mel = old_word_to_mel[word]
        else:
            mel = np.zeros((80, 16), dtype=np.float32)
        new_records.append(np.array([vec, mel, word], dtype=object))

    records_array = np.array(new_records, dtype=object)
    np.save(output_path, records_array)
    print(f"\nSaved expanded dataset: {output_path}")
    print(f"  {len(new_records)} words")

    # ---- 9. 保存变换参数 ----
    transform_path = os.path.join(base, 'word_space_transform.npy')
    np.save(transform_path, {
        'svd_components': svd.components_,
        'svd_std': cooc_std.astype(np.float32).flatten(),
        'n_words': len(new_records),
        'method': 'cooccurrence_svd_v3',
    })
    print(f"  Transform saved: {transform_path}")

    print(f"\n{'='*60}")
    print(f"  EXPANSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Original words:     543")
    print(f"  Expanded words:     {len(new_records)}")
    print(f"  Growth:             {len(new_records)/543:.1f}x")
    print(f"  Method:             co-occurrence SVD (no audio alignment)")

    return new_records


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-freq', type=int, default=10)
    ap.add_argument('--max-words', type=int, default=8000)
    ap.add_argument('--corpus', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'corpus.txt'))
    ap.add_argument('--output', type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                        'word_spectrum_dataset_expanded.npy'))
    args = ap.parse_args()

    build_expanded_dataset(
        corpus_path=args.corpus,
        output_path=args.output,
        min_freq=args.min_freq,
        max_words=args.max_words,
    )
