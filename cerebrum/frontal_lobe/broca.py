"""
broca.py —— Hebb 词序网络 (Broca 区)
双 Hebb 架构：概念网络管"想什么"，词序网络管"怎么说"。
纯 Hebb，零手写规则。

v2: 整句检索模式 — 直接取最近语料句输出，不拼词。
v3: 共享 PCA — 使用 TextEnvironment 的 encoder+PCA, 确保查询和检索在同一空间。
v4: Hebb 记忆检索 — 句子存储为 ClusterNetwork 的集群, 说话 = recall() 回忆,
    不是全局余弦扫描。哈希定位 O(1) + 桶内竞争。
"""
import numpy as np, re, os, logging

_log = logging.getLogger("cerebrum.broca")


class Broca:
    def __init__(self, text_env=None, load_corpus: bool = False):
        """初始化 Broca 区。

        Args:
            text_env: TextEnvironment 实例。如果提供，共享其 encoder 和 PCA，
                      确保检索空间与 Agent 感知空间一致。
            load_corpus: 是否从语料预训练网络 (v5.7默认False=纯净启动，
                        trigram/句/概念词网络从零开始，从互动中在线学习)
        """
        base = os.path.dirname(__file__)
        # 优先使用扩展词表 (12,000 词), 回退到原始词表 (543 词)
        expanded_path = os.path.join(base, 'word_spectrum_dataset_expanded.npy')
        original_path = os.path.join(base, 'word_spectrum_dataset.npy')
        if os.path.exists(expanded_path):
            raw = np.load(expanded_path, allow_pickle=True)
            _log.info(f"  Using expanded vocabulary ({len(raw)} words)")
        else:
            raw = np.load(original_path, allow_pickle=True)
        self.word_list = [r[2] for r in raw]
        self.word_vecs = np.stack([r[0] for r in raw])
        self.word_norms = np.linalg.norm(self.word_vecs, axis=1)
        # O(1) 词→索引映射 (12K 词时避免线性扫描)
        self._word_to_idx: dict[str, int] = {
            w: i for i, w in enumerate(self.word_list)}

        # ---- 共享 PCA 空间 (v3) ----
        self._text_env = text_env
        self._sentence_net = None       # ClusterNetwork — 每句一个集群
        self._cluster_to_sentence = {}  # {id(cluster): sentence_index}

        from cns.data_types import Theta, D
        from cerebrum.limbic_system.hippocampus import ClusterNetwork

        # 词序网络: trigram Hebb 链 — 纯净模式从空开始
        theta = Theta(); theta.cluster_threshold = 0.05
        self.word_order_net = ClusterNetwork(theta)

        # 概念→词 Hebb 联想网络 — 纯净模式从空开始
        cw_theta = Theta(); cw_theta.cluster_threshold = 0.15
        self.concept_word_net = ClusterNetwork(cw_theta)
        self._concept_word_built = False

        if load_corpus:
            # 旧模式: 从语料加载并预训练所有网络
            self._load_corpus_and_train()
        else:
            # v5.7 纯净模式: 网络从零开始, 从互动中在线学习
            self.sentences = []
            _log.info(f"Broca: clean mode — {len(self.word_list)} words, "
                  f"0 trigram clusters, 0 sentences "
                  f"(networks grow from interaction)")

    def _load_corpus_and_train(self):
        """从语料加载句子并构建 trigram/句/概念词网络 (旧模式)."""
        base = os.path.dirname(__file__)
        corpus_path = os.path.join(base, 'corpus.txt')
        root_corpus = os.path.join(base, '..', '..', 'corpus.txt')
        if not os.path.exists(corpus_path) and os.path.exists(root_corpus):
            corpus_path = root_corpus
        if not os.path.exists(corpus_path):
            # v7.6: pure mode — no corpus file, skip pre-training
            import logging
            logging.getLogger(__name__).info(
                "corpus.txt not found — skipping trigram pre-training (pure mode)")
            self.sentences = []
            return
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        raw_sentences = [s.strip() for s in re.split(r'[。！？；\n]+', text)
                         if len(s.strip()) > 5]
        self.sentences = []
        for s in raw_sentences:
            cleaned = re.sub(r'^[一-鿿\w/·]{1,6}[：:]\s*', '', s)
            if len(cleaned) >= 2:
                self.sentences.append(s)
        dropped = len(raw_sentences) - len(self.sentences)
        if dropped > 0:
            _log.info(f"  Filtered {dropped} name-prefix-only sentences")

        self._build_word_order()
        _log.info(f"Broca: {self.word_order_net.n_clusters} word-order clusters, "
              f"{len(self.word_list)} words, {len(self.sentences)} sentences")

    def _word_to_vec(self, w):
        idx = self._word_to_idx.get(w)
        if idx is not None:
            return self.word_vecs[idx]
        return None

    def _vec_to_word(self, v):
        # 向量化余弦相似度 (12K words)
        sims = np.dot(self.word_vecs, v) / (self.word_norms * np.linalg.norm(v) + 1e-8)
        return self.word_list[int(np.argmax(sims))]

    def _build_word_order(self, min_trigram_freq: int = 3,
                          max_clusters: int = 50000):
        """构建词序网络 — 每个唯一 trigram = 一个 Hebb 集群。

        v7: 扩展词表优化 — O(1) 词查找 + 频率过滤 + 容量上限。
        - min_trigram_freq: 最小 trigram 出现次数 (默认 3)
        - max_clusters: 最大集群数 (默认 50000, 超出的低频 trigram 丢弃)

        直接插入绕过 learn() 的 EMA 合并，每个唯一 trigram = 独立集群。
        """
        import jieba
        from cns.data_types import D, Cluster

        # ---- 收集所有 trigram (只存出现次数 ≥ min_freq 的) ----
        tri_counts: dict[tuple[str, str, str], int] = {}
        # 延迟计算 vecs: 只对高频 trigram 计算 (节省内存)
        n_skipped_words = 0

        n_sents = len(self.sentences)
        for si, sent in enumerate(self.sentences):
            if (si + 1) % 50000 == 0:
                _log.info(f"  Word-order: processing sentence {si+1}/{n_sents} "
                      f"({len(tri_counts)} unique trigrams so far)")
            words = [w for w in jieba.lcut(sent) if len(w.strip()) >= 1]
            for i in range(len(words) - 2):
                w_i, w_j, w_k = words[i], words[i + 1], words[i + 2]
                key = (w_i, w_j, w_k)
                tri_counts[key] = tri_counts.get(key, 0) + 1

        total_raw = sum(tri_counts.values())
        total_unique = len(tri_counts)

        # ---- 过滤: 只保留频率 ≥ min_trigram_freq ----
        filtered = [(key, cnt) for key, cnt in tri_counts.items()
                    if cnt >= min_trigram_freq]
        filtered.sort(key=lambda x: -x[1])  # 按频率降序

        if len(filtered) > max_clusters:
            _log.info(f"  Word-order: truncating {len(filtered)} → {max_clusters} "
                  f"(min freq threshold: {filtered[max_clusters-1][1]})")
            filtered = filtered[:max_clusters]

        _log.info(f"  Word-order: {total_unique} unique → {len(filtered)} filtered "
              f"(from {total_raw} total trigrams, min_freq={min_trigram_freq})")

        # ---- 构建 vecs 并插入集群 ----
        n_inserted = 0
        for (w_i, w_j, w_k), count in filtered:
            v_i = self._word_to_vec(w_i)
            v_j = self._word_to_vec(w_j)
            v_k = self._word_to_vec(w_k)
            if v_i is None or v_j is None or v_k is None:
                n_skipped_words += 1
                continue

            padded = np.zeros(D, dtype=np.float32)
            padded[:64] = v_i[:64]
            padded[64:128] = v_j[:64]
            padded[128:192] = v_k[:64]

            h = self.word_order_net.hash_features(padded)
            c = Cluster(centroid=h.copy())
            c.count = count
            c.activation = min(1.0, count * 0.01)

            self.word_order_net.clusters.append(c)
            hash_key = self.word_order_net._hash_to_bucket(h)
            self.word_order_net.buckets.setdefault(hash_key, []).append(c)
            n_inserted += 1

        if n_skipped_words > 0:
            _log.info(f"  Word-order: skipped {n_skipped_words} trigrams "
                  f"(words not in vocabulary)")

        _log.info(f"  Word-order: {n_inserted} trigram clusters inserted")

    def _learn_from_sentence(self, sentence: str):
        """从单句学习 trigram — 用于睡眠巩固中的词序强化。

        与 _build_word_order 相同的直接插入逻辑，
        但每次只处理一句，增量添加到现有网络。
        不会移除已有集群，只会添加新的 trigram 或增加计数。
        """
        import jieba
        from cns.data_types import D, Cluster

        words = [w for w in jieba.lcut(sentence) if len(w.strip()) >= 1]
        if len(words) < 3:
            return 0

        n_added = 0
        for i in range(len(words) - 2):
            w_i, w_j, w_k = words[i], words[i + 1], words[i + 2]
            v_i = self._word_to_vec(w_i)
            v_j = self._word_to_vec(w_j)
            v_k = self._word_to_vec(w_k)
            if v_i is None or v_j is None or v_k is None:
                continue

            padded = np.zeros(D, dtype=np.float32)
            padded[:64] = v_i[:64]
            padded[64:128] = v_j[:64]
            padded[128:192] = v_k[:64]

            # 检查是否已存在
            h = self.word_order_net.hash_features(padded)
            hash_key = self.word_order_net._hash_to_bucket(h)
            bucket = self.word_order_net.buckets.get(hash_key, [])
            mask = np.zeros(D, dtype=bool)
            mask[:192] = True

            existing = None
            from cerebrum.limbic_system.hippocampus import _masked_cosine
            for c in bucket:
                sim = _masked_cosine(h, c.centroid, mask)
                if sim > 0.95:  # 几乎相同的 trigram
                    existing = c
                    break

            if existing is not None:
                existing.count += 1
                existing.activation = min(1.0, existing.activation + 0.05)
            else:
                c = Cluster(centroid=h.copy())
                c.count = 1
                c.activation = 0.1
                self.word_order_net.clusters.append(c)
                self.word_order_net.buckets.setdefault(hash_key, []).append(c)
                n_added += 1

        return n_added

    def learn_from_interaction(self, human_text: str, agent_response: str,
                               age: int = 0):
        """v5.7 纯净模式: 从真实对话中在线学习 trigram

        每次人机对话后调用，从双方的文本中增量构建词序网络。
        这是"从零生长"的语言能力 — 不依赖任何预训练语料。

        年龄感知过滤 (v5.7):
          - age=0 (婴儿): 只学习人类输入, 不学自己的输出 (避免胡话污染)
          - age=1 (儿童): 学习人类+Agent (Agent回应如太短跳过)
          - age=2+: 全部学习

        Args:
            human_text: 用户输入的文本
            agent_response: Agent 的回应文本
            age: 发育年龄 (0=婴儿只模仿, 1=儿童, 2=青少年, 3=成人)

        Returns:
            (n_human_trigrams, n_agent_trigrams) 新增 trigram 数
        """
        n_human = self._learn_from_sentence(human_text) if human_text else 0

        # 年龄感知过滤
        if age <= 0:
            # 婴儿: 不学自己的输出 (胡话会污染网络)
            n_agent = 0
        elif age == 1 and agent_response:
            # 儿童: 只在回应足够长时学习 (短回应通常是模仿, 安全)
            if len(agent_response) >= 6:
                n_agent = self._learn_from_sentence(agent_response)
            else:
                n_agent = 0
        else:
            # 青少年+: 正常学习
            n_agent = self._learn_from_sentence(agent_response) if agent_response else 0

        # 同时学习概念→词映射 (始终学习人类输入)
        if human_text:
            try:
                concept_vec = None
                if self._text_env is not None:
                    concept_vec = self._text_env.encode_text(human_text)
                if concept_vec is not None:
                    import jieba
                    # 从人类输入学概念→词 (始终安全)
                    words = [w for w in jieba.lcut(human_text)
                            if len(w.strip()) >= 1]
                    self._learn_concept_word(concept_vec, words)
            except Exception:
                pass

        return (n_human, n_agent)

    def learn_sentence_online(self, text: str, text_vec: np.ndarray = None):
        """v5.7 纯净模式: 将对话中的句子增量存入句集群网络。

        Args:
            text: 句子文本
            text_vec: 预编码的文本向量 (64,), 可选
        """
        from cns.data_types import D, Cluster

        if self._sentence_net is None:
            # 懒初始化空句网络
            from cns.data_types import Theta
            theta = Theta(); theta.cluster_threshold = -1.0
            self._sentence_net = ClusterNetwork(theta)
            self._sentence_net.clusters = []
            self._sentence_net.buckets = {}

        # 编码
        if text_vec is None and self._text_env is not None:
            try:
                text_vec = self._text_env.encode_text(text)
            except Exception:
                text_vec = np.zeros(64, dtype=np.float32)
        if text_vec is None:
            text_vec = np.zeros(64, dtype=np.float32)

        padded = np.zeros(D, dtype=np.float32)
        padded[:64] = text_vec[:64].astype(np.float32)
        padded = np.tanh(padded + 1e-8).astype(np.float32)

        h = self._sentence_net.hash_features(padded)
        c = Cluster(centroid=h.copy())
        c.count = 1
        c.activation = 0.3
        c.age = len(self._sentence_net.clusters)
        self._sentence_net.clusters.append(c)
        hash_key = self._sentence_net._hash_to_bucket(h)
        self._sentence_net.buckets.setdefault(hash_key, []).append(c)
        self.sentences.append(text)
        self._cluster_to_sentence[id(c)] = len(self.sentences) - 1

    def next_word(self, prev1_vec: np.ndarray,
                  prev2_vec: np.ndarray = None) -> np.ndarray | None:
        """trigram 词预测: 给出前两个词，预测第三个词。

        查询 centroid[:128] = [prev1 | prev2]，匹配后返回 centroid[128:192]。
        如果只给一个词 (prev2=None) → bigram 回退 (mask 仅 [:64])。
        """
        if self.word_order_net.n_clusters == 0:
            return None

        from cns.data_types import D

        query = np.zeros(D, dtype=np.float32)
        mask = np.zeros(D, dtype=bool)

        if prev2_vec is not None:
            # trigram: 两个前词
            query[:64] = prev1_vec[:64]
            query[64:128] = prev2_vec[:64]
            mask[:128] = True
        else:
            # bigram 回退: 一个前词
            query[:64] = prev1_vec[:64]
            mask[:64] = True

        c = self.word_order_net.recall(query, mask=mask)
        if c is not None:
            if prev2_vec is not None:
                return c.centroid[128:192].copy()
            else:
                return c.centroid[64:128].copy()
        return None

    # ================================================================
    # v4: Hebb 记忆检索 — 句子 = 集群, 检索 = recall()
    # ================================================================

    def _ensure_sent_clusters(self):
        """懒加载句子集群网络。

        每句话 → 一个 Cluster (centroid = hash_features(padded_embedding)).
        检索时: hash(query) → bucket → 桶内 _masked_cosine 竞争 → 胜出集群.
        不是全局余弦扫描, 是 Hebb 记忆检索.

        v5.7: 纯净模式下句子列表为空 → 跳过构建, 保持空网络.
        """
        if self._sentence_net is not None:
            return
        if len(self.sentences) == 0:
            # 纯净模式: 无语料, 跳过构建. speak_sentence() 会返回空.
            from cns.data_types import Theta
            theta = Theta(); theta.cluster_threshold = -1.0
            self._sentence_net = ClusterNetwork(theta)
            self._sentence_net.clusters = []
            self._sentence_net.buckets = {}
            return

        MAX_SENTENCES = 12000  # 生物容量上限 — 只保留最可记忆的句子

        import hashlib
        from cns.data_types import D, Cluster
        from cerebrum.limbic_system.hippocampus import ClusterNetwork, _masked_cosine

        base = os.path.dirname(__file__)
        cache_dir = os.path.join(base, '.cache'); os.makedirs(cache_dir, exist_ok=True)

        # 缓存键
        corpus_hash = hashlib.md5(
            ''.join(self.sentences[:100]).encode()).hexdigest()[:8]
        if self._text_env is not None:
            pca_hash = hashlib.md5(
                str(self._text_env.pca.components_.sum()).encode()).hexdigest()[:8]
            cache_path = os.path.join(
                cache_dir, f'sent_clusters_{MAX_SENTENCES}_{corpus_hash}_shared_{pca_hash}.npy')
        else:
            cache_path = os.path.join(
                cache_dir, f'sent_clusters_{MAX_SENTENCES}_{corpus_hash}.npy')
        index_path = cache_path.replace('.npy', '_idx.npy')

        # ---- 加载或构建 ----
        if os.path.exists(cache_path) and os.path.exists(index_path):
            _log.info(f"  Broca: loading {MAX_SENTENCES} sentence clusters "
                  f"(Hebb memory, capped)...")
            centroids = np.load(cache_path)
            indices = np.load(index_path)
        else:
            centroids, indices = self._build_sent_clusters(cache_path,
                                                          MAX_SENTENCES)

        # ---- 构建 ClusterNetwork ----
        from cns.data_types import Theta
        # K 设大: 每句话一个集群, 不合并
        theta = Theta()
        theta.cluster_threshold = -1.0  # recall 永不自动合并
        self._sentence_net = ClusterNetwork(theta)
        # 绕过 K 限制: 直接把 clusters 列表替换
        self._sentence_net.clusters = []
        self._sentence_net.buckets = {}

        for i, (centroid, sent_idx) in enumerate(zip(centroids, indices)):
            c = Cluster(centroid=centroid.astype(np.float32))
            c.age = i  # 用 age 记录原始序号 (稳定排序)
            self._sentence_net.clusters.append(c)
            hash_key = self._sentence_net._hash_to_bucket(centroid)
            self._sentence_net.buckets.setdefault(hash_key, []).append(c)
            self._cluster_to_sentence[id(c)] = int(sent_idx)

        n_buckets = len(self._sentence_net.buckets)
        avg_per_bucket = len(centroids) / max(n_buckets, 1)
        _log.info(f"  Broca: {len(centroids)} sentence clusters, "
              f"{n_buckets} buckets (~{avg_per_bucket:.0f}/bucket)")

    def _build_sent_clusters(self, cache_path, max_sentences=None):
        """编码句子 → hash_features → 缓存为集群质心 (v2: 可选容量上限)"""
        from cns.data_types import D
        import numpy as np

        n_total = len(self.sentences)

        # 选择要编码的句子索引
        if max_sentences is not None and n_total > max_sentences:
            _log.info(f"  Broca: selecting {max_sentences} memorable sentences "
                  f"from {n_total} total...")
            # 基于长度评分: 长句包含更多 trigram → 更"可记忆"
            scores = np.array([min(len(s), 50) for s in self.sentences],
                            dtype=np.float32)
            # v3: 多样性奖励 ∝ 分数范围, 不用硬编码常数
            # 均匀采样以保证多样性: 每 N 句中取 1 句获得奖励
            max_score = float(np.max(scores)) if len(scores) > 0 else 50.0
            diversity_bonus = max_score * 0.4  # 40% of max → 默认等价于 20.0
            stride = n_total // max_sentences
            for i in range(0, n_total, max(stride, 1)):
                scores[i] += diversity_bonus
            selected_indices = np.argsort(scores)[-max_sentences:][::-1]
            selected_indices = sorted(selected_indices.tolist())
            target_sents = [self.sentences[i] for i in selected_indices]
            _log.info(f"  Broca: selected {len(target_sents)} sentences "
                  f"(len range: {min(len(s) for s in target_sents)}-"
                  f"{max(len(s) for s in target_sents)})")
        else:
            selected_indices = list(range(n_total))
            target_sents = self.sentences

        # 编码
        if self._text_env is not None:
            shared_encoder = self._text_env._encoder
            shared_pca = self._text_env.pca
            _log.info(f"  Broca: encoding {len(target_sents)} sentences with shared PCA...")
            full = shared_encoder.encode(
                target_sents, show_progress_bar=True, batch_size=64)
            projected = shared_pca.transform(full).astype(np.float32)
        else:
            _log.info(f"  Broca: encoding {len(target_sents)} sentences (standalone PCA)...")
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
            full = encoder.encode(
                target_sents, show_progress_bar=True, batch_size=64)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(64, full.shape[1]))
            projected = pca.fit_transform(full).astype(np.float32)

        # 填充到 D=330 + hash_features
        n_sent = len(target_sents)
        centroids = np.zeros((n_sent, D), dtype=np.float32)
        centroids[:, :min(64, projected.shape[1])] = projected[:, :64]
        centroids = np.tanh(centroids + 1e-8).astype(np.float32)

        indices = np.array(selected_indices, dtype=np.int32)

        np.save(cache_path, centroids)
        index_path = cache_path.replace('.npy', '_idx.npy')
        np.save(index_path, indices)
        _log.info(f"  Broca: sentence clusters cached ({centroids.shape})")
        return centroids, indices

    # ================================================================
    # 整句检索 — Hebb 记忆检索版
    # ================================================================

    def speak_sentence(self, query_vec: np.ndarray,
                       temperature: float = 0.7,
                       top_k: int = 5) -> tuple[list[str], np.ndarray | None]:
        """Hebb 记忆检索 — 不是全局余弦扫描。

        1. 查询向量 hash → 桶索引 (O(1))
        2. 桶内 _masked_cosine 竞争 → top-k (O(bucket_size), ~200)
        3. 温度 softmax 采样 → 选中的集群 = "回忆起的句子"

        这就是 "说话 = 记忆检索": Agent 在 Hebb 网络中回忆与当前
        信念最相似的听过的话, 然后说出来。不是符号生成, 不是全局扫描。

        Args:
            query_vec: 查询向量 (sensory[:64] 或信念 centroid[:64])
            temperature: softmax 温度 (越低越确定, 越高越随机)
            top_k: 从桶内 top_k 中采样

        Returns:
            (words, audio) — words 是词列表, audio 是 numpy 数组
        """
        self._ensure_sent_clusters()

        from cns.data_types import D
        from cerebrum.limbic_system.hippocampus import _masked_cosine

        # 构建查询 — 仅语义通道有效
        q = np.zeros(D, dtype=np.float32)
        q[:64] = query_vec[:64].astype(np.float32)
        h = self._sentence_net.hash_features(q)

        # ---- Hebb 检索: hash → bucket → 竞争 ----
        hash_key = self._sentence_net._hash_to_bucket(h)
        bucket = self._sentence_net.buckets.get(hash_key, [])

        if not bucket:
            # 极端情况: 空桶 → 回退到全集群扫描 (极少发生)
            bucket = self._sentence_net.clusters

        # 桶内竞争: 仅在哈希命中的桶内比较 (真正的 Hebb 检索)
        mask = np.zeros(D, dtype=bool)
        mask[:64] = True
        scored = []
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            scored.append((sim, c))

        # Top-k + 温度采样 (仅在桶内)
        scored.sort(key=lambda x: x[0], reverse=True)
        k_effective = min(top_k, len(scored))
        top = scored[:k_effective]

        top_sims = np.array([s for s, _ in top])
        eff_temp = max(temperature, 0.05)
        # 数值稳定: 减去 max
        probs = np.exp((top_sims - top_sims.max()) / eff_temp)
        probs /= probs.sum()
        chosen_cluster = top[int(np.random.choice(len(top), p=probs))][1]

        # ---- 集群 → 句子 ----
        sentence_idx = self._cluster_to_sentence[id(chosen_cluster)]
        sentence = self._clean_sentence(self.sentences[sentence_idx])

        import jieba
        words = [w for w in jieba.lcut(sentence) if w.strip()]

        audio = self._sentence_to_audio(words)
        return words, audio

    # ================================================================
    # v5: 词序 Hebb 链生成 — Agent "自己的话"
    # ================================================================

    def _next_words_topk(self, prev1_vec: np.ndarray,
                         prev2_vec: np.ndarray = None, k: int = 5
                         ) -> list[tuple[str, float, np.ndarray]]:
        """从词序 Hebb 网络获取 top-k 候选下一个词。

        v6: trigram 支持。提供两个前词 → 在 trigram 网络中匹配 centroid[:128]。
        如果只给一个前词 (prev2=None) → bigram 回退 (mask 仅 [:64])。

        Args:
            prev1_vec: 前一个词的向量 (64-dim, 音频频谱空间)
            prev2_vec: 再前一个词的向量 (可选; None → bigram 回退)
            k: 返回的候选数

        Returns:
            [(word_string, cosine_score, word_vector_64), ...]
        """
        from cns.data_types import D
        from cerebrum.limbic_system.hippocampus import _masked_cosine

        query = np.zeros(D, dtype=np.float32)
        mask = np.zeros(D, dtype=bool)

        if prev2_vec is not None:
            # Trigram: 两个前词 → 匹配 centroid[:128]
            query[:64] = prev1_vec[:64].astype(np.float32)
            query[64:128] = prev2_vec[:64].astype(np.float32)
            mask[:128] = True
            next_offset = 128  # 预测词在 centroid[128:192]
        else:
            # Bigram 回退: 一个前词 → 匹配 centroid[:64]
            query[:64] = prev1_vec[:64].astype(np.float32)
            mask[:64] = True
            next_offset = 64   # 预测词在 centroid[64:128]

        h = self.word_order_net.hash_features(query)
        hash_key = self.word_order_net._hash_to_bucket(h)
        bucket = self.word_order_net.buckets.get(hash_key, [])

        scored: list[tuple[str, float, np.ndarray]] = []
        seen_words: set[str] = set()

        # 桶内搜索
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            next_vec = c.centroid[next_offset:next_offset + 64].copy()
            word = self._vec_to_word(next_vec)
            if word and word not in seen_words:
                seen_words.add(word)
                scored.append((word, float(sim), next_vec))

        # 桶内无结果 → bigram 回退 (只用 prev1)
        if not scored and prev2_vec is not None:
            return self._next_words_topk(prev1_vec, None, k)

        # 仍无结果 → 全局簇搜索
        if not scored and self.word_order_net.n_clusters > 0:
            for c in self.word_order_net.clusters:
                sim = _masked_cosine(h, c.centroid, mask)
                next_vec = c.centroid[next_offset:next_offset + 64].copy()
                word = self._vec_to_word(next_vec)
                if word and word not in seen_words:
                    seen_words.add(word)
                    scored.append((word, float(sim), next_vec))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ================================================================
    # v3: 概念→词 Hebb 联想网络 — "我想表达 X → 用什么词"
    # ================================================================

    def _ensure_concept_word_net(self):
        """懒加载概念→词 Hebb 联想网络 (v3: 批量编码 + 缓存)。

        每个 cluster: centroid[:64] = 概念向量 (文本 PCA 空间)
                      centroid[64:128] = 关联词向量 (共现 SVD 空间)

        检索: query[:64] = concept → recall(mask[:64]) → 返回关联词向量。
        """
        if self._concept_word_built:
            return

        # v5.7 纯净模式: 无语料 → 跳过构建, 保持空网络
        if len(self.sentences) == 0:
            self._concept_word_built = True
            return

        import jieba, os, pickle
        from cns.data_types import D, Theta, Cluster

        base = os.path.dirname(__file__)
        cache_dir = os.path.join(base, '.cache'); os.makedirs(cache_dir, exist_ok=True)

        # 缓存键
        import hashlib
        corpus_sig = hashlib.md5(
            ''.join(self.sentences[:100]).encode()).hexdigest()[:8]
        cache_path = os.path.join(
            cache_dir, f'concept_word_net_{corpus_sig}.npy')
        words_cache = cache_path.replace('.npy', '_words.npy')

        # ---- 缓存命中 ----
        if os.path.exists(cache_path) and os.path.exists(words_cache):
            _log.info(f"  Loading concept→word Hebb network...")
            centroids = np.load(cache_path)
            word_indices = np.load(words_cache)

            for i in range(len(centroids)):
                c = Cluster(centroid=centroids[i].astype(np.float32))
                c.count = 1; c.activation = 0.3
                self.concept_word_net.clusters.append(c)
                hash_key = self.concept_word_net._hash_to_bucket(centroids[i])
                self.concept_word_net.buckets.setdefault(hash_key, []).append(c)

            self._concept_word_built = True
            _log.info(f"  Concept→word net: {len(centroids)} associations (cached)")
            return

        # ---- 构建 ----
        _log.info(f"  Building concept→word Hebb network...")
        self._ensure_sent_clusters()

        # 收集要编码的句子 (去重)
        sent_indices = [self._cluster_to_sentence.get(id(c), 0)
                       for c in self._sentence_net.clusters[:12000]]
        seen_si = set()
        unique_si = []
        for si in sent_indices:
            if si not in seen_si and si < len(self.sentences):
                seen_si.add(si)
                unique_si.append(si)

        target_sents = [self._clean_sentence(self.sentences[si])
                       for si in unique_si]

        # ---- 批量编码 (关键优化: 一次编码全部句子) ----
        if self._text_env is not None:
            concept_vecs = self._text_env.encode_batch(target_sents)
        else:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer('all-MiniLM-L6-v2')
            full = _encoder.encode(target_sents, show_progress_bar=True,
                                   batch_size=128)
            concept_vecs = np.array(full, dtype=np.float32)

        # ---- 构建集群 ----
        centroids_list = []
        word_idx_list = []

        for si_idx, sent in enumerate(target_sents):
            concept_vec = concept_vecs[si_idx]
            words = [w for w in jieba.lcut(sent) if len(w.strip()) >= 1]
            if not words:
                continue

            for wi, word in enumerate(words[:4]):  # 每句最多 4 个词
                wv = self._word_to_vec(word)
                if wv is None:
                    continue

                padded = np.zeros(D, dtype=np.float32)
                padded[:64] = concept_vec[:64].astype(np.float32)
                padded[64:128] = wv[:64].astype(np.float32)

                h = self.concept_word_net.hash_features(padded)
                centroids_list.append(h.astype(np.float32))
                word_idx_list.append(-1)  # placeholder (词向量已嵌入 centroid)

        # ---- 插入集群 + 缓存 ----
        centroids_arr = np.array(centroids_list, dtype=np.float32)
        word_idx_arr = np.array(word_idx_list, dtype=np.int32)

        for i in range(len(centroids_arr)):
            c = Cluster(centroid=centroids_arr[i])
            c.count = 1; c.activation = 0.3
            self.concept_word_net.clusters.append(c)
            hash_key = self.concept_word_net._hash_to_bucket(centroids_arr[i])
            self.concept_word_net.buckets.setdefault(hash_key, []).append(c)

        np.save(cache_path, centroids_arr)
        np.save(words_cache, word_idx_arr)

        self._concept_word_built = True
        _log.info(f"  Concept→word net: {len(centroids_arr)} associations "
              f"({len(unique_si)} sentences, cached)")

    def _learn_concept_word(self, concept_vec: np.ndarray,
                           words: list[str]):
        """增量学习: 单次概念→词关联 (用于对话中的在线学习)。

        当 Agent 生成回应后, 将 (理解向量, 回应词) 存入联想网络,
        使未来的概念→词映射随互动逐渐适应。
        """
        from cns.data_types import D, Cluster

        for word in words[:4]:  # 最多前 4 个词
            wv = self._word_to_vec(word)
            if wv is None:
                continue

            padded = np.zeros(D, dtype=np.float32)
            padded[:64] = concept_vec[:64].astype(np.float32)
            padded[64:128] = wv[:64].astype(np.float32)

            h = self.concept_word_net.hash_features(padded)

            # 检查是否存在 (用 Hebb recall)
            mask = np.zeros(D, dtype=bool)
            mask[:128] = True
            existing = self.concept_word_net.recall(padded, mask=mask)
            if existing is not None:
                existing.count += 1
                existing.activation = min(1.0, existing.activation + 0.05)
            else:
                c = Cluster(centroid=h.copy())
                c.count = 1
                c.activation = 0.3
                self.concept_word_net.clusters.append(c)
                hash_key = self.concept_word_net._hash_to_bucket(h)
                self.concept_word_net.buckets.setdefault(hash_key, []).append(c)

    def _find_seed_words(self, concept_vec: np.ndarray, k: int = 5
                         ) -> list[tuple[str, float]]:
        """v3: Hebb 概念→词联想检索 — 替代语料桥。

        直接从 concept_word_net 检索与概念关联的词:
        1. 概念向量 hash → 桶索引 (O(1))
        2. 桶内 mask[:64] 竞争 → top-k 匹配的概念→词集群
        3. centroid[64:128] → 最近词汇表词 → 返回种子词

        这不是语料桥——概念和词的关联是通过 Hebb 学习建立的
        (暖机时: 概念向量=句子编码, 关联词=该句分词)。
        """
        self._ensure_concept_word_net()

        from cns.data_types import D
        from cerebrum.limbic_system.hippocampus import _masked_cosine

        if self.concept_word_net.n_clusters == 0:
            return self._find_seed_words_fallback(concept_vec, k)

        q = np.zeros(D, dtype=np.float32)
        q[:64] = concept_vec[:64].astype(np.float32)
        h = self.concept_word_net.hash_features(q)
        hash_key = self.concept_word_net._hash_to_bucket(h)
        bucket = self.concept_word_net.buckets.get(hash_key, [])

        if not bucket:
            bucket = self.concept_word_net.clusters[:min(500,
                len(self.concept_word_net.clusters))]

        mask = np.zeros(D, dtype=bool)
        mask[:64] = True  # 只在概念空间比较

        word_scores: dict[str, float] = {}
        for c in bucket:
            sim = _masked_cosine(h, c.centroid, mask)
            word_vec = c.centroid[64:128].copy()  # 关联词向量
            word = self._vec_to_word(word_vec)
            if word and word not in word_scores:
                word_scores[word] = float(sim)
            elif word and float(sim) > word_scores[word]:
                word_scores[word] = float(sim)

        result = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return result[:k] if result else self._find_seed_words_fallback(concept_vec, k)

    def _find_seed_words_fallback(self, concept_vec, k):
        """回退: 无概念→词网络时的全局词搜索 (极少触发)"""
        from cns.data_types import D
        sims = np.dot(self.word_vecs, concept_vec[:64]) / (
            self.word_norms * np.linalg.norm(concept_vec[:64]) + 1e-8)
        top_indices = np.argsort(sims)[-k * 3:][::-1]
        seen = set()
        result = []
        for idx in top_indices:
            w = self.word_list[int(idx)]
            if w not in seen:
                seen.add(w)
                result.append((w, float(sims[int(idx)])))
        return result[:k]

    # ================================================================
    # v5.7: 婴儿模仿模式 — 空网络时的连贯回应
    # ================================================================

    # 功能词 (语法粒子) — 用于模仿时添加自然语气
    _GRAMMAR_PARTICLES = {'的', '了', '呢', '吗', '吧', '啊', '呀', '哦',
                          '嗯', '是', '在', '有', '和', '也', '都', '就',
                          '会', '要', '能', '很', '太', '不', '没'}

    # 重要的单字内容词 (模仿时保留)
    _IMPORTANT_1CHAR = {'你', '我', '他', '她', '它', '好', '爱', '想',
                        '说', '看', '吃', '去', '来', '做', '写', '读',
                        '听', '走', '跑', '笑', '哭', '大', '小', '多',
                        '少', '新', '旧', '快', '慢', '高', '低', '热',
                        '冷', '亮', '暗', '甜', '苦', '花', '猫', '狗',
                        '鸟', '鱼', '树', '草', '水', '火', '天', '地',
                        '人', '心', '手', '头', '门', '路', '车', '书'}

    def _extract_content_words(self, text: str, max_n: int = 12
                               ) -> list[str]:
        """从文本中提取内容词, 过滤纯功能词.

        模仿模式的核心 — 从人类输入中提取有意义的内容词,
        保留原序, 用于构建连贯的模仿回应.

        Args:
            text: 输入文本
            max_n: 最多提取词数

        Returns:
            内容词列表 (保持原序)
        """
        if not text or not text.strip():
            return []

        import jieba
        words = [w for w in jieba.lcut(text.strip()) if len(w.strip()) >= 1]
        if not words:
            return []

        # 内容词过滤: ≥2字 或 在重要单字集合中
        content = []
        for w in words:
            w_clean = w.strip()
            if not w_clean:
                continue
            if len(w_clean) >= 2:
                content.append(w_clean)
            elif w_clean in self._IMPORTANT_1CHAR:
                content.append(w_clean)
            # 跳过单字功能词 (保留在语法粒子集合中的)
            elif w_clean in self._GRAMMAR_PARTICLES and len(content) > 0:
                # 功能词只在已有内容词后才偶尔保留 (避免纯功能词回应)
                pass

        # 如果全是功能词 → 保留全部
        if not content:
            content = [w for w in words if len(w.strip()) >= 1][:max_n]

        return content[:max_n]

    def _speak_imitate(self, human_text: str, query_vec: np.ndarray = None,
                       valence: float = 0.0, arousal: float = 0.0,
                       max_words: int = 10) -> list[str]:
        """v5.7: 婴儿模仿模式 — 用人类输入的内容词构建回应.

        当 trigram 网络为空或极稀疏时, 不使用词向量相似度随机选词
        (会产生胡话), 而是从人类输入中提取内容词并回响.

        这是人类婴儿学习语言的方式 — 父母说 → 婴儿模仿关键音节.

        模仿策略 (按trigram网络成熟度分级):
          - 0 trigrams: 纯回响 — 保留原序, 截断至 max_words
          - <50 trigrams: 轻度重排 — 内容词+少量语法粒子, 微调顺序
          - 50+: 混合模式 — trigram链+模仿回退

        Args:
            human_text: 原始人类输入文本
            query_vec: 编码的查询向量 (用于情感调制)
            valence: 当前效价
            arousal: 当前唤醒
            max_words: 最多输出的词数

        Returns:
            词列表
        """
        content = self._extract_content_words(human_text, max_n=12)
        if not content:
            # 无内容词 → 使用最小安全回应
            if valence > 0.2:
                return ["嗯", "好"]
            elif valence < -0.2:
                return ["嗯"]
            else:
                return ["嗯"]

        n_tri = self.word_order_net.n_clusters

        if n_tri == 0:
            # === 纯模仿 (0 trigrams): 回响内容词 ===
            result = content[:max_words]

            # 添加简单的情感粒子
            if len(result) > 0:
                if valence > 0.3:
                    if result[-1] not in ('呢', '呀', '啊', '吧'):
                        result.append('呢')
                elif valence < -0.2:
                    pass  # 负面时保持简洁

            return result

        elif n_tri < 50:
            # === 轻度模仿 (<50 trigrams): 内容词 + 情感调制 ===
            result = content[:max(4, max_words - 2)]

            # 根据情感替换或添加词
            if valence > 0.4 and len(result) < max_words:
                if '好' not in result:
                    result.insert(0, '好')
            elif valence < -0.3 and len(result) < max_words:
                if '不' not in result[-2:]:
                    idx = max(0, len(result) - 1)
                    result.insert(idx, '不')

            return result[:max_words]

        else:
            # === 混合模式 (50+ trigrams): 模仿+网络 ===
            # 从内容词中选种子, 让 trigram 链扩展
            if len(content) >= 2:
                seed1 = content[0]
                seed2 = content[1] if len(content) > 1 else content[0]
                seed1_vec = self._word_to_vec(seed1)
                seed2_vec = self._word_to_vec(seed2)

                if seed1_vec is not None and seed2_vec is not None:
                    # 尝试 trigram 扩展
                    candidates = self._next_words_topk(seed1_vec, seed2_vec, k=8)
                    if candidates:
                        # trigram 有候选 → 混合模式
                        result = content[:3]  # 以人类的内容词开头
                        for word, score, _ in candidates[:3]:
                            if word not in result:
                                result.append(word)
                        return result[:max_words]

            # trigram 无候选 → 回退到纯模仿
            return content[:max_words]

    def speak_from_state(self,
                         belief_vec: np.ndarray,
                         body_state,
                         query_vec: np.ndarray,
                         valence: float = 0.0,
                         arousal: float = 0.0,
                         max_words: int = 18,
                         temperature: float = 0.7,
                         anti_repeat_window: int = 8,
                         phrase_network = None,
                         phrase_strength: float = 0.3,
                         human_text: str = None,
                         ) -> tuple[list[str], np.ndarray | None]:
        """Agent 用"自己的话"说话——词序 Hebb 链逐词生成 (v5.7: 短语结构约束).

        与 speak_sentence() 的根本区别:
        ┌──────────────────────┬──────────────────────────────────┐
        │ speak_sentence()     │ speak_from_state()               │
        ├──────────────────────┼──────────────────────────────────┤
        │ Hebb 检索整句语料     │ 信念→种子词→词序Hebb链→新句       │
        │ 原样输出              │ 逐词生成, 语法来自统计, 语义来自信念 │
        │ 像"引用别人说过的话"   │ 像"用自己的话说"                   │
        └──────────────────────┴──────────────────────────────────┘

        流程:
        0. (v5.7) 网络为空 → 婴儿模仿模式: 提取人类输入内容词回响
        1. 信念 + 人类输入混合 → 概念向量
        2. 概念向量 → 最近语料句 → 提取种子词 (corpus bridge)
        3. 种子词 → word_order_net._next_words_topk() → 逐词采样
        4. Valence/Arousal/Body 调制温度、长度、词偏好
        5. 反重复: 最近用过的词惩罚

        Args:
            belief_vec: Agent 的信念向量 (最激活集群的 centroid)
            body_state: BodyVector — 身体稳态状态
            query_vec: 感觉/人类输入的语义向量
            valence: 当前效价 [-1, 1]
            arousal: 当前唤醒度 [0, 1]
            max_words: 最大词数
            temperature: 基础温度 (越高越随机)
            anti_repeat_window: 反重复窗口大小
            human_text: 原始人类输入文本 (v5.7: 用于婴儿模仿模式)

        Returns:
            (words, audio) — words 是词列表, audio 是 numpy 数组或 None
        """
        import numpy as np

        # ---- Step 0: v5.7 婴儿模仿模式 — trigram网络为空时的连贯回应 ----
        # 当网络从零开始时, 词向量相似度会选出语义相关但语法上
        # 无意义的词序列 (胡话). 模仿人类输入的内容词 → 连贯回应,
        # 同时通过 learn_from_interaction() 积累 trigram 网络.
        n_tri = self.word_order_net.n_clusters
        is_clean_mode = (n_tri == 0
                        and (self._sentence_net is None
                             or len(self._sentence_net.clusters) == 0))

        if is_clean_mode and human_text and human_text.strip():
            # 纯模仿: 回响人类输入的内容词 (像婴儿学语)
            words = self._speak_imitate(
                human_text, query_vec, valence, arousal, max_words)
            if words:
                audio = self._sentence_to_audio(words)
                return words, audio
        elif is_clean_mode and not human_text:
            # v5.7: 无人类输入+空网络 (如内部言语) → 安全默认回应
            # 避免词向量相似度产生胡话
            if valence > 0.2:
                return (["嗯", "好"],
                        self._sentence_to_audio(["嗯", "好"]))
            else:
                return (["嗯"],
                        self._sentence_to_audio(["嗯"]))
        elif n_tri < 50 and human_text and human_text.strip():
            # 轻度模仿: trigram极少时仍用模仿, 确保连贯性
            words = self._speak_imitate(
                human_text, query_vec, valence, arousal, max_words)
            if words:
                audio = self._sentence_to_audio(words)
                return words, audio

        # ---- Step 1: 混合概念 (v3: 身体状态驱动, 不用硬编码系数) ----
        # F_body 高 → 更关注自身需求 → 信念主导 (内生)
        # F_body 低 → 更关注外界 → 输入主导 (外生)
        # belief_weight ∈ [0.3, 0.8]
        body_dev = float(body_state.compute_deviation())
        belief_weight = 0.3 + min(0.5, body_dev * 0.8)  # 偏离大 → 自我聚焦
        sensory_weight = 1.0 - belief_weight
        concept = (belief_vec[:64].astype(np.float32) * belief_weight
                   + query_vec[:64].astype(np.float32) * sensory_weight)

        # ---- Step 2: 找种子词 ----
        seed_candidates = self._find_seed_words(concept, k=6)

        if not seed_candidates:
            # 回退: 整句检索
            return self.speak_sentence(query_vec, temperature, top_k=3)

        # 温度采样选种子
        seed_scores = np.array([s for _, s in seed_candidates], dtype=np.float32)
        seed_probs = np.exp((seed_scores - seed_scores.max())
                            / max(temperature, 0.05))
        seed_probs /= seed_probs.sum()
        chosen_idx = int(np.random.choice(len(seed_candidates), p=seed_probs))
        seed_word = seed_candidates[chosen_idx][0]

        # ---- Step 3: 词序 Hebb 链 (trigram) ----
        seed1_vec = self._word_to_vec(seed_word)
        if seed1_vec is None:
            return self.speak_sentence(query_vec, temperature, top_k=3)

        # 找第二个种子词: trigram 需要两个前词来预测第三个
        seed2_candidates = self._next_words_topk(seed1_vec, None, k=5)
        seed2_word = seed2_candidates[0][0] if seed2_candidates else seed_word
        seed2_vec = self._word_to_vec(seed2_word)

        words = [seed_word, seed2_word] if seed2_word != seed_word else [seed_word]
        recent_words: list[str] = words[:]

        # 长度调制
        base_len = 5
        arousal_bonus = int(arousal * 8)
        valence_mod = 1.0 if valence > 0 else 0.5
        target_len = base_len + int(arousal_bonus * valence_mod)
        target_len = max(4, min(target_len, max_words))

        social_need = max(0.0, float(body_state.setpoints[0] - body_state.b[0]))
        if social_need > 0.3:
            target_len = min(target_len + 4, max_words)

        chain_temp = temperature * (0.6 + arousal * 0.8)

        # v5.7: 短语结构追踪 (用于 modulate_candidates)
        current_phrase_words: list[str] = words[:]  # 当前短语中的词
        phrase_active = phrase_network is not None and phrase_network._trained

        # v3: 无手写终结词 — 自然终止条件:
        # 1. trigram 网络找不到后继 (候选为空或相似度太低)
        # 2. 已达最大长度
        # 语法终结从 trigram 统计中涌现: 高频终结词自然很少有后继 trigram
        TERMINATE_SIM_THRESHOLD = 0.02  # 低于此相似度的候选视为无效

        # trigram 链: 用最后两个词预测下一个
        for step in range(target_len - len(words)):
            if len(words) >= 2:
                # trigram: 两个前词 → 更精确的预测
                p1_vec = self._word_to_vec(words[-2])
                p2_vec = self._word_to_vec(words[-1])
                if p1_vec is not None and p2_vec is not None:
                    candidates = self._next_words_topk(p1_vec, p2_vec, k=10)
                else:
                    candidates = self._next_words_topk(seed1_vec, None, k=10)
            elif len(words) == 1:
                # bigram 回退: 只有一个词
                p1_vec = self._word_to_vec(words[-1])
                candidates = self._next_words_topk(p1_vec, None, k=10) if p1_vec is not None else []
            else:
                break

            if not candidates:
                # trigram 失败 → bigram 回退
                if len(words) >= 1:
                    p1_vec = self._word_to_vec(words[-1])
                    if p1_vec is not None:
                        candidates = self._next_words_topk(p1_vec, None, k=10)
                if not candidates:
                    break

            # v5.7: 短语结构约束 — 调制候选词得分
            if phrase_active:
                candidates = phrase_network.modulate_candidates(
                    candidates, current_phrase_words, phrase_strength)

            # 反重复: 最近用过的词大幅降权
            adjusted: list[tuple[str, float, np.ndarray]] = []
            for word, score, vec in candidates:
                if word in recent_words[-anti_repeat_window:]:
                    penalty = 3.0 if word == recent_words[-1] else 1.5
                    score -= penalty
                adjusted.append((word, score, vec))

            adjusted.sort(key=lambda x: x[1], reverse=True)
            top = adjusted[:max(5, min(8, len(adjusted)))]

            # 温度 softmax 采样
            top_scores = np.array([s for _, s, _ in top], dtype=np.float32)
            probs = np.exp((top_scores - top_scores.max())
                           / max(chain_temp, 0.05))
            probs /= probs.sum()

            chosen_idx_step = int(np.random.choice(len(top), p=probs))
            chosen_word, _, next_vec = top[chosen_idx_step]

            # v3: 自然终止 — 无手写终结词集
            # 如果最佳候选相似度低于阈值 → trigram 统计中这个词组合
            # 极少出现 → 自然的句法边界 → 概率终止
            best_sim = top[0][1]
            if best_sim < TERMINATE_SIM_THRESHOLD and len(words) >= 3:
                if np.random.random() < 0.5:
                    break

            # 候选质量下降 + 已有足够长度 → 终止概率上升
            if best_sim < 0.05 and len(words) >= 6:
                if np.random.random() < 0.3:
                    break

            words.append(chosen_word)
            recent_words.append(chosen_word)
            if len(recent_words) > anti_repeat_window * 2:
                recent_words.pop(0)

            # v5.7: 短语结构追踪 — 检测短语边界
            if phrase_active:
                current_phrase_words.append(chosen_word)
                closure = phrase_network.phrase_closure_probability(
                    current_phrase_words)
                # 短语结束概率高 → 开启新短语
                if closure > 0.5 and len(current_phrase_words) >= 2:
                    current_phrase_words = []  # 重置为新短语

        # ---- Step 4: 后处理 + 混合回退 ----
        sentence = ''.join(words)

        # v5.7 纯净模式: 网络为空时用词向量直接选择 (无语料回退)
        if self.word_order_net.n_clusters == 0 and self._sentence_net is None:
            # 无预训练网络 → 用词向量语义相似度直接选择多个相关词
            if len(words) < max(3, target_len // 2):
                extra = self._find_seed_words_fallback(concept, k=12)
                seen = set(words)
                for word, score in extra:
                    if word not in seen and len(words) < target_len:
                        seen.add(word)
                        words.append(word)
                sentence = ''.join(words)

        # 4a. 太短 → 语料句补全 (仅当语料网络可用时)
        if len(words) < 4 and self._sentence_net is not None:
            # 检索最相关的语料句
            fallback_words, _ = self.speak_sentence(query_vec, temperature=0.3, top_k=3)
            if fallback_words and len(fallback_words) > len(words):
                import jieba
                fb_sentence = ''.join(fallback_words)
                fb_parts = [w for w in jieba.lcut(fb_sentence) if w.strip()]
                fragment_len = min(len(fb_parts), max(3, min(6, target_len // 2)))
                fragment = fb_parts[:fragment_len]
                if fragment != words[:len(fragment)]:
                    words = fragment
                    sentence = ''.join(words)
                    # 尝试从骨架末尾继续词序链
                    if len(words) >= 2:
                        p1 = self._word_to_vec(words[-2])
                        p2 = self._word_to_vec(words[-1])
                        if p1 is not None and p2 is not None:
                            cv = p2
                            for _ in range(min(5, target_len - len(words))):
                                cands = self._next_words_topk(p1, p2, k=8)
                                if not cands:
                                    cands = self._next_words_topk(cv, None, k=8)
                                if not cands:
                                    break
                                added = False
                                for cw, cs, nv in cands:
                                    if cw not in words[-3:]:
                                        words.append(cw)
                                        p1, p2 = p2, nv
                                        cv = nv
                                        added = True
                                        break
                                if not added:
                                    break
                            sentence = ''.join(words)

        # 4b. 仍太短 → 最终回退
        if len(words) < 2:
            if self._sentence_net is not None:
                fallback_words, _ = self.speak_sentence(query_vec, temperature=0.3, top_k=2)
                if fallback_words and len(fallback_words) >= 2:
                    half = max(2, len(fallback_words) // 2 + 1)
                    words = fallback_words[:half]
                    sentence = ''.join(words)
                else:
                    # v5.7: 语义回退 — 取与query最相似的3-5个词
                    extra = self._find_seed_words_fallback(concept, k=5)
                    words = [w for w, _ in extra[:4]] if extra else ["嗯"]
                    sentence = ''.join(words)
            else:
                # v5.7 纯净模式: 语义回退
                extra = self._find_seed_words_fallback(concept, k=5)
                words = [w for w, _ in extra[:4]] if extra else ["嗯"]
                sentence = ''.join(words)

        import jieba
        final_words = [w for w in jieba.lcut(sentence) if w.strip()]

        # v3: Hebb 在线学习 — 强化 (concept, generated_words) 关联
        # 这次检索成功 → 加强这个映射, 使未来类似概念更容易找到这些词
        if len(final_words) >= 2:
            self._learn_concept_word(concept, final_words)

        audio = self._sentence_to_audio(final_words)
        return final_words, audio

    # ================================================================
    # 工具方法
    # ================================================================

    def _clean_sentence(self, sentence: str) -> str:
        """去掉角色名前缀 (格式: '角色名：台词' 或 '我/xxx：台词')

        v5.2: 前缀剥离后即使很短也保留内容，避免"格雷亚姆：哼"泄露人名。
        """
        cleaned = re.sub(
            r'^[一-鿿\w/·]{1,6}[：:]\s*', '', sentence)
        # 空内容 → 保留原句 (没有前缀可剥)
        if len(cleaned) == 0:
            return sentence
        # v5.2: 有内容就返回，不管多短。"哼" 比 "格雷亚姆：哼" 好
        return cleaned

    def _sentence_to_audio(self, words: list[str]) -> np.ndarray | None:
        """将词列表拼接为音频"""
        import soundfile as sf
        speaker = None
        try:
            from tools.word_speech import get_speaker
            speaker = get_speaker()
        except Exception:
            pass

        if speaker is None:
            return None

        word_audios = []
        for w in words:
            speaker.ensure_word_audio(w)
            wav_path = os.path.join(os.path.dirname(__file__), 'word_audio', f'{w}.wav')
            try:
                a, _ = sf.read(wav_path)
                word_audios.append(a)
            except Exception:
                word_audios.append(np.zeros(2205, dtype=np.float32))

        if not word_audios:
            return None
        s_short = np.zeros(int(0.08 * 22050), dtype=np.float32)
        full = word_audios[0]
        for a in word_audios[1:]:
            full = np.concatenate([full, s_short, a])
        return full

    def assemble(self, concept_vecs, agent_net, word_speaker):
        # 1. 概念→最近语料句→取实体词作种子
        if not hasattr(self, '_seed_env'):
            from environments.text_interface import TextEnvironment
            self._seed_env = TextEnvironment()
        import jieba
        seed_words = []
        for vec in concept_vecs:
            q_n = np.linalg.norm(vec[:64])
            sims = np.dot(self._seed_env.embeddings, vec[:64]) / (
                np.linalg.norm(self._seed_env.embeddings, axis=1) * q_n + 1e-8)
            sent = self._seed_env.chunks[int(np.argmax(sims))]
            ws = [w for w in jieba.lcut(sent) if len(w.strip()) >= 1]
            seed_words.append(ws[0] if ws else self._vec_to_word(vec[:64]))

        # 2. 词序链式生成 (反重复)
        all_words = [seed_words[0]]
        current_vec = self._word_to_vec(seed_words[0])
        if current_vec is None: current_vec = concept_vecs[0][:64]
        seen = {seed_words[0]}

        for seed_w in seed_words[1:]:
            sub = []
            for _ in range(4):
                nv = self.next_word(current_vec)
                if nv is None: break
                nw = self._vec_to_word(nv)
                if nw in seen: break
                seen.add(nw)
                sub.append(nw)
                current_vec = self._word_to_vec(nw)
                if current_vec is None: break
            all_words.extend(sub if sub else [seed_w])

        # 3. 内部监听
        if agent_net.n_clusters > 0 and len(''.join(all_words)) > 2:
            if not hasattr(self, '_monitor_env'):
                from environments.text_interface import TextEnvironment
                self._monitor_env = TextEnvironment()
            emb = self._monitor_env.encode_text(''.join(all_words))
            padded = np.zeros(330, dtype=np.float32); padded[:64] = emb
            if agent_net.recall(padded) is None:
                all_words = all_words[:max(2, len(all_words)//3)]

        # 4. 转音频
        word_audios, word_labels = [], []
        s_short = np.zeros(int(0.1*22050))
        for w in all_words:
            word_speaker.ensure_word_audio(w)
            try:
                import soundfile as sf
                a, _ = sf.read(os.path.join(os.path.dirname(__file__), 'word_audio', f'{w}.wav'))
            except Exception: a = np.zeros(2205)
            word_audios.append(a); word_labels.append(w)
        full = word_audios[0]
        for a in word_audios[1:]: full = np.concatenate([full, s_short, a])
        return word_labels, full
