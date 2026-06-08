"""
text_interface.py —— 文本环境 (Stage 2)
自由能原理智能体

替换 GridWorld。每步输出一句文本的嵌入作为 s。
嵌入用 sentence-transformer (384d → PCA 64d)。

Agent 的行动决定下一步读哪段文本。
每种行动产生可预测的不同程度 s 变化。
"""

import re
import numpy as np
from cns.data_types import D


class TextEnvironment:
    """文本环境 — 语料导航 (v2: 语义嵌入 + 有梯度的行动)

    每步: get_sensory() → s (D=138)
          step(action) → 移动光标，s 变化幅度取决于行动类型
    无外部奖励 — 纯 F 最小化驱动。

    行动梯度:
      A₀(前进1句) → s 变化小 (叙事相邻)
      A₁(后退5句) → s 变化中
      A₂(跳转)    → s 变化大 (跳到语义最远的簇)
      A₃(表达)    → s[96:104] 编码输出反馈 (闭合回路)
      A₄(观察)    → s 不变, body b₂ 恢复
    """

    def __init__(self, corpus: list[str] = None, embed_dim: int = 64,
                 use_semantic: bool = True, load_corpus: bool = True):
        import os, pickle
        self.embed_dim = embed_dim

        if corpus is not None:
            self.chunks = corpus
        elif load_corpus:
            self.chunks = _fetch_corpus()
        else:
            # v6.4: headless mode — skip corpus loading, only need encoder
            self.chunks = []

        self.n_chunks = len(self.chunks)
        self.cursor = 0
        self.steps = 0
        self.last_output_embedding = None  # A₃ 输出的嵌入
        self.last_output_text = ''          # A₃ 输出的文本

        if self.n_chunks > 0:
            print(f"  Corpus: {self.n_chunks} sentences")
        else:
            print(f"  TextEnvironment: headless mode (no corpus, encoder only)")

        if use_semantic and self.n_chunks > 0:
            cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
            os.makedirs(cache_dir, exist_ok=True)
            # 用 corpus 的 hash 做缓存键，避免语料变了还用旧缓存
            import hashlib
            corpus_hash = hashlib.md5(
                ''.join(self.chunks[:100]).encode()).hexdigest()[:8]
            emb_cache = os.path.join(
                cache_dir, f'embeddings_{self.n_chunks}_{corpus_hash}.npy')
            pca_cache = os.path.join(
                cache_dir, f'pca_{self.n_chunks}_{corpus_hash}.pkl')

            if os.path.exists(emb_cache) and os.path.exists(pca_cache):
                print(f"  Loading cached embeddings...")
                self.embeddings = np.load(emb_cache)
                with open(pca_cache, 'rb') as f:
                    self.pca = pickle.load(f)
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"  Cached: {self.embeddings.shape}")
            else:
                print(f"  Encoding {self.n_chunks} sentences (this may take a while)...")
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                self._encoder = model
                full = model.encode(
                    self.chunks, show_progress_bar=True,
                    batch_size=64,
                )  # (N, 384)
                from sklearn.decomposition import PCA
                n_pca = min(embed_dim, min(self.n_chunks, full.shape[1]))
                self.pca = PCA(n_components=n_pca)
                self.embeddings = self.pca.fit_transform(full).astype(np.float32)
                if n_pca < embed_dim:
                    padded = np.zeros((self.n_chunks, embed_dim), dtype=np.float32)
                    padded[:, :n_pca] = self.embeddings
                    self.embeddings = padded
                self.full_embeddings = full
                var = self.pca.explained_variance_ratio_.sum()
                print(f"  Semantic: 384d → {n_pca}d PCA (var={var:.1%})")
                # 缓存
                np.save(emb_cache, self.embeddings)
                with open(pca_cache, 'wb') as f:
                    pickle.dump(self.pca, f)
                print(f"  Cached to: {emb_cache}")
        elif use_semantic and self.n_chunks == 0:
            # v6.4: headless mode — no PCA, encoder loads lazily, raw embedding
            self.pca = None
            self.embeddings = np.zeros((0, embed_dim), dtype=np.float32)
            print(f"  TextEnvironment: encoder will load on first encode_text()")
        else:
            rng = np.random.default_rng(42)
            self.embeddings = rng.normal(
                0, 1.0 / np.sqrt(embed_dim),
                (self.n_chunks, embed_dim)).astype(np.float32)
            print(f"  Random projection: {embed_dim}d")

    # ---- 感知 ----
    def get_sensory(self, body: np.ndarray = None) -> np.ndarray:
        s = np.zeros(D)
        s[0:self.embed_dim] = self.embeddings[self.cursor]

        ctx_start = max(0, self.cursor - 3)
        if ctx_start < self.cursor:
            s[64:64 + self.embed_dim] = np.mean(
                self.embeddings[ctx_start:self.cursor], axis=0)

        # 输出反馈 [96:104] — A₃ "上次我说了什么"
        if self.last_output_embedding is not None:
            s[96:104] = self.last_output_embedding[:8]

        s[112] = np.sin(2 * np.pi * self.steps / 1000.0)
        s[113] = np.cos(2 * np.pi * self.steps / 1000.0)
        s[114] = self.steps / max(self.steps, 1)
        return s

    # ---- 行动 (v2: 有梯度的 s 变化) ----
    def step(self, action_idx: int) -> float:
        """执行行动。s 变化幅度取决于行动类型。

        A₀: +1句     → Δs 小   (相邻叙事 → 嵌入高度相似)
        A₁: −5句     → Δs 中   (段落内跳转)
        A₂: 语义跳转  → Δs 大   (跳到底部 5% 最不相似的句子)
        A₃: 表达     → Δs=0   (body b₀ 恢复)
        A₄: 观察     → Δs=0   (body b₂ 恢复)
        """
        if action_idx == 0:
            self.cursor = (self.cursor + 1) % self.n_chunks
        elif action_idx == 1:
            self.cursor = max(0, self.cursor - 5)
        elif action_idx == 2:
            cur_emb = self.embeddings[self.cursor]
            norms = np.linalg.norm(self.embeddings, axis=1)
            cur_norm = np.linalg.norm(cur_emb)
            sims = np.array([float(np.dot(cur_emb, e) / (cur_norm * n + 1e-8))
                            for e, n in zip(self.embeddings, norms)])
            sims[self.cursor] = 1.0  # 排除自己
            threshold = np.percentile(sims, 5)
            candidates = np.where(sims <= threshold)[0]
            if len(candidates) > 0:
                self.cursor = int(np.random.choice(candidates))
        elif action_idx == 3:  # A₃: 表达 — 输出最相似句
            cur_emb = self.embeddings[self.cursor]
            norms = np.linalg.norm(self.embeddings, axis=1)
            cur_norm = np.linalg.norm(cur_emb)
            sims = np.array([float(np.dot(cur_emb, e) / (cur_norm * n + 1e-8))
                            for e, n in zip(self.embeddings, norms)])
            sims[self.cursor] = -1.0
            top = int(np.argmax(sims))
            self.last_output_embedding = self.embeddings[top].copy()
            self.last_output_text = self.chunks[top]
        elif action_idx == 4:
            pass  # 观察 — body b₂ 恢复
        self.steps += 1
        return 0.0

    # ---- 摘要 ----
    def encode_text(self, text: str) -> np.ndarray:
        """任意句子 → 64 维语义向量 (复用已加载的 encoder)"""
        if not hasattr(self, '_encoder'):
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        full = self._encoder.encode([text])[0]  # (384,)
        if self.pca is not None:
            emb = self.pca.transform(full.reshape(1, -1))[0]
        else:
            # v6.4: headless mode — truncate/pad raw embedding to 64d
            emb = np.zeros(64, dtype=np.float32)
            n = min(64, len(full))
            emb[:n] = full[:n]
        if len(emb) < 64:
            p = np.zeros(64); p[:len(emb)] = emb; emb = p
        return emb.astype(float)

    def encode_batch(self, texts: list[str], batch_size: int = 128
                    ) -> np.ndarray:
        """批量编码 → (N, 64) 语义向量 (一次 transformer 调用)

        用于 Broca concept_word_net 构建等需要大量编码的场景。
        """
        if not hasattr(self, '_encoder'):
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        full = self._encoder.encode(texts, show_progress_bar=True,
                                    batch_size=batch_size)
        if self.pca is not None:
            emb = self.pca.transform(full).astype(np.float32)
        else:
            # v6.4: headless mode — truncate/pad raw embedding to 64d
            emb = np.zeros((full.shape[0], 64), dtype=np.float32)
            n = min(64, full.shape[1])
            emb[:, :n] = full[:, :n]
        if emb.shape[1] < 64:
            p = np.zeros((emb.shape[0], 64), dtype=np.float32)
            p[:, :emb.shape[1]] = emb
            emb = p
        return emb

    def get_state_summary(self) -> dict:
        return {
            'cursor': self.cursor, 'n_chunks': self.n_chunks,
            'steps': self.steps, 'current_text': self.chunks[self.cursor][:60],
        }


def _fetch_corpus() -> list[str]:
    """获取语料。优先从本地文件加载，回退到 requests 下载。"""
    import os

    local_path = os.path.join(os.path.dirname(__file__), 'corpus.txt')
    # Fallback to project root (monorepo layout)
    root_path = os.path.join(os.path.dirname(__file__), '..', 'corpus.txt')
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  Loaded corpus.txt ({len(text)} chars)")
    elif os.path.exists(root_path):
        with open(root_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  Loaded corpus.txt from root ({len(text)} chars)")
    else:
        # 尝试下载《小王子》中文全文
        try:
            import requests
            urls = [
                "https://raw.githubusercontent.com/nicksmrz13/the-little-prince-zh/main/book.txt",
                "https://gist.githubusercontent.com/ymotongpoo/8d4c4f3c6c8a3c1b2f8e/raw/little_prince_zh.txt",
            ]
            text = None
            for url in urls:
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200 and len(resp.text) > 1000:
                        text = resp.text
                        break
                except Exception:
                    continue
            if text is None:
                raise RuntimeError("Failed to download corpus")
            # 缓存到本地
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"  Downloaded corpus ({len(text)} chars) → saved to corpus.txt")
        except Exception:
            print("  Download failed, using built-in corpus")
            text = _builtin_corpus_text()

    sentences = re.split(r'[。！？\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    # 去重（大语料下非常必要）
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    print(f"  Unique sentences: {len(unique)} (from {len(sentences)} raw)")
    return unique[:50000]


def _builtin_corpus_text() -> str:
    return """我六岁的时候，在一本描写原始森林的书里看到一幅精美的插图。
书里画着一条蟒蛇正在吞食一头野兽。
我把我的第一幅画拿给大人们看，问他们怕不怕。
大人们回答我说：一顶帽子有什么可怕的。
于是我画了第二幅画，把蟒蛇的肚子剖开给他们看。
大人们劝我把剖开的和完整的蟒蛇画都放到一边。
还是把兴趣放在地理、历史、算术和语法上。
就这样，六岁那年，我放弃了画家生涯。
我只好选择另一种职业，学会了开飞机。
我差不多飞遍了世界各地。
地理确实帮了我的大忙。
我一眼就能分辨出中国和亚利桑那州。
如果夜里迷失方向，这些知识就很有用。
在我的一生中，我跟很多严肃的人打过交道。
我在大人们中间生活了很久，近距离观察过他们。
但这并没有怎么改变我对他们的看法。
每当我遇到一个稍微头脑清醒的大人，我就拿出我一直保存着的第一幅画做试验。
我想知道他是否真的有理解力。
可是，不管是谁，他或她总是说：这是一顶帽子。
于是我就不再跟他谈蟒蛇、原始森林或者星星了。
我会迁就他，跟他谈桥牌、高尔夫球、政治和领带。
这样大人们就会很高兴，觉得认识了一个通情达理的人。
我就这样孤独地生活着，没有一个真正谈得来的人。
直到六年前，我的飞机在撒哈拉沙漠出了故障。
发动机里有什么东西坏了。
当时我身边没有机械师，也没有乘客。
我只好独自完成这项困难的修理工作。
这关系到生死存亡：我带的水只够维持八天。
第一天晚上，我就在远离人烟的沙漠里睡着了。
我比大海中遇难的人还要孤独。
你可以想象第二天早上，当一个微弱而奇怪的声音叫醒我时，我是多么惊讶。
那个声音说：请你给我画一只绵羊。
什么？给我画一只绵羊。
我像被雷击中一样跳了起来。
我使劲揉了揉眼睛，仔细看了看。
我看见一个非常奇特的小人，正认真地打量着我。
这就是我后来画给他的最好的画像。
不过我的画当然远不如他本人可爱。
这不是我的错。
六岁时大人们就让我放弃了画家生涯。
除了剖开的和完整的蟒蛇，我没有学过画别的。
于是我惊讶地睁大眼睛看着这个突然出现的小家伙。
别忘了，我当时在远离人烟千里之外的沙漠里。
然而这个小家伙既不像是在沙漠里迷了路。
也完全不像是因疲劳、饥饿、干渴或恐惧而虚弱。
他丝毫看不出是一个在远离人烟的沙漠中迷路的孩子。
当我终于能开口说话时，我对他说：你在这里做什么。
他于是又重复了一遍，像是在说一件很重要的事：请你给我画一只绵羊。
当神秘太强烈时，我们就不敢违抗了。
尽管在远离人烟千里之外，面临着死亡的危险，我还是从口袋里掏出了一张纸和一支钢笔。"""
