"""
word_speech.py —— 词级离散组合语音输出
He扩散路径 → 词序列 → 拼接预存词音频
"""
import numpy as np, os


class WordSpeaker:
    def __init__(self):
        base = os.path.dirname(__file__)
        root = os.path.join(base, '..')  # project root fallback
        # 优先使用扩展词表 (12,000 词), 回退到原始词表 (543 词)
        expanded_path = os.path.join(base, 'word_spectrum_dataset_expanded.npy')
        original_path = os.path.join(base, 'word_spectrum_dataset.npy')
        if os.path.exists(expanded_path):
            raw = np.load(expanded_path, allow_pickle=True)
            print(f"  Using expanded vocabulary ({len(raw)} words)")
        elif os.path.exists(original_path):
            raw = np.load(original_path, allow_pickle=True)
        else:
            # Fallback to project root
            expanded_path = os.path.join(root, 'word_spectrum_dataset_expanded.npy')
            original_path = os.path.join(root, 'word_spectrum_dataset.npy')
            if os.path.exists(expanded_path):
                raw = np.load(expanded_path, allow_pickle=True)
            else:
                raw = np.load(original_path, allow_pickle=True)
        self.words = [r[2] for r in raw]
        self.word_vecs = np.stack([r[0] for r in raw])
        self.word_norms = np.linalg.norm(self.word_vecs, axis=1)

        self.audio_dir = os.path.join(base, 'word_audio')
        os.makedirs(self.audio_dir, exist_ok=True)
        # 懒生成: 只在实际用到这个词时才生成音频
        # 不再在启动时 eager-generate (扩展词表有 ~8K 词, edge-tts 太慢)
        self._missing_checked = False
        print(f"WordSpeaker: {len(self.words)} words, "
              f"audio dir: {self.audio_dir}")

    def ensure_word_audio(self, word: str):
        """确保某个词有音频（懒加载——缺词时即时生成）

        edge-tts 不可用时静默跳过 — 语音输出是可选的增强功能。
        """
        wav_path = os.path.join(self.audio_dir, f'{word}.wav')
        if os.path.exists(wav_path):
            return
        try:
            import asyncio, edge_tts, librosa, soundfile as sf
        except ImportError:
            return  # 静默跳过 — edge-tts 不可用
        async def _gen():
            mp3 = os.path.join(self.audio_dir, f'_tmp_{word}.mp3')
            await edge_tts.Communicate(word, 'zh-CN-XiaoxiaoNeural').save(mp3)
            audio, sr = librosa.load(mp3, sr=22050, mono=True)
            peak = np.abs(audio).max()
            if peak > 0: audio = audio / peak * 0.95
            sf.write(wav_path, audio, sr)
            os.remove(mp3)
        try:
            asyncio.run(_gen())
        except Exception:
            pass

    def _generate_missing_audio(self):
        import asyncio
        missing = [w for w in self.words
                   if not os.path.exists(os.path.join(self.audio_dir, f'{w}.wav'))]
        if not missing: return
        print(f"  Generating {len(missing)} word audio files...")
        asyncio.run(self._tts_batch(missing))

    async def _tts_batch(self, words):
        import edge_tts, librosa, soundfile as sf
        for i, w in enumerate(words):
            try:
                mp3 = os.path.join(self.audio_dir, f'_tmp_{w}.mp3')
                wav = os.path.join(self.audio_dir, f'{w}.wav')
                await edge_tts.Communicate(w, "zh-CN-XiaoxiaoNeural").save(mp3)
                audio, sr = librosa.load(mp3, sr=22050, mono=True)
                peak = np.abs(audio).max()
                if peak > 0: audio = audio / peak * 0.95
                sf.write(wav, audio, sr)
                os.remove(mp3)
                if (i+1) % 50 == 0: print(f"    {i+1}/{len(words)}")
            except Exception as e: print(f"    [{w}] skip: {e}")

    def concept_to_word(self, concept_vec_64):
        q_n = np.linalg.norm(concept_vec_64)
        sims = np.dot(self.word_vecs, concept_vec_64) / (self.word_norms * q_n + 1e-8)
        return int(np.argmax(sims))

    def speak_concept_path(self, path_centroids, output_path=None,
                          broca=None):
        """概念路径 → 语音输出 (v3: Hebb 检索, 可选回退)

        v3: 优先使用 Broca 的 Hebb 句子集群检索 (O(1) hash + 桶内竞争)。
        无 Broca 时回退到全局余弦扫描 (legacy, M3 gridworld 兼容)。

        Args:
            path_centroids: 概念质心路径列表
            output_path: 输出 wav 路径 (可选)
            broca: Broca 实例 (可选, 用于 Hebb 检索)
        """
        import librosa, soundfile as sf, jieba, re

        # v3: Hebb 检索路径 (优先)
        if broca is not None:
            broca._ensure_sent_clusters()
            from cns.data_types import D
            from cerebrum.limbic_system.hippocampus import _masked_cosine

            word_audios = []; word_labels = []
            for concept_vec in path_centroids:
                q = np.zeros(D, dtype=np.float32)
                q[:64] = concept_vec[:64].astype(np.float32)
                h = broca._sentence_net.hash_features(q)
                hash_key = broca._sentence_net._hash_to_bucket(h)
                bucket = broca._sentence_net.buckets.get(hash_key, [])
                if not bucket:
                    bucket = broca._sentence_net.clusters[:200]

                mask = np.zeros(D, dtype=bool)
                mask[:64] = True
                best_sim = -1.0
                best_si = 0
                for c in bucket:
                    sim = _masked_cosine(h, c.centroid, mask)
                    if sim > best_sim:
                        best_sim = sim
                        best_si = broca._cluster_to_sentence.get(id(c), 0)

                sent = broca._clean_sentence(broca.sentences[best_si])
                words = [w for w in jieba.lcut(sent)
                        if len(w) >= 2 and not w.isspace()]
                for w in words[:2]:
                    wav_path = os.path.join(self.audio_dir, f'{w}.wav')
                    try: audio, _ = librosa.load(wav_path, sr=22050, mono=True)
                    except Exception: audio = np.zeros(2205)
                    word_audios.append(audio); word_labels.append(w)

        else:
            # ---- 概念→语料句映射 (同一 PCA 空间) [LEGACY: O(N) 全局扫描] ----
            if not hasattr(self, '_corpus_sentences'):
                corpus_path = os.path.join(os.path.dirname(__file__), 'corpus.txt')
                with open(corpus_path, 'r', encoding='utf-8') as f: text = f.read()
                self._corpus_sentences = [s.strip() for s in re.split(r'[。！？；\n]+', text)
                                         if len(s.strip()) > 5]
                from environments.text_interface import TextEnvironment
                self._corpus_env = TextEnvironment()
                self._corpus_embs = self._corpus_env.embeddings

            word_audios = []; word_labels = []
            for concept_vec in path_centroids:
                q = np.linalg.norm(concept_vec[:64])
                sims = np.dot(self._corpus_embs, concept_vec[:64]) / (
                    np.linalg.norm(self._corpus_embs, axis=1) * q + 1e-8)
                sent = self._corpus_sentences[int(np.argmax(sims))]
                words = [w for w in jieba.lcut(sent) if len(w) >= 2 and not w.isspace()]
                for w in words[:2]:
                    wav_path = os.path.join(self.audio_dir, f'{w}.wav')
                    try: audio, _ = librosa.load(wav_path, sr=22050, mono=True)
                    except Exception: audio = np.zeros(2205)
                    word_audios.append(audio); word_labels.append(w)

        silence = np.zeros(int(0.2 * 22050))
        full = word_audios[0]
        for a in word_audios[1:]: full = np.concatenate([full, silence, a])
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            sf.write(output_path, full, 22050)
        try:
            import sounddevice as sd
            sd.play(full, 22050)
            sd.wait()
        except Exception:
            pass
        return full, word_labels


_speaker = None
def get_speaker():
    global _speaker
    if _speaker is None:
        _speaker = WordSpeaker()
    return _speaker
