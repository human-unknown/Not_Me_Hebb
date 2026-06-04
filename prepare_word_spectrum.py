"""
prepare_word_spectrum.py —— 词级频谱训练数据
流程: corpus.txt → jieba分词 → 去重 → sentence-transformer逐词编码
      → edge-tts逐词朗读 → 短频谱 (target_frames=16)
输出: word_spectrum_dataset.npy — [(vec_64, mel, word), ...]
"""
import numpy as np, librosa, os, re, asyncio
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


async def tts_to_file(text, out_path, voice="zh-CN-XiaoxiaoNeural"):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


async def generate_all(words, encoder, out_dir, sr=22050,
                       n_mels=80, hop_length=256, n_fft=1024,
                       target_frames=16):
    tmp_mp3 = os.path.join(out_dir, '_tmp_word.mp3')
    dataset = []
    for i, word in enumerate(words):
        try:
            await tts_to_file(word, tmp_mp3, "zh-CN-XiaoxiaoNeural")
            audio, _ = librosa.load(tmp_mp3, sr=sr, mono=True)
            if len(audio) < n_fft: audio = np.pad(audio, (0, n_fft-len(audio)))
            peak = np.max(np.abs(audio))
            if peak > 1e-8: audio = audio / peak * 0.9
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
            if mel.shape[1] >= target_frames: mel = mel[:,:target_frames]
            else: mel = np.pad(mel, ((0,0),(0,target_frames-mel.shape[1])))
            m = mel.max()
            mel_norm = (mel / m).astype(np.float32) if m > 1e-8 else mel.astype(np.float32)
            vec_full = encoder.encode(word)
            dataset.append((vec_full.astype(np.float32), mel_norm, word))
            if (len(dataset)) % 20 == 0: print(f"  {len(dataset)}/{len(words)} done")
        except Exception as e: print(f"  [{word}] skip: {e}")
    if os.path.exists(tmp_mp3): os.remove(tmp_mp3)
    return dataset


def main():
    import jieba
    corpus_path = os.path.join(os.path.dirname(__file__), 'corpus.txt')
    with open(corpus_path, 'r', encoding='utf-8') as f: text = f.read()
    text_clean = re.sub(r'[^一-鿿\w]+', ' ', text)
    words_raw = jieba.lcut(text_clean)
    seen = set()
    words = []
    for w in words_raw:
        w = w.strip()
        if len(w) >= 2 and w not in seen and not w.isdigit():
            seen.add(w); words.append(w)
    print(f"Corpus: {len(words)} unique words (len>=2)")

    print("Encoding word embeddings...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    full_embs = encoder.encode(words, show_progress_bar=True)
    n_pca = min(64, len(words))
    pca = PCA(n_components=n_pca)
    embs_64 = pca.fit_transform(full_embs).astype(np.float32)
    if n_pca < 64:
        p = np.zeros((len(words),64), dtype=np.float32); p[:,:n_pca]=embs_64; embs_64=p
    print(f"PCA: 384d -> 64d, var={pca.explained_variance_ratio_.sum():.1%}")

    out_dir = os.path.dirname(__file__)
    print(f"Generating speech for {len(words)} words (~2-3 min)...")
    mel_data = asyncio.run(generate_all(words, encoder, out_dir, target_frames=16))

    dataset = [(embs_64[i], mel, word) for i, (_, mel, word) in enumerate(mel_data)]
    out_path = os.path.join(out_dir, 'word_spectrum_dataset.npy')
    np.save(out_path, np.array(dataset, dtype=object))
    print(f"\nSaved {len(dataset)} pairs to word_spectrum_dataset.npy")
    print("First 10 words:", ' '.join(w for _, _, w in dataset[:10]))


if __name__ == '__main__':
    main()
