"""
prepare_spectrum_data.py —— 准备频谱训练数据 (edge-tts 高速版)
对每条语料: edge-tts 朗读 → .mp3 → librosa 提取梅尔频谱
输出: spectrum_dataset.npy — [(vec_64, mel), ...]
"""
import numpy as np, librosa, os, re, asyncio, tempfile
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


async def tts_to_file(text: str, out_path: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


async def generate_all(sentences, out_dir, voice, start=0, total=200):
    sr, n_mels, hop, n_fft, target = 22050, 80, 256, 1024, 64
    dataset = []
    tmp_mp3 = os.path.join(out_dir, '_tmp_tts.mp3')

    for i, sent in enumerate(sentences):
        try:
            await tts_to_file(sent, tmp_mp3, voice)
            audio, _ = librosa.load(tmp_mp3, sr=sr, mono=True)
            if len(audio) < n_fft:
                audio = np.pad(audio, (0, n_fft - len(audio)))
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop, n_fft=n_fft)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            if mel_db.shape[1] >= target: mel_db = mel_db[:, :target]
            else: mel_db = np.pad(mel_db, ((0, 0), (0, target - mel_db.shape[1])))
            dataset.append((i, mel_db.astype(np.float32)))
            if (len(dataset)) % 10 == 0:
                print(f"  {start + len(dataset)}/{total} done")
        except Exception as e:
            print(f"  [{i}] skip: {sent[:30]}... — {e}")

    if os.path.exists(tmp_mp3): os.remove(tmp_mp3)
    return dataset


def main():
    corpus_path = os.path.join(os.path.dirname(__file__), 'corpus.txt')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [s.strip() for s in re.split(r'[。！？；\n]+', text) if len(s.strip()) > 5]
    print(f"Corpus: {len(sentences)} sentences")

    print("Encoding embeddings...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    full_emb = encoder.encode(sentences, show_progress_bar=True)
    n_pca = min(64, len(sentences))
    pca = PCA(n_components=n_pca)
    emb_64 = pca.fit_transform(full_emb).astype(np.float32)
    if n_pca < 64:
        padded = np.zeros((len(sentences), 64), dtype=np.float32)
        padded[:, :n_pca] = emb_64
        emb_64 = padded
    print(f"PCA: 384d -> 64d, var={pca.explained_variance_ratio_.sum():.1%}")

    sample_n = min(100, len(sentences))
    indices = np.linspace(0, len(sentences) - 1, sample_n, dtype=int)
    sampled_sentences = [sentences[i] for i in indices]
    sampled_emb = emb_64[indices]
    print(f"Sampled {sample_n} sentences for TTS")

    out_dir = os.path.dirname(__file__)
    print("Generating speech with edge-tts (~1-2 min)...")
    mel_data = asyncio.run(generate_all(sampled_sentences, out_dir,
                                         "zh-CN-XiaoxiaoNeural",
                                         start=0, total=sample_n))

    dataset = [(sampled_emb[i], mel) for i, mel in mel_data]
    out_path = os.path.join(out_dir, 'spectrum_dataset.npy')
    np.save(out_path, np.array(dataset, dtype=object))
    print(f"\nSaved {len(dataset)} pairs to spectrum_dataset.npy")


if __name__ == '__main__':
    main()
