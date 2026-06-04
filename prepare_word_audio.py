"""prepare_word_audio.py — 原始音频数据集 (零合成，零Griffin-Lim)"""
import numpy as np, os, re, asyncio
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


async def tts_to_file(text, out_path, voice="zh-CN-XiaoxiaoNeural"):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


async def generate_all(words, out_dir, sr=22050):
    import librosa
    tmp = os.path.join(out_dir, '_tmp_word.mp3')
    audios = []
    for i, word in enumerate(words):
        try:
            await tts_to_file(word, tmp, "zh-CN-XiaoxiaoNeural")
            audio, _ = librosa.load(tmp, sr=sr, mono=True)
            audios.append(audio.astype(np.float32))
            if (i+1) % 30 == 0: print(f"  {i+1}/{len(words)}")
        except Exception as e: print(f"  [{word}] skip: {e}")
    if os.path.exists(tmp): os.remove(tmp)
    return audios


def main():
    import jieba
    corpus_path = os.path.join(os.path.dirname(__file__), 'corpus.txt')
    with open(corpus_path, 'r', encoding='utf-8') as f: text = f.read()
    text_clean = re.sub(r'[^一-鿿\w]+', ' ', text)
    words_raw = jieba.lcut(text_clean)
    seen = set(); words = []
    for w in words_raw:
        w = w.strip()
        if len(w) >= 2 and w not in seen and not w.isdigit():
            seen.add(w); words.append(w)
    print(f"Corpus: {len(words)} words")

    print("Encoding...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    full = encoder.encode(words, show_progress_bar=True)
    pca = PCA(n_components=min(64, len(words)))
    embs = pca.fit_transform(full).astype(np.float32)
    if embs.shape[1] < 64:
        p = np.zeros((len(words),64), dtype=np.float32); p[:,:embs.shape[1]]=embs; embs=p

    out_dir = os.path.dirname(__file__)
    print(f"TTS for {len(words)} words...")
    audios = asyncio.run(generate_all(words, out_dir))

    # 存为 pickle (list of numpy arrays)
    import pickle
    data = {'embs': embs, 'audios': audios, 'words': words}
    with open(os.path.join(out_dir, 'word_audio_dataset.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(audios)} audio arrays, {len(words)} words")

if __name__ == '__main__':
    main()
