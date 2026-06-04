"""
speech_output.py —— Stage 5A: 语音输出
自由能原理智能体

将 Hebb 扩散融合向量映射到最接近的语料句子并朗读。
"""

import numpy as np

# 模块级 TTS 引擎（延迟初始化）
_tts_engine = None


def _get_engine():
    global _tts_engine
    if _tts_engine is None:
        import pyttsx3
        _tts_engine = pyttsx3.init()
    return _tts_engine


def speak_fused_vector(fused_vec_64: np.ndarray,
                       sentences: list,
                       embeddings_64: np.ndarray) -> str:
    """将融合向量映射到语料最相似句并朗读。

    Args:
        fused_vec_64: diffuse() 返回的 64-dim 融合向量
        sentences:  语料句子列表
        embeddings_64: 语料句子的 64-dim 嵌入 (N, 64)

    Returns:
        最接近的句子文本
    """
    # 余弦相似度
    norms_s = np.linalg.norm(embeddings_64, axis=1)
    norm_f = np.linalg.norm(fused_vec_64)
    sims = np.dot(embeddings_64, fused_vec_64) / (norms_s * norm_f + 1e-8)
    top_idx = int(np.argmax(sims))

    spoken = sentences[top_idx]

    # TTS 朗读
    try:
        engine = _get_engine()
        engine.say(spoken)
        engine.runAndWait()
    except Exception:
        pass  # 无声音设备时静默

    return spoken


def speak_spectrum(fused_vec_64: np.ndarray,
                   model, norm_params: dict,
                   output_path: str = None) -> np.ndarray:
    """融合向量 → 频谱 → 音频（端到端，无文字）"""
    import torch

    v = (fused_vec_64 - norm_params['vec_mean'].flatten()) / norm_params['vec_std'].flatten()
    v_t = torch.from_numpy(v).float().unsqueeze(0)

    model.eval()
    with torch.no_grad():
        mel_norm = model(v_t).squeeze(0).numpy()

    mel_db = mel_norm * (norm_params['mel_max'] - norm_params['mel_min']) + norm_params['mel_min']

    from vocoder import mel_to_audio
    audio = mel_to_audio(mel_db)

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        from vocoder import save_audio
        save_audio(audio, output_path)

    return audio


def explain_blended(fused_vec_64: np.ndarray,
                    net,  # ClusterNetwork
                    sentences: list,
                    embeddings_64: np.ndarray,
                    top_n: int = 3) -> list:
    """找融合向量最接近的几个集群，显示各自的语料标签。

    Returns:
        [(sentence, similarity), ...]  top_n 个概念贡献
    """
    results = []
    for c in net.clusters:
        sim = float(np.dot(c.centroid[:64], fused_vec_64) /
                    (np.linalg.norm(c.centroid[:64]) *
                     np.linalg.norm(fused_vec_64) + 1e-8))
        c._blend_sim = sim  # 临时标注
    sorted_clusters = sorted(net.clusters,
                             key=lambda c: getattr(c, '_blend_sim', 0),
                             reverse=True)

    for c in sorted_clusters[:top_n]:
        sims = np.dot(embeddings_64, c.centroid[:64]) / (
            np.linalg.norm(embeddings_64, axis=1) *
            np.linalg.norm(c.centroid[:64]) + 1e-8)
        top_s = sentences[int(np.argmax(sims))]
        results.append((top_s, float(getattr(c, '_blend_sim', 0))))
    return results
