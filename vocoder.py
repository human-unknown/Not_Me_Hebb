"""vocoder.py — Griffin-Lim 梅尔频谱→音频"""
import numpy as np, librosa, soundfile as sf

def mel_to_audio(mel_input, sr=22050, hop_length=256, n_fft=1024, n_iter=32,
                 input_is_power: bool = False):
    """mel spectrogram → audio waveform.
    If input_is_power=True, skip db_to_power step.
    """
    if input_is_power:
        mel_power = mel_input
    else:
        mel_power = librosa.db_to_power(mel_input)
    return librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, hop_length=hop_length, n_fft=n_fft,
        n_iter=n_iter, window='hann')

def save_audio(audio, path, sr=22050):
    # 归一化到 -3dB 峰值
    peak = np.max(np.abs(audio)) + 1e-8
    audio = audio / peak * 0.7
    sf.write(path, audio, sr)
