# fm demodulation and audio

import wave

import numpy as np

MAX_DEVIATION = 75000
AUDIO_RATE = 50000


def demodulate(iq, sample_rate):
    # phase change between samples is the frequency
    # no dc removal here because the station itself is at 0 hz
    phase_change = np.angle(iq[1:] * np.conj(iq[:-1]))
    return phase_change * sample_rate / (2 * np.pi * MAX_DEVIATION)


def lowpass_filter(cutoff_hz, sample_rate, num_taps=129):
    n = np.arange(num_taps) - (num_taps - 1) / 2
    taps = np.sinc(2 * cutoff_hz / sample_rate * n) * np.hamming(num_taps)
    return taps / np.sum(taps)


def get_audio(baseband, sample_rate):
    filtered = np.convolve(baseband, lowpass_filter(15000, sample_rate), mode="same")
    audio = filtered[::sample_rate // AUDIO_RATE]

    # undo the treble boost stations add
    alpha = 1 - np.exp(-1 / (AUDIO_RATE * 75e-6))
    out = np.zeros(len(audio))
    y = 0.0
    for i, x in enumerate(audio.tolist()):
        y = y + alpha * (x - y)
        out[i] = y
    return out


def save_wav(filename, audio):
    peak = np.max(np.abs(audio))
    if peak == 0:
        peak = 1
    samples = (audio / peak * 0.9 * 32767).astype(np.int16)
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(AUDIO_RATE)
        f.writeframes(samples.tobytes())
