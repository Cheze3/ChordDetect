"""Baseline chord detector: FFT -> chroma -> template matching.

Usage:
    python chord_detect.py            # synthetic self-test
    python chord_detect.py file.wav   # print chord timeline for a recording

Requires: numpy, scipy
"""
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# ---- Parameters (freeze these; the C++ port must match exactly) ----
SR = 16000          # sample rate (Hz)
N_FFT = 4096        # frame length (0.256 s, ~3.9 Hz bin spacing) 
HOP = 2048          # hop between frames
FMIN = 130.8        # C3; below this, bin spacing is too coarse
FMAX = 800.0        # ignore high harmonics (lower FMAX helped on synthetic data)
COMPRESS = 1.0      # log compression strength (weaker was better on synthetic data)
SILENCE_RMS = 0.01  # below this, report no chord

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WINDOW = np.hanning(N_FFT).astype(np.float32)

# Precompute bin -> pitch class map
_freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
_valid = (_freqs >= FMIN) & (_freqs <= FMAX)
_midi = np.round(69 + 12 * np.log2(_freqs[_valid] / 440.0)).astype(int)
_pc = _midi % 12
_valid_idx = np.where(_valid)[0]


def chroma(frame):
    """12-bin chroma vector (L2 normalized) from one frame of N_FFT samples."""
    spec = np.abs(np.fft.rfft(frame * WINDOW))
    spec = np.log1p(COMPRESS * spec)
    c = np.zeros(12, dtype=np.float32)
    np.add.at(c, _pc, spec[_valid_idx])
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def build_templates():
    """24 templates: 12 major then 12 minor, L2 normalized."""
    names, rows = [], []
    for quality, intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
        for root in range(12):
            t = np.zeros(12, dtype=np.float32)
            for i in intervals:
                t[(root + i) % 12] = 1.0
            rows.append(t / np.linalg.norm(t))
            names.append(NOTE_NAMES[root] + quality)
    return names, np.stack(rows)


CHORD_NAMES, TEMPLATES = build_templates()


def detect(frame):
    """Return (chord_name or None, score)."""
    if np.sqrt(np.mean(frame ** 2)) < SILENCE_RMS:
        return None, 0.0
    scores = TEMPLATES @ chroma(frame)
    k = int(np.argmax(scores))
    return CHORD_NAMES[k], float(scores[k])


# ---- Synthetic test ----
def piano_like(freq, dur, n_harm=6):
    """Decaying note with harmonics, crude piano approximation."""
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        y += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
    return y * np.exp(-1.5 * t)


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def make_chord(root_pc, quality, octave_base=48, inversion=0):
    intervals = (0, 4, 7) if quality == "" else (0, 3, 7)
    notes = [octave_base + root_pc + i for i in intervals]
    for _ in range(inversion):
        notes.append(notes.pop(0) + 12)
    y = sum(piano_like(midi_to_hz(n), N_FFT / SR) for n in notes)
    return y / np.max(np.abs(y)) * 0.5


def self_test(noise_std=0.02, seed=3):
    rng = np.random.default_rng(seed)
    correct = total = 0
    for root in range(12):
        for q in ("", "m"):
            for inv in (0, 1, 2):
                y = make_chord(root, q, inversion=inv)
                y = y + rng.normal(0, noise_std, y.shape)
                pred, _ = detect(y.astype(np.float32))
                total += 1
                correct += pred == NOTE_NAMES[root] + q
    print(f"synthetic accuracy: {correct}/{total} = {correct / total:.1%}")


# ---- WAV mode ----
def run_file(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype.kind == "i":
        x = x / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    for start in range(0, len(x) - N_FFT, HOP):
        name, score = detect(x[start:start + N_FFT])
        print(f"{start / SR:6.2f}s  {name or '-':4s}  {score:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        self_test()
