"""Export test vectors from the Python detector for the C++ port.

Writes to cpp/test_data/:
  window.bin    Hann window (float32, N_FFT values)
  frames.bin    test frames (float32, N_FFT values each, back to back)
  spectra.bin   FFT magnitudes of the first SPECTRA_COUNT windowed frames (float32)
  expected.txt  one line per frame: index kind note_count notes... chord
  naming.txt    chord-naming cases: note_count notes... chord ('-' means no chord)

Usage:
    python export_vectors.py                    # synthetic cases only
    python export_vectors.py rec1.wav rec2.wav  # also add frames from recordings
"""
import sys
from pathlib import Path

import numpy as np

import chord_detect as cd

OUT = Path(__file__).parent / "cpp" / "test_data"
SPECTRA_COUNT = 8


def fit(y):
    """Trim or zero-pad to exactly N_FFT samples, as float32."""
    y = np.asarray(y, dtype=np.float32)[: cd.N_FFT]
    return np.pad(y, (0, cd.N_FFT - len(y)))


def synth(notes, rng, noise, cents=0.0):
    y = 0
    for n in notes:
        f = cd.midi_to_hz(n) * 2 ** (rng.uniform(-cents, cents) / 1200)
        y = y + cd.piano_like(f, cd.N_FFT / cd.SR)
    y = y / np.max(np.abs(y)) * 0.5
    return fit(y + rng.normal(0, noise, len(y)))


def synthetic_cases(rng):
    cases = []
    for m in range(48, 80):
        cases.append(("single", synth([m], rng, 0.02)))
    for r in range(48, 72, 3):
        for i in (3, 4, 5, 7):
            cases.append(("dyad", synth([r, r + i], rng, 0.02)))
    for root in range(12):
        for q in ("", "m"):
            for inv in (0, 1, 2):
                notes = cd.chord_notes(root, q, inversion=inv)
                cases.append(("triad", synth(notes, rng, 0.02)))
                cases.append(("triad_noisy", synth(notes, rng, 0.1, cents=15)))
    for r in range(12):
        cases.append(("seventh", synth([48 + r, 52 + r, 55 + r, 58 + r], rng, 0.02)))
        cases.append(("octave_doubled", synth([48 + r, 52 + r, 55 + r, 60 + r], rng, 0.02)))
    # Levels around SILENCE_RMS, so the silence threshold is actually exercised
    base = synth([48, 52, 55], rng, 0.0)
    for target in (0.005, 0.0095, 0.0105, 0.02):
        scaled = base * (target / float(np.sqrt(np.mean(base ** 2))))
        cases.append((f"level_{target}", scaled.astype(np.float32)))
    cases.append(("silence", np.zeros(cd.N_FFT, dtype=np.float32)))
    cases.append(("quiet_noise", fit(rng.normal(0, 0.002, cd.N_FFT))))
    return cases


def wav_cases(paths):
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    cases = []
    for path in paths:
        sr, x = wavfile.read(path)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.dtype.kind == "i":
            x = x / np.iinfo(x.dtype).max
        x = x.astype(np.float32)
        if sr != cd.SR:
            g = np.gcd(sr, cd.SR)
            x = resample_poly(x, cd.SR // g, sr // g).astype(np.float32)
        for start in range(0, len(x) - cd.N_FFT, cd.HOP):
            cases.append((f"wav:{Path(path).name}@{start / cd.SR:.2f}s",
                          x[start:start + cd.N_FFT].copy()))
    return cases


def naming_cases(rng):
    sets = []
    for root in range(12):                      # every chord type, every inversion
        for _, intervals in cd.CHORD_TYPES:
            notes = [48 + root + i for i in intervals]
            for _ in range(len(notes)):
                sets.append(list(notes))
                notes = notes[1:] + [notes[0] + 12]
    for _ in range(500):                        # random note sets, mostly not chords
        k = int(rng.integers(1, 7))
        sets.append(sorted(int(n) for n in rng.integers(48, 85, size=k)))
    return sets


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cases = synthetic_cases(rng) + wav_cases(sys.argv[1:])

    cd.WINDOW.astype("<f4").tofile(OUT / "window.bin")
    frames = np.stack([f for _, f in cases]).astype("<f4")
    frames.tofile(OUT / "frames.bin")

    spectra = [np.abs(np.fft.rfft(f * cd.WINDOW)).astype("<f4")
               for f in frames[:SPECTRA_COUNT]]
    np.stack(spectra).tofile(OUT / "spectra.bin")

    with open(OUT / "expected.txt", "w") as fh:
        for i, ((kind, _), frame) in enumerate(zip(cases, frames)):
            notes = cd.detect_notes(frame)
            chord = cd.name_chord(notes) or "-"
            fh.write(f"{i} {kind.replace(' ', '_')} {len(notes)} "
                     f"{' '.join(map(str, notes))} {chord}\n".replace("  ", " "))

    with open(OUT / "naming.txt", "w") as fh:
        for notes in naming_cases(rng):
            fh.write(f"{len(notes)} {' '.join(map(str, notes))} {cd.name_chord(notes) or '-'}\n")

    print(f"wrote {len(cases)} frames, {SPECTRA_COUNT} spectra, to {OUT}")


if __name__ == "__main__":
    main()
