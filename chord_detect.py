"""Note and chord detector: FFT -> note detection -> chord naming.

Per frame it finds the individual notes being played. If the notes form a
known chord, the chord name is shown; otherwise the notes are shown.
The older chroma/template method (detect) is kept as a baseline.

Usage:
    python chord_detect.py            # synthetic self-tests
    python chord_detect.py file.wav   # print note/chord timeline for a recording

Requires: numpy, scipy
"""
import sys

import numpy as np

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


# ---- Note detection (multi-pitch) ----
NOTE_LO, NOTE_HI = 48, 84   # MIDI range searched: C3..C6
N_HARM = 6                  # harmonics summed per candidate note
WEIGHT_POW = 0.5            # harmonic h is weighted h ** -WEIGHT_POW
FUND_FRAC = 0.15            # a note's fundamental must reach this fraction of the spectrum max
NOTE_THRESHOLD = 0.3        # stop when next note's salience < this fraction of the first
MAX_NOTES = 6
_DF = SR / N_FFT


def _hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _band(m, h, w=1):
    c = round(_hz(m) * h / _DF)
    return max(c - w, 0), min(c + w + 1, N_FFT // 2 + 1)


_BANDS = {
    (m, h): _band(m, h)
    for m in range(NOTE_LO, NOTE_HI + 1)
    for h in range(1, N_HARM + 1)
}


def detect_notes(frame):
    """Return a sorted list of MIDI notes found in one frame.

    Greedy: pick the note whose harmonics carry the most energy, subtract
    its harmonics from the spectrum, repeat until nothing significant remains.
    """
    if np.sqrt(np.mean(frame ** 2)) < SILENCE_RMS:
        return []
    spec = np.abs(np.fft.rfft(frame * WINDOW)).astype(np.float64)
    smax = spec.max()
    found, first = [], None
    for _ in range(MAX_NOTES):
        best_m, best_s = None, 0.0
        for m in range(NOTE_LO, NOTE_HI + 1):
            if m in found:
                continue
            a, b = _BANDS[(m, 1)]
            if spec[a:b].max() < FUND_FRAC * smax:
                continue
            s = 0.0
            for h in range(1, N_HARM + 1):
                a, b = _BANDS[(m, h)]
                s += spec[a:b].max() * h ** -WEIGHT_POW
            if s > best_s:
                best_m, best_s = m, s
        if best_m is None:
            break
        if first is None:
            first = best_s
        elif best_s < NOTE_THRESHOLD * first:
            break
        found.append(best_m)
        a, b = _BANDS[(best_m, 1)]
        p1 = spec[a:b].max()
        for h in range(1, N_HARM + 1):
            a, b = _BANDS[(best_m, h)]
            spec[a:b] = np.maximum(spec[a:b] - p1 / h, 0)
    return sorted(found)


# Chord qualities as semitone intervals above the root, in priority order
CHORD_TYPES = [
    ("", (0, 4, 7)), ("m", (0, 3, 7)), ("dim", (0, 3, 6)), ("aug", (0, 4, 8)),
    ("sus2", (0, 2, 7)), ("sus4", (0, 5, 7)),
    ("maj7", (0, 4, 7, 11)), ("7", (0, 4, 7, 10)), ("m7", (0, 3, 7, 10)),
]


def name_chord(notes):
    """Chord name for a list of MIDI notes, or None if they form no known chord.

    Matching uses pitch classes only (octave doublings are fine). When several
    roots fit (e.g. Csus2 = Gsus4), the bass note's root is preferred.
    A slash is added for inversions, e.g. C/E.
    """
    if len(notes) < 3:
        return None
    pcs = frozenset(n % 12 for n in notes)
    bass = min(notes) % 12
    matches = []
    for root in range(12):
        for quality, intervals in CHORD_TYPES:
            if frozenset((root + i) % 12 for i in intervals) == pcs:
                matches.append((root, quality))
    if not matches:
        return None
    root, quality = next((m for m in matches if m[0] == bass), matches[0])
    name = NOTE_NAMES[root] + quality
    return name if root == bass else f"{name}/{NOTE_NAMES[bass]}"


def describe(notes):
    """(note list text, label): the chord if the notes form one, else the notes."""
    if not notes:
        return "", "-"
    text = " ".join(midi_name(n) for n in notes)
    return text, name_chord(notes) or text


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


def chord_notes(root_pc, quality, octave_base=48, inversion=0):
    """MIDI note numbers of a chord, after applying inversions."""
    intervals = (0, 4, 7) if quality == "" else (0, 3, 7)
    notes = [octave_base + root_pc + i for i in intervals]
    for _ in range(inversion):
        notes.append(notes.pop(0) + 12)
    return notes


def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def make_chord(root_pc, quality, octave_base=48, inversion=0):
    notes = chord_notes(root_pc, quality, octave_base, inversion)
    y = sum(piano_like(midi_to_hz(n), N_FFT / SR) for n in notes)
    return y / np.max(np.abs(y)) * 0.5


def self_test(noise_std=0.02, seed=0):
    rng = np.random.default_rng(seed)
    failures = []
    total = 0
    for root in range(12):
        for q in ("", "m"):
            for inv in (0, 1, 2):
                expected = NOTE_NAMES[root] + q
                y = make_chord(root, q, inversion=inv)
                y = y + rng.normal(0, noise_std, y.shape)
                pred, score = detect(y.astype(np.float32))
                total += 1
                if pred != expected:
                    notes = " ".join(
                        midi_name(n) for n in chord_notes(root, q, inversion=inv)
                    )
                    failures.append((expected, inv, notes, pred, score))

    correct = total - len(failures)
    print(f"synthetic accuracy: {correct}/{total} = {correct / total:.1%}")
    if failures:
        print(f"\n{'expected':9s}{'inv':5s}{'notes played':16s}{'predicted':11s}score")
        for exp, inv, notes, pred, score in failures:
            print(f"{exp:9s}{inv:<5d}{notes:16s}{pred!s:11s}{score:.2f}")
    return failures


def note_test(noise_std=0.02, seed=0):
    """Check note detection (single notes, intervals, chords) and chord naming."""
    rng = np.random.default_rng(seed)
    cases = [("single", [m]) for m in range(48, 80)]
    cases += [("interval", [r, r + i]) for r in range(48, 72, 3) for i in (3, 4, 5, 7)]
    for root in range(12):
        for q in ("", "m"):
            for inv in (0, 1, 2):
                cases.append(("chord", chord_notes(root, q, inversion=inv)))
    failures = []
    for kind, notes in cases:
        y = sum(piano_like(midi_to_hz(n), N_FFT / SR) for n in notes)
        y = y / np.max(np.abs(y)) * 0.5 + rng.normal(0, noise_std, N_FFT * 1)
        got = detect_notes(y.astype(np.float32))
        if got != sorted(notes):
            failures.append((kind, notes, got))
    print(f"\nnote detection: {len(cases) - len(failures)}/{len(cases)} correct")
    for kind, notes, got in failures:
        print(f"  {kind:9s} expected {' '.join(map(midi_name, notes)):14s}"
              f" got {' '.join(map(midi_name, got)) or '-'}")
    return failures


# ---- WAV mode ----
def run_file(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
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
        notes = detect_notes(x[start:start + N_FFT])
        text, label = describe(notes)
        print(f"{start / SR:6.2f}s  {label:12s}  {text if label != text else ''}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        self_test()
        note_test()