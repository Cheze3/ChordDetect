"""Pitch-class wheel + keyboard strip, drawn on a 1-bit canvas (RLCD prototype).

The canvas is a PIL image in mode "1" (pure black/white, no anti-aliasing), so what
you see is close to what a 1-bit reflective LCD will show.

Usage:
    python visualizer.py --png preview.png     # render sample states to a PNG
    python visualizer.py --demo                # animated demo in a window, no mic
    python visualizer.py --live                # window driven by the microphone
Options:
    --portrait   300x400 canvas instead of 400x300
    --fifths     arrange the wheel in circle-of-fifths order
    --scale N    window/PNG magnification (default 2)

Requires: numpy, scipy, pillow; pygame for --demo/--live; sounddevice for --live.
"""
import argparse
import math
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

import chord_detect as cd

NOTE_LO, NOTE_HI = cd.NOTE_LO, cd.NOTE_HI    # keyboard strip range (C3..C6)
WHITE_PCS = (0, 2, 4, 5, 7, 9, 11)
BLACK_AFTER = {0: 1, 2: 3, 5: 6, 7: 8, 9: 10}   # white pc -> black key pc to its right
INK, PAPER = 0, 255   # mode "1" pixels are 0 or 255


# ---------- helpers ----------
_font_cache = {}


def font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.load_default(size=size)
    return _font_cache[size]


def text_size(draw, s, f):
    x0, y0, x1, y1 = draw.textbbox((0, 0), s, font=f)
    return x1 - x0, y1 - y0


def draw_text(draw, xy, s, size, anchor="lt"):
    """Draw text with the given anchor (PIL anchor codes) in ink."""
    draw.text(xy, s, fill=INK, font=font(size), anchor=anchor)


def fit_size(draw, s, max_w, max_size, min_size=10):
    for size in range(max_size, min_size - 1, -2):
        if text_size(draw, s, font(size))[0] <= max_w:
            return size
    return min_size


_pattern_cache = {}


def _pattern(size, density):
    key = (size, density >= 0.5)
    if key not in _pattern_cache:
        w, h = size
        ys, xs = np.mgrid[0:h, 0:w]
        on = ((xs + ys) % 2 == 0) if density >= 0.5 else ((xs % 2 == 0) & (ys % 2 == 0))
        arr = np.where(on, INK, PAPER).astype(np.uint8)
        _pattern_cache[key] = Image.fromarray(arr).convert("1")
    return _pattern_cache[key]


def hatch(img, mask, density):
    """Fill the masked area with an ordered-dither gray (density 0.25 or 0.5)."""
    pat = _pattern(img.size, density)
    ink_mask = ImageChops.logical_and(mask, ImageChops.invert(pat))
    img.paste(INK, mask=ink_mask)


def root_pc_of(label):
    """Pitch class of the chord root from a label like 'Am/C', or None."""
    if label in ("-", "") or label is None:
        return None
    name = label[:2] if len(label) > 1 and label[1] == "#" else label[:1]
    if name not in cd.NOTE_NAMES:
        return None
    return cd.NOTE_NAMES.index(name)


def is_chord(label, notes_text):
    return label not in ("-", "") and label != notes_text


# ---------- drawing ----------
def draw_wheel(img, draw, cx, cy, radius, active_pcs, root_pc, fifths, label_font):
    def pos(pc):
        return (pc * 7) % 12 if fifths else pc

    def point(pc, r):
        ang = math.radians(-90 + 30 * pos(pc))
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=INK, width=1)

    # polygon through active pitch classes, in angular order
    act = sorted(active_pcs, key=pos)
    if len(act) >= 3:
        pts = [point(pc, radius) for pc in act]
        mask = Image.new("1", img.size, 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        hatch(img, mask, 0.25)
        draw.line(pts + [pts[0]], fill=INK, width=2)
    elif len(act) == 2:
        draw.line([point(act[0], radius), point(act[1], radius)], fill=INK, width=2)

    for pc in range(12):
        x, y = point(pc, radius)
        if pc in active_pcs:
            r = 9 if pc == root_pc else 7
            draw.ellipse([x - r, y - r, x + r, y + r], fill=INK)
            if pc == root_pc:                       # root: ring around the dot
                draw.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], outline=INK, width=2)
        else:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=PAPER, outline=INK, width=1)
        lx, ly = point(pc, radius + 24)
        draw.text((lx, ly), cd.NOTE_NAMES[pc], fill=INK, font=label_font, anchor="mm")


def draw_center(draw, cx, cy, label, notes_text, inner_w):
    if label in ("-", ""):
        draw.text((cx, cy), "listening", fill=INK, font=font(16), anchor="mm")
        return
    s = label
    size = fit_size(draw, s, inner_w, 44)
    f = font(size)
    w, h = text_size(draw, s, f)
    pad = 5
    draw.rectangle([cx - w / 2 - pad, cy - h / 2 - pad - 2, cx + w / 2 + pad, cy + h / 2 + pad],
                   fill=PAPER, outline=INK, width=1)
    draw.text((cx, cy), s, fill=INK, font=f, anchor="mm")


def draw_keyboard(img, draw, x0, y0, width, height, active_notes, bass_note):
    n_white = sum(1 for n in range(NOTE_LO, NOTE_HI + 1) if n % 12 in WHITE_PCS)
    kw = width / n_white
    black_h = int(height * 0.62)
    white_idx = {}
    i = 0
    for n in range(NOTE_LO, NOTE_HI + 1):
        if n % 12 in WHITE_PCS:
            white_idx[n] = i
            i += 1
    active = set(active_notes)

    for n, i in white_idx.items():
        xa, xb = x0 + i * kw, x0 + (i + 1) * kw
        draw.rectangle([xa, y0, xb, y0 + height], fill=PAPER, outline=INK, width=1)
        if n in active:
            r = 5 if n == bass_note else 4
            draw.ellipse([(xa + xb) / 2 - r, y0 + height - 4 - 2 * r, (xa + xb) / 2 + r, y0 + height - 4],
                         fill=INK)
    for n in range(NOTE_LO, NOTE_HI + 1):
        if n % 12 in BLACK_AFTER.values():
            left = white_idx[n - 1]
            xc = x0 + (left + 1) * kw
            bw = kw * 0.6
            draw.rectangle([xc - bw / 2, y0, xc + bw / 2, y0 + black_h], fill=INK)
            if n in active:
                r = 4
                cyk = y0 + black_h - 8
                draw.ellipse([xc - r, cyk - r, xc + r, cyk + r], fill=PAPER, outline=PAPER)

    # C labels under each octave start
    for n in range(NOTE_LO, NOTE_HI + 1, 12):
        draw.text((x0 + white_idx[n] * kw + kw / 2, y0 + height + 3), f"C{n // 12 - 1}",
                  fill=INK, font=font(11), anchor="mt")


def key_centers(x0, width):
    """x position of every key's center (black keys sit on the boundary between whites)."""
    n_white = sum(1 for n in range(NOTE_LO, NOTE_HI + 1) if n % 12 in WHITE_PCS)
    kw = width / n_white
    xs, i = {}, 0
    for n in range(NOTE_LO, NOTE_HI + 1):
        if n % 12 in WHITE_PCS:
            xs[n] = x0 + (i + 0.5) * kw
            i += 1
        else:
            xs[n] = x0 + i * kw          # i = whites placed so far = boundary to the right of n-1
    return xs


def draw_spectrum(img, draw, x0, y0, width, height, levels, active_notes):
    """Spectrum silhouette aligned to the keyboard below: one point per key."""
    base = y0 + height
    draw.line([(x0, base), (x0 + width, base)], fill=INK, width=1)
    if levels is None:
        return
    xs = key_centers(x0, width)
    notes = list(range(NOTE_LO, NOTE_HI + 1))
    pts = [(xs[n], base - float(levels[n - NOTE_LO]) * (height - 2)) for n in notes]
    poly = [(pts[0][0], base)] + pts + [(pts[-1][0], base)]
    mask = Image.new("1", img.size, 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    hatch(img, mask, 0.25)
    draw.line(pts, fill=INK, width=1)
    for n in active_notes:
        if NOTE_LO <= n <= NOTE_HI:
            x, y = pts[n - NOTE_LO]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=INK)


def render(notes, label, notes_text, history, size=(400, 300), fifths=False, levels=None):
    """Draw one state. notes: MIDI list; label: chord name or note text or '-'."""
    W, H = size
    img = Image.new("1", size, PAPER)
    draw = ImageDraw.Draw(img)
    landscape = W >= H

    key_h, spec_h, margin = 44, 36, 8
    kb_y = H - key_h - 18
    spec_y = kb_y - spec_h - 2
    if landscape:
        avail_h = spec_y - 6 - margin
        radius = int(min(avail_h / 2 - 26, (W - 2 * 88) / 2 - 32))
        cx, cy = W / 2, margin + avail_h / 2
    else:
        avail_h = spec_y - 24 - margin
        radius = int(min(W / 2 - 42, avail_h / 2 - 30))
        cx, cy = W / 2, margin + avail_h / 2

    active_pcs = {n % 12 for n in notes}
    chord = is_chord(label, notes_text)
    root = root_pc_of(label) if chord else None

    draw_wheel(img, draw, cx, cy, radius, active_pcs, root, fifths, font(15))
    draw_center(draw, cx, cy, label if chord else (notes_text if notes_text else "-"),
                notes_text, int(radius * 1.15))
    draw_spectrum(img, draw, margin, spec_y, W - 2 * margin, spec_h, levels, notes)
    draw_keyboard(img, draw, margin, kb_y, W - 2 * margin, key_h, notes, min(notes) if notes else None)

    if not landscape:
        # no side columns in portrait: one line of detected notes above the keyboard
        draw.text((W / 2, spec_y - 8), notes_text, fill=INK, font=font(16), anchor="mm")
    if landscape:
        # left: detected notes
        draw_text(draw, (margin, margin), "NOTES", 12)
        for i, n in enumerate(notes[:6]):
            draw_text(draw, (margin, margin + 18 + i * 22), cd.midi_name(n), 18)
        # right: recent chords
        draw_text(draw, (W - margin, margin), "RECENT", 12, anchor="rt")
        for i, h in enumerate(list(history)[::-1][:5]):
            sz = fit_size(draw, h, 70, 22 if i == 0 else 16)
            draw_text(draw, (W - margin, margin + 18 + i * 26), h, sz, anchor="rt")
    return img


# ---------- preview / demo / live ----------
def synth_levels(notes, noise=0.003, seed=0):
    """Per-key levels for a synthetic frame of the given notes (plus mic noise)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(cd.N_FFT)
    for n in notes:
        y = y + cd.piano_like(cd.midi_to_hz(n), cd.N_FFT / cd.SR)[: cd.N_FFT]
    if notes:
        y = y / np.max(np.abs(y)) * 0.3
    y = y + rng.normal(0, noise, cd.N_FFT)
    return cd.key_levels(y.astype(np.float32))


def sample_states():
    ch = cd.chord_notes
    rows = [
        (ch(0, ""), "C", "C4 E4 G4", ["Am", "F", "G"]),
        ([57, 60, 64], "Am", "A3 C4 E4", ["C", "F", "G"]),
        ([64, 67, 72], "C/E", "E4 G4 C5", ["Am", "F", "G"]),
        ([55, 59, 62, 65], "G7", "G3 B3 D4 F4", ["C", "Am", "F"]),
        ([60, 64], "C4 E4", "C4 E4", ["C", "F", "G"]),
        ([], "-", "", ["C", "F", "G"]),
    ]
    return [(n, lab, txt, hist, synth_levels(n)) for n, lab, txt, hist in rows]


def save_preview(path, size, fifths, scale):
    imgs = [render(n, lab, txt, hist, size, fifths, lv) for n, lab, txt, hist, lv in sample_states()]
    cols = 2
    rows = (len(imgs) + cols - 1) // cols
    gap = 10
    sheet = Image.new("1", (cols * size[0] + (cols + 1) * gap, rows * size[1] + (rows + 1) * gap), PAPER)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im, (gap + c * (size[0] + gap), gap + r * (size[1] + gap)))
        ImageDraw.Draw(sheet).rectangle(
            [gap + c * (size[0] + gap) - 1, gap + r * (size[1] + gap) - 1,
             gap + c * (size[0] + gap) + size[0], gap + r * (size[1] + gap) + size[1]], outline=INK)
    sheet = sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST)
    sheet.convert("L").save(path)
    print(f"saved {path} ({sheet.width}x{sheet.height})")


def to_surface(pygame, img, scale):
    big = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.NEAREST)
    return pygame.image.fromstring(big.tobytes(), big.size, "RGB")


def run_window(source, size, fifths, scale):
    """source: generator yielding (notes, label, text, levels) per update (None = no update)."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((size[0] * scale, size[1] * scale))
    pygame.display.set_caption("Chord visualizer (1-bit preview)")
    clock = pygame.time.Clock()
    history, last = deque(maxlen=8), "-"
    state = ([], "-", "")
    shown_levels = np.zeros(NOTE_HI - NOTE_LO + 1)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
        upd = next(source, "end")
        if upd == "end":
            break
        shown_levels *= 0.92                      # release: bars fall between updates
        if upd is not None:
            state = upd[:3]
            shown_levels = np.maximum(shown_levels, upd[3])   # attack: rise at once
            label = state[1]
            if label != "-" and label != last:
                history.append(label)
            last = label
        notes, label, text = state
        img = render(notes, label, text, history, size, fifths, shown_levels)
        screen.blit(to_surface(pygame, img, scale), (0, 0))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def demo_source():
    seq = [[60, 64, 67], [64, 67, 72], [55, 59, 62, 65], [57, 60, 64], [53, 57, 60], [60, 64], [60], []]
    while True:
        for notes in seq:
            text, label = cd.describe(notes)
            levels = synth_levels(notes)
            for i in range(45):
                yield notes, label, text, (levels if i % 4 == 0 else np.zeros_like(levels))


def live_source(device):
    import queue

    import sounddevice as sd
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    def open_stream(rate):
        return sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device,
                              blocksize=round(cd.HOP * rate / cd.SR), callback=cb)
    try:
        stream, rate = open_stream(cd.SR), cd.SR
    except sd.PortAudioError:
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
        stream = open_stream(rate)
    analyzer = cd.LiveAnalyzer(rate)
    with stream:
        while True:
            try:
                block = q.get(timeout=0.02)
            except queue.Empty:
                yield None
                continue
            res = analyzer.push(block)
            yield None if res is None else (list(analyzer.shown_notes), res[0], res[1], analyzer.levels)


def main():
    ap = argparse.ArgumentParser(description="Pitch-class wheel visualizer (1-bit preview)")
    ap.add_argument("--png", help="render sample states to this PNG")
    ap.add_argument("--demo", action="store_true", help="animated demo, no microphone")
    ap.add_argument("--live", action="store_true", help="drive from the microphone")
    ap.add_argument("--device", help="input device index or name")
    ap.add_argument("--portrait", action="store_true", help="300x400 instead of 400x300")
    ap.add_argument("--fifths", action="store_true", help="circle-of-fifths order")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--silence", type=float)
    args = ap.parse_args()
    size = (300, 400) if args.portrait else (400, 300)
    if args.silence is not None:
        cd.SILENCE_RMS = args.silence

    if args.png:
        save_preview(args.png, size, args.fifths, args.scale)
    elif args.demo:
        run_window(demo_source(), size, args.fifths, args.scale)
    elif args.live:
        dev = int(args.device) if args.device and args.device.isdigit() else args.device
        run_window(live_source(dev), size, args.fifths, args.scale)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()