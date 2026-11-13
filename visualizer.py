"""Pitch-class wheel + keyboard strip, drawn on a 1-bit canvas (RLCD prototype).

The canvas is a PIL image in mode "1" (pure black/white, no anti-aliasing), so what
you see is close to what a 1-bit reflective LCD will show.

Usage:
    python visualizer.py --png preview.png     # render sample states to a PNG
    python visualizer.py --demo                # animated demo in a window, no mic
    python visualizer.py --live                # window driven by the microphone
In a window, press N to switch between the UIs (wheel, poster, tape); Q or Esc quits.
Options:
    --ui NAME    start in this UI: wheel (default), poster or tape (poster/tape are landscape only)
    --portrait   300x400 canvas instead of 400x300
    --fifths     arrange the wheel in circle-of-fifths order
    --scale N    window/PNG magnification (default 2)

Requires: numpy, scipy, pillow; pygame for --demo/--live; sounddevice for --live.
"""
import argparse
import math
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

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


METER_FLOOR_DB = -60.0   # meter scale: RMS in dB relative to full scale


def _meter_frac(rms):
    db = 20 * np.log10(max(rms, 1e-9))
    return float(np.clip((db - METER_FLOOR_DB) / -METER_FLOOR_DB, 0.0, 1.0))


def draw_meter(draw, x0, y0, w, h, rms):
    """Level bar with a tick at the silence threshold (cd.SILENCE_RMS)."""
    draw.text((x0, y0 - 14), "LEVEL", fill=INK, font=font(11), anchor="lt")
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=PAPER, outline=INK)
    fill_x = x0 + 1 + _meter_frac(rms) * (w - 2)
    draw.rectangle([x0 + 1, y0 + 1, fill_x, y0 + h - 1], fill=INK)
    xt = x0 + _meter_frac(cd.SILENCE_RMS) * w
    draw.line([(xt, y0 - 4), (xt, y0 + h + 4)], fill=INK, width=2)
    draw.text((xt, y0 + h + 5), "min", fill=INK, font=font(10), anchor="mt")


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


def render(notes, label, notes_text, history, size=(400, 300), fifths=False, levels=None, rms=None):
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

    if rms is not None:
        if landscape:
            draw_meter(draw, margin, 170, 84, 10, rms)
        else:
            draw_meter(draw, margin, 24, 60, 10, rms)
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


# ---------- UI modes: poster and tape (landscape 400x300 only) ----------
UI_W, UI_H = 400, 300
MODES = ("wheel", "poster", "tape")
FONT_DIR = Path(__file__).resolve().parent / "fonts"      # drop font files here
FONT_CANDIDATES = {
    "big": [("Inter-Black.otf", 0), ("/System/Library/Fonts/Supplemental/Arial Black.ttf", 0),
            ("/usr/share/fonts/opentype/inter/Inter-Black.otf", 0)],
    "mono": [("DejaVuSansMono-Bold.ttf", 0), ("/System/Library/Fonts/Menlo.ttc", 1),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 0)],
}
_ui_fonts = {}
_warned = set()


def fnt(role, size):
    """Font for a role ('big' heavy display type, 'mono' bold monospace) at a size."""
    key = (role, size)
    if key in _ui_fonts:
        return _ui_fonts[key]
    for name, index in FONT_CANDIDATES[role]:
        path = name if Path(name).is_absolute() else str(FONT_DIR / name)
        try:
            _ui_fonts[key] = ImageFont.truetype(path, size, index=index)
            return _ui_fonts[key]
        except OSError:
            continue
    if role not in _warned:
        _warned.add(role)
        print(f"note: no '{role}' font found, using Pillow's default; "
              f"put {FONT_CANDIDATES[role][0][0]} in {FONT_DIR} for the intended look", file=sys.stderr)
    _ui_fonts[key] = ImageFont.load_default(size=size)
    return _ui_fonts[key]


def fit(d, s, role, max_w, max_h, max_size, min_size=12):
    for size in range(max_size, min_size - 1, -2):
        x0, y0, x1, y1 = d.textbbox((0, 0), s, font=fnt(role, size))
        if x1 - x0 <= max_w and y1 - y0 <= max_h:
            return size
    return min_size


def text_box(d, xy, s, font, anchor="l", fill=INK):
    """Draw text so its glyph box starts at xy (anchor l), ends at xy (r) or is centered on it (m)."""
    x0, y0, x1, y1 = d.textbbox((0, 0), s, font=font)
    w, h = x1 - x0, y1 - y0
    x, y = xy
    if anchor == "r":
        x -= w
    elif anchor == "m":
        x -= w / 2
        y -= h / 2
    d.text((x - x0, y - y0), s, font=font, fill=fill)
    return w, h


def draw_label(d, xy, label, max_w, max_h, max_size, anchor="l"):
    """Chord name in the big font; the silent state shows LISTENING in the mono font."""
    if label in ("-", ""):
        s, role, max_size = "LISTENING", "mono", min(max_size, 24)
    else:
        s, role = label, "big"
    size = fit(d, s, role, max_w, max_h, max_size)
    text_box(d, xy, s, fnt(role, size), anchor)


def fold(levels):
    """Per-key levels (C3..C6) -> 12 pitch-class levels (max over octaves)."""
    e = np.zeros(12)
    for i, lv in enumerate(levels):
        e[i % 12] = max(e[i % 12], float(lv))
    return e


def segments(d, x0, y0, w, h, rms, seg=7, gap=3):
    """Segmented level meter, with a taller outlined segment marking the silence threshold."""
    n = int(w // (seg + gap))
    fill_n = round(_meter_frac(rms) * n)
    thr_i = round(_meter_frac(cd.SILENCE_RMS) * n)
    for i in range(n):
        x = x0 + i * (seg + gap)
        if i == thr_i:
            d.rectangle([x, y0 - 5, x + seg - 1, y0 + h + 4], outline=INK, width=1)
        if i < fill_n:
            d.rectangle([x, y0, x + seg - 1, y0 + h], fill=INK)
        else:
            d.rectangle([x, y0, x + seg - 1, y0 + h], outline=INK)


def tape(sc):
    """Pitch class x time dot matrix. sc.rows: up to 16 chroma rows, oldest first."""
    W, H = UI_W, UI_H
    img = Image.new("1", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    draw_label(d, (14, 10), sc.label, 190, 40, 46)
    text_box(d, (W - 14, 12), "TAPE", fnt("mono", 12), "r")
    text_box(d, (W - 14, 30), sc.text, fnt("mono", 12), "r")
    d.rectangle([12, 56, W - 12, 58], fill=INK)

    mx, y0 = 18, 68
    rows = list(sc.rows)[-16:]
    rows = [np.zeros(12)] * (16 - len(rows)) + rows
    px = (W - 2 * mx) / 12
    py = (262 - y0) / len(rows)
    root = root_pc_of(sc.label)
    for r, row in enumerate(rows):
        cy = y0 + (r + 0.5) * py
        for c in range(12):
            cx = mx + (c + 0.5) * px
            e = row[c]
            if e >= 0.55:
                rad = 2.0 + 4.0 * (e - 0.55) / 0.45
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=INK)
            else:
                d.rectangle([cx - 1, cy - 1, cx, cy], fill=INK)
    by = 262 - py                                       # bracket around the newest row
    for x in (mx - 6, W - mx + 6):
        d.line([(x, by), (x, 262)], fill=INK, width=2)
    for y in (by, 262):
        d.line([(mx - 6, y), (mx - 2, y)], fill=INK, width=2)
        d.line([(W - mx + 2, y), (W - mx + 6, y)], fill=INK, width=2)
    for c in range(12):
        cx = mx + (c + 0.5) * px
        if c == root:
            d.rectangle([cx - 13, 270, cx + 13, 287], fill=INK)
            text_box(d, (cx, 279), cd.NOTE_NAMES[c], fnt("mono", 12), "m", fill=PAPER)
        else:
            text_box(d, (cx, 279), cd.NOTE_NAMES[c], fnt("mono", 12), "m")
    return img


def poster(sc):
    """Huge chord name, note pills, pitch-class lights and a segmented level meter."""
    W, H = UI_W, UI_H
    img = Image.new("1", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    for y in range(6, H, 12):                          # graph-paper dots
        for x in range(6, W, 12):
            d.point((x, y), fill=INK)

    draw_label(d, (16, 14), sc.label, W - 40, 112, 130)

    x = 16
    bass = min(sc.notes) if sc.notes else None
    for n in sc.notes:
        s = cd.midi_name(n)
        f = fnt("mono", 14)
        pw = d.textbbox((0, 0), s, font=f)[2] + 20
        is_bass = n == bass
        d.rounded_rectangle([x, 144, x + pw, 170], radius=13, fill=INK if is_bass else PAPER,
                            outline=INK, width=2)
        text_box(d, (x + pw / 2, 157), s, f, "m", fill=PAPER if is_bass else INK)
        x += pw + 8

    active = {n % 12 for n in sc.notes}
    root = root_pc_of(sc.label)
    for pc in range(12):
        x0 = 17 + pc * 31
        if pc in active:
            d.rectangle([x0, 188, x0 + 24, 212], fill=INK)
            if pc == root:
                d.rectangle([x0 + 8, 196, x0 + 16, 204], fill=PAPER)
        else:
            d.rectangle([x0, 188, x0 + 24, 212], fill=PAPER, outline=INK, width=2)
        text_box(d, (x0 + 12, 224), cd.NOTE_NAMES[pc], fnt("mono", 11), "m")

    text_box(d, (17, 244), "IN", fnt("mono", 11))
    segments(d, 17, 258, W - 34, 14, sc.rms)
    text_box(d, (17, 284), "RECENT  " + "  ".join(reversed(list(sc.history)[-3:])), fnt("mono", 12))
    return img


def render_mode(mode, sc, size=(400, 300), fifths=False):
    """Render one frame in the chosen UI. sc has notes, label, text, history, levels, rms, rows."""
    if mode == "wheel" or size != (UI_W, UI_H):        # poster/tape are laid out for 400x300
        return render(sc.notes, sc.label, sc.text, sc.history, size, fifths, sc.levels, sc.rms)
    return {"poster": poster, "tape": tape}[mode](sc)


# ---------- preview / demo / live ----------
def synth_levels(notes, noise=0.003, seed=0, amp=0.3):
    """(per-key levels, frame RMS) for a synthetic frame of the given notes plus mic noise."""
    rng = np.random.default_rng(seed)
    y = np.zeros(cd.N_FFT)
    for n in notes:
        y = y + cd.piano_like(cd.midi_to_hz(n), cd.N_FFT / cd.SR)[: cd.N_FFT]
    if notes:
        y = y / np.max(np.abs(y)) * amp
    y = (y + rng.normal(0, noise, cd.N_FFT)).astype(np.float32)
    return cd.key_levels(y), float(np.sqrt(np.mean(y ** 2)))


def demo_rows():
    """16 chroma rows for previews: Am, F, G, C, four rows each (oldest first)."""
    rows = []
    for notes in ([57, 60, 64], [53, 57, 60], [55, 59, 62], [60, 64, 67]):
        lv, _ = synth_levels(notes)
        rows += [fold(lv) * (1 - 0.04 * k) for k in range(4)]
    return rows


def sample_states():
    ch = cd.chord_notes
    rows = [
        (ch(0, ""), "C", "C4 E4 G4", ["Am", "F", "G"], ch(0, ""), 0.3),
        ([57, 60, 64], "Am", "A3 C4 E4", ["C", "F", "G"], [57, 60, 64], 0.3),
        ([64, 67, 72], "C/E", "E4 G4 C5", ["Am", "F", "G"], [64, 67, 72], 0.3),
        ([55, 59, 62, 65], "G7", "G3 B3 D4 F4", ["C", "Am", "F"], [55, 59, 62, 65], 0.3),
        ([60, 64], "C4 E4", "C4 E4", ["C", "F", "G"], [60, 64], 0.3),
        # played too quietly to pass the silence threshold: shown as "listening"
        ([], "-", "", ["C", "F", "G"], [60, 64, 67], 0.02),
    ]
    out = []
    for notes, label, text, hist, played, amp in rows:
        levels, rms = synth_levels(played, amp=amp)
        out.append((notes, label, text, hist, levels, rms))
    return out


def save_preview(path, size, fifths, scale, mode="wheel"):
    imgs = []
    for n, lab, txt, hist, lv, r in sample_states():
        rows = demo_rows()
        rows[-1] = fold(lv)
        sc = SimpleNamespace(notes=n, label=lab, text=txt, history=hist, levels=lv, rms=r, rows=rows)
        imgs.append(render_mode(mode, sc, size, fifths))
    cols = 2
    rows_n = (len(imgs) + cols - 1) // cols
    gap = 10
    sheet = Image.new("1", (cols * size[0] + (cols + 1) * gap, rows_n * size[1] + (rows_n + 1) * gap), PAPER)
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
    make = getattr(pygame.image, "frombytes", None) or pygame.image.fromstring
    return make(big.tobytes(), big.size, "RGB")


def run_window(source, size, fifths, scale, mode="wheel"):
    """source: generator yielding (notes, label, text, levels, rms) per update (None = no update).

    N switches UI (wheel, poster, tape); Q or Esc quits.
    """
    import pygame
    modes = MODES if size == (UI_W, UI_H) else ("wheel",)    # poster/tape are landscape only
    mode_idx = modes.index(mode) if mode in modes else 0
    if len(modes) == 1 and mode != "wheel":
        print("poster/tape are landscape only; showing the wheel", file=sys.stderr)

    def caption():
        hint = "N: next UI" if len(modes) > 1 else "wheel only in portrait"
        return f"Chord visualizer | {modes[mode_idx].upper()} | {hint}"

    pygame.init()
    screen = pygame.display.set_mode((size[0] * scale, size[1] * scale))
    pygame.display.set_caption(caption())
    clock = pygame.time.Clock()
    history, last = deque(maxlen=8), "-"
    state = ([], "-", "")
    shown_levels = np.zeros(NOTE_HI - NOTE_LO + 1)
    shown_rms = 0.0
    rows = deque([np.zeros(12)] * 16, maxlen=16)               # chroma history for TAPE
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                mode_idx = (mode_idx + 1) % len(modes)
                pygame.display.set_caption(caption())
        upd = next(source, "end")
        if upd == "end":
            break
        shown_levels *= 0.92                      # release: bars fall between updates
        shown_rms *= 0.92
        if upd is not None:
            state = upd[:3]
            shown_levels = np.maximum(shown_levels, upd[3])   # attack: rise at once
            shown_rms = max(shown_rms, upd[4])
            rows.append(fold(upd[3]))
            label = state[1]
            if label != "-" and label != last:
                history.append(label)
            last = label
        notes, label, text = state
        sc = SimpleNamespace(notes=notes, label=label, text=text, history=list(history),
                             levels=shown_levels, rms=shown_rms, rows=list(rows))
        img = render_mode(modes[mode_idx], sc, size, fifths)
        screen.blit(to_surface(pygame, img, scale), (0, 0))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def demo_source():
    seq = [[60, 64, 67], [64, 67, 72], [55, 59, 62, 65], [57, 60, 64], [53, 57, 60], [60, 64], [60], []]
    while True:
        for notes in seq:
            text, label = cd.describe(notes)
            levels, rms = synth_levels(notes)
            for i in range(45):
                yield (notes, label, text, levels, rms) if i % 4 == 0 else None


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
            yield None if res is None else (list(analyzer.shown_notes), res[0], res[1], analyzer.levels, res[2])


def main():
    ap = argparse.ArgumentParser(description="Pitch-class wheel visualizer (1-bit preview)")
    ap.add_argument("--png", help="render sample states to this PNG")
    ap.add_argument("--demo", action="store_true", help="animated demo, no microphone")
    ap.add_argument("--live", action="store_true", help="drive from the microphone")
    ap.add_argument("--device", help="input device index or name")
    ap.add_argument("--portrait", action="store_true", help="300x400 instead of 400x300")
    ap.add_argument("--fifths", action="store_true", help="circle-of-fifths order")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--ui", choices=MODES, default="wheel", help="starting UI (N switches in a window)")
    ap.add_argument("--silence", type=float)
    args = ap.parse_args()
    size = (300, 400) if args.portrait else (400, 300)
    if args.silence is not None:
        cd.SILENCE_RMS = args.silence

    if args.png:
        save_preview(args.png, size, args.fifths, args.scale, args.ui)
    elif args.demo:
        run_window(demo_source(), size, args.fifths, args.scale, args.ui)
    elif args.live:
        dev = int(args.device) if args.device and args.device.isdigit() else args.device
        run_window(live_source(dev), size, args.fifths, args.scale, args.ui)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()