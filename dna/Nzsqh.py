"""
Nzsqh.py — a DNA double helix building itself in 3D ASCII.
Two backbones twist upwards and every base pair (A-T, G-C) snaps into place.

Setup (once):
    pip install numpy pillow

Run:
    python Nzsqh.py            -> live window (pre-renders for a few seconds, then plays smoothly)
    python Nzsqh.py --save     -> renders Nzsqh.mp4 (requires: pip install imageio-ffmpeg)
"""
import argparse, base64, io, math, re, shutil, subprocess, sys, time, zlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------- canvas
W, H, FPS = 720, 1280, 30
SIGNATURE = "@you · 2026"
CW, CH = 5, 9
COLS, ROWS = W // CW, H // CH
OX = (W - COLS * CW) // 2
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

def find_font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSansMono-Bold.ttf", "consolab.ttf"] if bold else []
    names += ["DejaVuSansMono.ttf", "consola.ttf", "Consolas.ttf", "Menlo.ttc",
              "LiberationMono-Regular.ttf", "cour.ttf", "Courier New.ttf"]
    for name in names:
        for base in ["", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/",
                     "/System/Library/Fonts/", "/Library/Fonts/", "/usr/share/fonts/truetype/liberation/"]:
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                pass
    return ImageFont.load_default()

FONT_HUD = find_font(14)
CHARS = " " + "".join(sorted(set(RAMP + "·✦♥*+=") - {" "}))
IDX = {c: i for i, c in enumerate(CHARS)}
RAMP_IDX = np.array([IDX[c] for c in RAMP])

def build_atlas():
    f = find_font(9)
    a = np.zeros((len(CHARS), CH, CW), np.float32)
    for c, i in IDX.items():
        im = Image.new("L", (CW, CH))
        ImageDraw.Draw(im).text((CW / 2, CH / 2), c, font=f, fill=255, anchor="mm")
        a[i] = np.asarray(im) / 255.0
    return a

ATLAS = build_atlas()

def smooth(a, b, x):
    x = np.clip((x - a) / (b - a), 0, 1)
    return float(x * x * (3 - 2 * x)) if np.ndim(x) == 0 else x * x * (3 - 2 * x)

def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def cells(px, py):
    return np.floor((px - OX) / CW).astype(int), np.floor(py / CH).astype(int)

def accumulate(px, py, lum, col):
    """Additive light: every point adds its brightness and colour to its cell."""
    ix, iy = cells(px, py)
    ok = (ix >= 0) & (ix < COLS) & (iy >= 0) & (iy < ROWS)
    key = iy[ok] * COLS + ix[ok]
    n = ROWS * COLS
    L = np.bincount(key, lum[ok], n)
    C = np.stack([np.bincount(key, lum[ok] * col[ok, k], n) for k in range(3)], -1)
    return L.reshape(ROWS, COLS), C.reshape(ROWS, COLS, 3)

def tone(L, C, exposure, gamma=0.8):
    """Light per cell -> a letter (denser = brighter) and a colour."""
    v = 1 - np.exp(-L * exposure)
    chars = RAMP_IDX[np.clip((v ** gamma * (len(RAMP) - 1)).astype(int), 0, len(RAMP) - 1)]
    cols = C / (L[..., None] + 1e-9) * (0.25 + 1.0 * v[..., None])
    return chars, cols.astype(np.float32)

def zbuffer(px, py, z):
    """Solid surfaces: keep the closest point in each cell."""
    ix, iy = cells(px, py)
    ok = (ix >= 0) & (ix < COLS) & (iy >= 0) & (iy < ROWS)
    key = iy[ok] * COLS + ix[ok]
    order = np.lexsort((-z[ok], key))
    _, first = np.unique(key[order], return_index=True)
    sel = np.flatnonzero(ok)[order[first]]
    return sel, ix[sel], iy[sel]

def to_pixels(chars, cols):
    img = ATLAS[chars][..., None] * cols[:, :, None, None, :]
    img = img.transpose(0, 2, 1, 3, 4).reshape(ROWS * CH, COLS * CW, 3)
    canvas = np.zeros((H, W, 3), np.float32)
    canvas[:ROWS * CH, OX:OX + COLS * CW] = img
    return canvas

def glow(img, k=0.9, blur=5):
    p = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    p = p.filter(ImageFilter.GaussianBlur(blur)).resize((W, H), Image.BILINEAR)
    img = img + k * np.asarray(p, np.float32) / 255
    return 1 - np.exp(-1.3 * img)

def caption(dr, text, t, t0, y, colour, size=34, spacing=4):
    """A short line typed letter by letter (used for the final message)."""
    if t < t0: return
    f = find_font(size)
    n = min(len(text), int((t - t0) * 12))
    widths = [f.getlength(c) + spacing for c in text]
    x = W / 2 - (sum(widths) - spacing) / 2
    a = smooth(t0, t0 + 0.6, t)
    for i in range(n):
        dr.text((x, y), text[i], font=f, fill=tuple(int(c * a) for c in colour), anchor="lm")
        x += widths[i]

def hud(dr, t, code, lines):
    grey = (120, 120, 132)
    y = 970
    for k, l in enumerate(lines):
        dr.text((16, y + 22 * k), l, font=FONT_HUD, fill=grey)
    dr.text((8, y + 76), "py Nzsqh.py", font=FONT_HUD, fill=(172, 172, 182))
    dr.text((W - 8, y + 76), f"running · {FPS} fps", font=FONT_HUD, fill=(108, 108, 118), anchor="ra")
    dr.line([(0, y + 100), (W, y + 100)], fill=(40, 40, 52))
    type_code(dr, code, t, y + 110)

def source_from(marker):
    lines = open(__file__, encoding="utf-8").read().splitlines()
    return lines[next(i for i, l in enumerate(lines) if l.startswith(marker)):]


# ================================================================ the scene
DURATION, TITLE = 18.0, "dna"
CX, CY = W / 2, 470.0
FOCAL = 1150.0
TURNS, LENGTH, RADIUS = 2.0, 11.0, 1.7     # a double helix: ~10 base pairs per turn

BASES = {"A": np.array([1.0, 0.35, 0.45]), "T": np.array([1.0, 0.80, 0.30]),
         "G": np.array([0.35, 0.85, 1.0]), "C": np.array([0.55, 1.0, 0.55])}
PAIR = {"A": "T", "T": "A", "G": "C", "C": "G"}
STRAND1, STRAND2 = np.array([0.75, 0.55, 1.0]), np.array([1.0, 0.45, 0.85])

def helix(n_pairs, rng):
    """Two backbones twisting around each other, and the base pairs that hold them together."""
    s = np.linspace(0, 1, 2400)
    a = 2 * np.pi * TURNS * s
    y = (s - 0.5) * LENGTH
    back1 = np.stack([RADIUS * np.cos(a), y, RADIUS * np.sin(a)], 1)
    back2 = np.stack([RADIUS * np.cos(a + 2.4), y, RADIUS * np.sin(a + 2.4)], 1)   # the minor/major grooves
    seq = rng.choice(list("ATGC"), n_pairs)
    rungs = []
    for i, b in enumerate(seq):
        u = (i + 0.5) / n_pairs
        ai = 2 * np.pi * TURNS * u
        p1 = np.array([RADIUS * np.cos(ai), (u - 0.5) * LENGTH, RADIUS * np.sin(ai)])
        p2 = np.array([RADIUS * np.cos(ai + 2.4), (u - 0.5) * LENGTH, RADIUS * np.sin(ai + 2.4)])
        rungs.append((u, p1, p2, b, PAIR[b]))
    return s, back1, back2, rungs

class Renderer:
    def __init__(self):
        rng = np.random.default_rng(23)
        self.s, self.b1, self.b2, self.rungs = helix(22, rng)
        self.dust = normalize(rng.normal(0, 1, (900, 3))) * rng.uniform(3, 9, (900, 1))
        self.code = source_from("def helix")

    def image(self, t):
        build = smooth(0.4, 8.0, t)                         # the molecule is built from bottom to top
        M = rot_x(math.radians(12 * math.sin(0.25 * t))) @ rot_y(0.55 * t) @ rot_z(math.radians(-10))
        X, Y, Lm, Cm = [], [], [], []

        def add(P, lum, col):
            Q = P @ M.T
            sc = FOCAL / (12.0 - Q[:, 2])
            X.append(CX + Q[:, 0] * sc * 0.76); Y.append(CY - Q[:, 1] * sc * 0.76)
            Lm.append(lum * np.clip(0.65 + Q[:, 2] / 3, 0.25, 1.4)); Cm.append(col)

        m = self.s <= build
        for B, col in ((self.b1, STRAND1), (self.b2, STRAND2)):
            P = B[m]
            if len(P):
                # each backbone is a thick tube: add a few offset copies
                for dx, dz in ((0, 0), (0.08, 0), (-0.08, 0), (0, 0.08), (0, -0.08)):
                    add(P + [dx, 0, dz], np.full(len(P), 0.3), np.tile(col, (len(P), 1)))
                head = self.s[m][-1] < 1
                if head:
                    tip = P[-3:]
                    add(tip, np.full(len(tip), 6.0), np.tile([1.0, 1.0, 1.0], (len(tip), 1)))
        for (u, p1, p2, b1, b2) in self.rungs:
            g = np.clip((build - u) / 0.04, 0, 1)
            if g <= 0: continue
            k = np.linspace(0, 1, 120)[:, None]
            mid = (p1 + p2) / 2
            left = p1 + (mid - p1) * k * g                   # each half grows from its backbone
            right = p2 + (mid - p2) * k * g
            snap = math.exp(-((build - u - 0.04) / 0.02) ** 2) if build < 1 else 0.0
            for P, b in ((left, b1), (right, b2)):
                for dy in (-0.05, 0.05):
                    add(P + [0, dy, 0], np.full(len(P), 0.75 + 3 * snap), np.tile(BASES[b] * (1 + snap), (len(P), 1)))

        D = self.dust @ rot_y(0.05 * t).T
        ds = FOCAL / (14.0 - D[:, 2])
        X.append(CX + D[:, 0] * ds * 0.76); Y.append(CY - D[:, 1] * ds * 0.76)
        Lm.append(np.full(len(D), 0.05) * smooth(1, 4, t)); Cm.append(np.tile([0.6, 0.6, 0.9], (len(D), 1)))

        L, C = accumulate(np.concatenate(X), np.concatenate(Y), np.concatenate(Lm), np.concatenate(Cm))
        L[int(955 / CH):] = 0
        chars, cols = tone(L, C, 0.5, 0.8)
        im = Image.fromarray((glow(to_pixels(chars, cols), 1.15, 6) * 255).astype(np.uint8))
        hud(ImageDraw.Draw(im), t, self.code, [])
        return np.asarray(im)

KEYWORDS = {"def", "for", "in", "return", "if", "else", "import", "np", "range", "not", "class", "self"}

def type_code(dr, code, t, y0):
    """The source code typing itself at the bottom of the screen."""
    left, lines = int(t * 40), []
    for l in code:
        if left <= 0: break
        lines.append(l[:left]); left -= len(l) + 1
    first = max(0, len(lines) - 7)
    for k, l in enumerate(lines[first:]):
        y = y0 + k * 24
        dr.text((10, y), f"{first + k + 70:3d}", font=FONT_HUD, fill=(70, 70, 80))
        x = 52
        for word in re.split(r"(\W)", l):
            if not word: continue
            c = (220, 110, 150) if word in KEYWORDS else (110, 160, 210) if word[:1].isdigit() else (165, 165, 175)
            dr.text((x, y), word, font=FONT_HUD, fill=c); x += FONT_HUD.getlength(word)
        if k == len(lines[first:]) - 1 and int(t * 3) % 2 == 0:
            dr.rectangle([x + 1, y + 2, x + 8, y + 17], fill=(230, 120, 160))

# ================================================================ playback / export
# Frames are too heavy to render live at 30 fps, so they are pre-rendered on every
# CPU core and then played back at exactly 30 fps.
_RENDERER, _SIZE = None, None

def _init_worker(size):
    global _RENDERER, _SIZE
    _RENDERER, _SIZE = Renderer(), size

def _render(i):
    im = Image.fromarray(_RENDERER.image(i / FPS))
    if _SIZE is None:
        return im.tobytes()
    buf = io.BytesIO()
    im.resize(_SIZE, Image.BILINEAR).save(buf, "JPEG", quality=92)
    return buf.getvalue()

def render_parallel(size):
    import multiprocessing as mp
    n = int(DURATION * FPS)
    cores = max(1, (mp.cpu_count() or 2) - 1)
    pool = mp.get_context("spawn").Pool(cores, _init_worker, (size,))
    return n, cores, pool.imap(_render, range(n), chunksize=4)

def window():
    import threading
    import tkinter as tk
    from PIL import ImageTk
    root = tk.Tk()
    root.title(TITLE)
    root.configure(bg="black")
    k = min(1.0, (root.winfo_screenheight() - 120) / H)
    size = (int(W * k), int(H * k))
    blank = tk.PhotoImage(width=1, height=1)
    view = tk.Label(root, bg="black", fg="#ff5a78", bd=0, font=("Consolas", 14),
                    width=size[0], height=size[1], image=blank, compound="center")
    view.image = blank
    view.pack()
    n, cores, stream = render_parallel(size)
    frames, started = [], time.time()
    threading.Thread(target=lambda: [frames.append(b) for b in stream], daemon=True).start()
    state = {"play": None}

    def far_enough():
        done = len(frames)
        if done >= n: return True
        if done < 30: return False
        speed = done / (time.time() - started)
        return done >= n * (1 - speed / FPS) * 1.1 + 30

    def tick():
        if state["play"] is None:
            if not far_enough():
                view.configure(text=f"rendering...  {100 * len(frames) // n} %\n\n({cores} cores working)")
                root.after(100, tick); return
            state["play"] = time.time()
            view.configure(text="")
        i = int((time.time() - state["play"]) * FPS)
        if i >= n + 2 * FPS:
            state["play"], i = time.time(), 0
        i = min(i, n - 1)
        if i >= len(frames):
            state["play"] += 1 / FPS
            i = len(frames) - 1
        photo = ImageTk.PhotoImage(Image.open(io.BytesIO(frames[i])))
        view.configure(image=photo); view.image = photo
        root.after(5, tick)

    root.bind("<Escape>", lambda e: root.destroy())
    root.after(50, tick)
    root.mainloop()
    import os
    os._exit(0)

def save(name):
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg") or sys.exit("To export the video, first run:  pip install imageio-ffmpeg")
    proc = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", name],
                            stdin=subprocess.PIPE)
    n, cores, stream = render_parallel(None)
    for i, b in enumerate(stream):
        proc.stdin.write(b)
        print(f"\rframe {i + 1}/{n}  ({cores} cores)", end="", flush=True)
    proc.stdin.close(); proc.wait()
    print(f"\nVideo ready -> {name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", nargs="?", const="Nzsqh.mp4")
    a = ap.parse_args()
    save(a.save) if a.save else window()
