"""
Nzsqh.py — an ASCII envelope appears, a cursor clicks the heart seal,
the letter opens, a red rose blooms and "For You" is handwritten below it.

Setup (once):
    pip install numpy pillow

Run:
    python Nzsqh.py            -> live window (pre-renders for a few seconds, then plays smoothly)
    python Nzsqh.py --save     -> renders Nzsqh.mp4 (requires: pip install imageio-ffmpeg)
"""
import argparse, io, math, re, shutil, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------- settings
W, H, FPS, DURATION = 720, 1280, 30, 17.0
CW, CH = 7, 13
COLS, ROWS = W // CW, H // CH
OX, OY = (W - COLS * CW) // 2, 2
SIGNATURE = "@you · 2026"
MESSAGE = "For You"                    # the handwritten word under the rose

# key moments (seconds)
T_ENVELOPE, T_HEART, T_CURSOR, T_CLICK = 0.2, 2.2, 3.0, 4.6
T_FLAP, T_DISSOLVE, T_STEM, T_BLOOM, T_WORD = 4.9, 6.4, 7.0, 8.2, 11.6

PAPER = np.array([0.95, 0.90, 0.84])
HEART_PINK = np.array([1.0, 0.40, 0.78])
VIOLET = np.array([0.66, 0.40, 1.0])
RED = np.array([1.0, 0.10, 0.16])
GREEN = np.array([0.30, 0.85, 0.40])

LEVELS = [".,:'`", ";-~=+<>", "()[]{}|/\\", "aeonuvxzcsr", "$&8%#@B0QS"]
EXTRA = "♥·o*+|/\\-_()v^●°"

# ---------------------------------------------------------------- fonts
def find_font(size):
    for name in ["DejaVuSansMono.ttf", "consola.ttf", "Consolas.ttf", "Menlo.ttc",
                 "LiberationMono-Regular.ttf", "cour.ttf", "Courier New.ttf"]:
        for base in ["", "/usr/share/fonts/truetype/dejavu/", "C:/Windows/Fonts/",
                     "/System/Library/Fonts/", "/Library/Fonts/",
                     "/usr/share/fonts/truetype/liberation/"]:
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                pass
    return ImageFont.load_default()

FONT, FONT_HUD = find_font(12), find_font(14)
CHARS = " " + "".join(sorted(set("".join(LEVELS) + EXTRA)))
IDX = {c: i for i, c in enumerate(CHARS)}

def build_atlas():
    g = np.zeros((len(CHARS), CH, CW), np.float32)
    for i, c in enumerate(CHARS):
        im = Image.new("L", (CW, CH))
        ImageDraw.Draw(im).text((CW / 2, CH / 2), c, font=FONT, fill=255, anchor="mm")
        g[i] = np.asarray(im) / 255.0
    return g

ATLAS = build_atlas()
LMAX = max(len(s) for s in LEVELS)
LUT = np.array([[IDX[s[k % len(s)]] for k in range(LMAX)] for s in LEVELS])
LENS = np.array([len(s) for s in LEVELS])

def smooth(a, b, x):
    x = np.clip((x - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)

def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

# ---------------------------------------------------------------- envelope
EC, ER = COLS // 2, 46            # envelope center (column, row)
EW, EH = 38, 15                   # half width, half height (in cells)
TIP = 0.58                        # flap depth

def envelope(opening, noise):
    """Returns (chars, colors, mask) for the envelope with its flap opened by 'opening' (0..1)."""
    r, c = np.mgrid[0:ROWS, 0:COLS]
    u = (c - EC) / EW                     # -1..1
    v = (r - (ER - EH)) / (2 * EH)        # 0 at the top, 1 at the bottom
    inside = (np.abs(u) <= 1) & (v >= 0) & (v <= 1)
    tip = TIP * (1 - 2 * opening)
    top_edge = TIP * (1 - np.abs(u))
    bottom_edge = 1 - 0.45 * (1 - np.abs(u))

    lum = np.zeros((ROWS, COLS))
    lum[inside] = 0.5                                    # sides
    lum[inside & (v > bottom_edge)] = 0.34               # bottom flap
    under = inside & (v < top_edge)
    lum[under] = 0.62 if opening == 0 else 0.25          # under the flap = dark inside
    if tip >= 0:
        flap = (np.abs(u) <= 1) & (v >= 0) & (v < tip * (1 - np.abs(u)))
        lum[flap] = 0.9
    else:
        flap = (np.abs(u) <= 1) & (v < 0) & (v > tip * (1 - np.abs(u)))
        lum[flap] = 0.38
    lum *= 0.85 + 0.3 * noise
    mask = inside | flap

    lvl = np.clip((lum * 5).astype(int), 0, 4)
    chars = LUT[lvl, (noise * 997).astype(int) % LENS[lvl]]
    # fold lines (empty cells) and //// borders
    fold = inside & (np.abs(v - bottom_edge) * 2 * EH < 0.9) & (np.abs(u) < 0.98)
    flap_edge = flap & (np.abs(v - tip * (1 - np.abs(u))) * 2 * EH < 1.1)
    chars[fold | flap_edge] = 0
    border = inside & ((np.abs(np.abs(u) - 1) * EW < 0.5) | (v * 2 * EH < 0.5) | ((1 - v) * 2 * EH < 0.5))
    chars[border] = IDX["/"]
    lum[border] = 0.8
    return chars, lum[..., None] * PAPER, mask

def heart_mask(scale):
    r, c = np.mgrid[0:ROWS, 0:COLS]
    x = (c - EC) / (11.0 * scale)
    y = -(r - (ER + 1.5)) / (6.2 * scale)
    return (x * x + y * y - 1) ** 3 - x * x * y ** 3 <= 0, y

# ---------------------------------------------------------------- 3D rose
def rose():
    """The classic parametric rose (a spiral surface of petals)."""
    x, th = np.meshgrid(np.linspace(0, 1, 50), np.linspace(-2 * np.pi, 15 * np.pi, 1800))
    phi = (np.pi / 2) * np.exp(-th / (8 * np.pi))
    X = 1 - 0.5 * ((5 / 4) * (1 - np.mod(3.6 * th, 2 * np.pi) / np.pi) ** 2 - 1 / 4) ** 2
    y = 1.95653 * x ** 2 * (1.27689 * x - 1) ** 2 * np.sin(phi)
    r = X * (x * np.sin(phi) + y * np.cos(phi))
    h = X * (x * np.cos(phi) - y * np.sin(phi))
    P = np.stack([r * np.sin(th), h, r * np.cos(th)], -1)
    du, dv = np.gradient(P, axis=0), np.gradient(P, axis=1)
    N = normalize(np.cross(du, dv))
    return P.reshape(-1, 3), N.reshape(-1, 3), th.ravel(), x.ravel()

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

HEAD = (W / 2, 400.0)                 # where the flower sits on screen
TILT = 55
STEM_BOTTOM = 830.0

# ---------------------------------------------------------------- renderer
class Renderer:
    def __init__(self):
        rng = np.random.default_rng(3)
        self.rng = np.random.default_rng(5)
        self.noise = rng.uniform(0, 1, (ROWS, COLS))
        r, c = np.mgrid[0:ROWS, 0:COLS]
        dist = np.hypot((c - EC) / EW, (r - ER) / EH)
        self.birth = T_ENVELOPE + 1.7 * (c - EC + EW) / (2 * EW) + 0.25 * self.noise   # left -> right sweep
        self.melt = T_DISSOLVE + 1.0 * dist + 0.3 * self.noise                         # center -> outside
        self.vel = np.stack([(c - EC) * 0.06 + rng.normal(0, 1.5, c.shape),
                             (r - ER) * 0.05 + rng.normal(-1.5, 1.5, c.shape)], -1)
        self.RP, self.RN, self.Rth, self.Rr = rose()
        self.Rseed = rng.integers(0, 10_000, len(self.RP))
        self.stars = (rng.integers(0, ROWS, 260), rng.integers(0, COLS, 260),
                      rng.uniform(0, 6.28, 260), rng.choice(list(".·+*"), 260))
        src = open(__file__, encoding="utf-8").read().splitlines()
        self.code = src[next(i for i, l in enumerate(src) if l.startswith("def envelope")):]

    # -- small helper: put letters in the grid
    def put(self, rr, cc, chars, cols, only_empty=False):
        ok = (rr >= 0) & (rr < ROWS) & (cc >= 0) & (cc < COLS)
        rr, cc = rr[ok], cc[ok]
        chars = np.broadcast_to(chars, ok.shape)[ok]
        cols = np.broadcast_to(cols, ok.shape + (3,))[ok]
        if only_empty:
            m = self.G_char[rr, cc] == 0
            rr, cc, chars, cols = rr[m], cc[m], chars[m], cols[m]
        self.G_char[rr, cc] = chars
        self.G_col[rr, cc] = cols

    def image(self, t):
        self.G_char = np.zeros((ROWS, COLS), int)
        self.G_col = np.zeros((ROWS, COLS, 3), np.float32)
        opening = float(smooth(T_FLAP, T_FLAP + 1.2, t))

        # ---- the envelope (and how it dissolves into particles)
        if t < T_DISSOLVE + 3:
            chars, cols, mask = envelope(opening, self.noise)
            age = t - self.birth
            vis = mask & (age >= 0)
            new = vis & (age < 0.3)
            chars[new] = LUT[4, self.rng.integers(0, 100, new.sum()) % LENS[4]]
            cols[new] = HEART_PINK * 1.4
            solid = vis & (t < self.melt)
            r, c = np.nonzero(solid)
            self.put(r, c, chars[solid], cols[solid])
            dust = vis & (t >= self.melt)
            if dust.any():
                r, c = np.nonzero(dust)
                a = t - self.melt[dust]
                v = self.vel[dust]
                pc = (c + v[:, 0] * a * 2.2).astype(int)
                pr = (r + v[:, 1] * a * 2.2 + 5 * a * a).astype(int)
                f = np.clip(1 - a / 1.1, 0, 1)[:, None] ** 1.5
                self.put(pr, pc, chars[dust], cols[dust] * f * 1.3 + HEART_PINK * 0.4 * f * (1 - f))
            # pink glow coming out of the open letter
            if opening > 0.3 and t < T_DISSOLVE + 1.5:
                g = float(smooth(T_FLAP + 0.4, T_DISSOLVE, t)) * (1 - float(smooth(T_DISSOLVE, T_DISSOLVE + 1.5, t)))
                rows, cols_ = np.mgrid[0:ROWS, 0:COLS]
                rr, cc = np.nonzero(mask & (np.hypot((cols_ - EC) / EW, (rows - (ER - EH)) / EH) < 0.8))
                self.G_col[rr, cc] += HEART_PINK * 0.35 * g

        # ---- the pink heart (the seal)
        if T_HEART <= t < T_CLICK + 1.6:
            appear = float(smooth(T_HEART, T_HEART + 0.6, t))
            hover = float(smooth(T_CLICK - 0.6, T_CLICK - 0.1, t))
            scale = 1.12 if t >= T_CLICK else appear * (1 + 0.05 * math.sin(t * 6) + 0.12 * hover)
            m, y = heart_mask(max(scale, 0.05))
            r, c = np.nonzero(m)
            mix = np.clip((y[m] + 1) / 2.2, 0, 1)[:, None]
            col = (HEART_PINK * mix + VIOLET * (1 - mix)) * (1.4 + 0.6 * hover)
            ch = LUT[4, (self.noise[m] * 997).astype(int) % LENS[4]]
            ch[self.noise[m] > 0.82] = IDX["♥"]
            if t < T_CLICK + 0.15:
                self.put(r, c, ch, col)
            else:                                     # the heart shatters
                a = t - T_CLICK - 0.15
                ang = self.noise[m] * 6.28 + c * 0.3
                pc = (c + np.cos(ang) * a * 26 * (0.5 + self.noise[m])).astype(int)
                pr = (r + np.sin(ang) * a * 14 * (0.5 + self.noise[m]) + 10 * a * a).astype(int)
                self.put(pr, pc, ch, col * max(0.0, 1 - a / 1.4))

        # ---- click ripple
        if T_CLICK <= t < T_CLICK + 0.9:
            a = (t - T_CLICK) / 0.9
            ang = np.linspace(0, 2 * np.pi, 140)
            rr = (ER + 1.5 + np.sin(ang) * a * 14).astype(int)
            cc = (EC + np.cos(ang) * a * 26).astype(int)
            self.put(rr, cc, IDX["·"], HEART_PINK * 1.5 * (1 - a))

        # ---- the rose
        if t >= T_STEM:
            self.draw_rose(t)

        # ---- stars
        sr, sc, ph, ch = self.stars
        self.put(sr, sc, np.array([IDX[x] for x in ch]),
                 (0.12 + 0.12 * np.sin(2 * t + ph))[:, None] * np.ones(3), only_empty=True)

        img = ATLAS[self.G_char][..., None] * self.G_col[:, :, None, None, :]
        img = img.transpose(0, 2, 1, 3, 4).reshape(ROWS * CH, COLS * CW, 3)
        canvas = np.zeros((H, W, 3), np.float32)
        canvas[OY:OY + ROWS * CH, OX:OX + COLS * CW] = img
        im = Image.fromarray((glow(canvas) * 255).astype(np.uint8))
        self.cursor(im, t)
        self.hud(im, t)
        return np.asarray(im)

    def draw_rose(self, t):
        hx, hy = HEAD
        # the stem grows from the bottom up
        grow = float(smooth(T_STEM, T_STEM + 1.3, t))
        ys = np.arange(STEM_BOTTOM, STEM_BOTTOM - (STEM_BOTTOM - hy - 60) * grow, -CH / 2)
        xs = hx + 14 * np.sin((ys - hy) / 90)
        slope = np.gradient(xs) if len(xs) > 1 else np.zeros_like(xs)
        chars = np.where(np.abs(slope) < 1.2, IDX["|"], np.where(slope > 0, IDX["/"], IDX["\\"]))
        self.put(((ys - OY) / CH).astype(int), ((xs - OX) / CW).astype(int), chars, GREEN * 0.9)
        # two leaves
        for (yb, side, start) in [(690, -1, T_STEM + 0.8), (620, 1, T_STEM + 1.1)]:
            f = float(smooth(start, start + 0.7, t))
            if f <= 0: continue
            xb = hx + 14 * np.sin((yb - hy) / 90)
            s = np.linspace(0, 1, 60)[:, None]; w = np.linspace(-1, 1, 9)[None, :]
            lx = xb + side * s * 95 * f + 0 * w
            ly = yb - s * 45 * f + w * 13 * f * np.sin(np.pi * s)
            c = np.where(np.abs(w) < 0.2, IDX["-"], IDX["/" if side > 0 else "\\"]) + 0 * s.astype(int)
            lum = (0.5 + 0.5 * (1 - np.abs(w))) * np.ones_like(s)
            self.put(((ly - OY) / CH).astype(int).ravel(), ((lx - OX) / CW).astype(int).ravel(),
                     c.ravel(), (lum[..., None] * GREEN).reshape(-1, 3) * 0.9)

        # flower head: petals unfold from the center outwards
        bloom = float(smooth(T_BLOOM, T_BLOOM + 3.5, t))
        if bloom <= 0: return
        vis = self.Rth >= 15 * np.pi - bloom * 15.6 * np.pi
        scale = 128 * (0.35 + 0.65 * bloom)
        M = rot_x(math.radians(TILT)) @ rot_y(t * 0.45)
        P = self.RP[vis] @ M.T; N = self.RN[vis] @ M.T
        P[:, 1] -= 0.45
        L = normalize(np.array([-0.4, 0.7, 0.6]))
        rim = self.Rr[vis]
        lum = (0.15 + 0.7 * np.abs(N @ L) ** 1.5) * (0.35 + 0.65 * rim ** 2) + 0.5 * rim ** 8
        s = scale * 6 / (6 - P[:, 2])
        px, py = hx + P[:, 0] * s, hy - P[:, 1] * s
        ix, iy = ((px - OX) / CW).astype(int), ((py - OY) / CH).astype(int)
        ok = (ix >= 0) & (ix < COLS) & (iy >= 0) & (iy < ROWS)
        key = iy[ok] * COLS + ix[ok]
        order = np.lexsort((-P[ok, 2], key))
        _, first = np.unique(key[order], return_index=True)
        sel = np.flatnonzero(ok)[order[first]]          # z-buffer: closest point per cell
        lvl = np.clip((lum[sel] / 0.24).astype(int), 0, 4)
        chars = LUT[lvl, self.Rseed[vis][sel] % LENS[lvl]]
        red = RED * lum[sel, None] + np.array([0.5, 0.25, 0.3]) * np.clip(lum[sel] - 0.8, 0, 1)[:, None]
        # petal outlines: a cell right behind a closer petal gets darker
        zg = np.full((ROWS + 2, COLS + 2), -9.0)
        zg[iy[sel] + 1, ix[sel] + 1] = P[sel, 2]
        z0 = zg[iy[sel] + 1, ix[sel] + 1]
        front = np.zeros(len(sel))
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            front = np.maximum(front, zg[iy[sel] + 1 + dy, ix[sel] + 1 + dx] - z0)
        gap = front > 0.09
        chars[gap] = IDX["."]
        red[gap] *= 0.3
        self.G_char[iy[sel], ix[sel]] = chars
        self.G_col[iy[sel], ix[sel]] = red * 1.15
        # a few sparkles
        k = self.rng.choice(sel, min(len(sel), 5), replace=False)
        self.G_char[iy[k], ix[k]] = IDX["*"]; self.G_col[iy[k], ix[k]] = [1.6, 1.0, 1.1]

    def cursor(self, im, t):
        if not (T_CURSOR <= t < T_FLAP + 1.2): return
        target = np.array([OX + EC * CW + CW / 2, OY + (ER + 1.5) * CH])
        start = np.array([W + 40.0, 900.0])
        a = float(smooth(T_CURSOR, T_CLICK - 0.2, t))
        p = start + (target - start) * a + np.array([0, -60 * math.sin(math.pi * a)])
        if t > T_FLAP:                                  # leaves after the click
            p = p + np.array([1, 0.6]) * 260 * float(smooth(T_FLAP, T_FLAP + 1.2, t))
        e = 1.3 * (0.8 if T_CLICK <= t < T_CLICK + 0.18 else 1.0)
        shape = [(0, 0), (0, 26), (7, 20), (12, 31), (16, 29), (11, 19), (19, 19)]
        ImageDraw.Draw(im).polygon([(p[0] + x * e, p[1] + y * e) for x, y in shape],
                                   fill=(250, 250, 250), outline=(20, 20, 20))

    def hud(self, im, t):
        handwrite(im, t)
        dr = ImageDraw.Draw(im)
        grey = (120, 120, 128)
        y = 970
        state = "sealed" if t < T_CLICK else "opening" if t < T_STEM else "blooming" if t < T_BLOOM + 3.5 else "for you"
        dr.text((16, y), SIGNATURE, font=FONT_HUD, fill=grey)
        dr.text((16, y + 22), f"letter  {state}", font=FONT_HUD, fill=grey)
        dr.text((16, y + 44), f"letters {(self.G_char > 0).sum():,}", font=FONT_HUD, fill=grey)
        dr.text((8, y + 76), "py Nzsqh.py", font=FONT_HUD, fill=(170, 170, 170))
        dr.text((W - 8, y + 76), f"running · {FPS} fps", font=FONT_HUD, fill=(110, 110, 110), anchor="ra")
        dr.line([(0, y + 100), (W, y + 100)], fill=(40, 40, 46))
        type_code(dr, self.code, t, y + 110)

def glow(img):
    p = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    p = p.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR)
    img = img + 0.9 * np.asarray(p, np.float32) / 255
    return 1 - np.exp(-1.3 * img)

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

# ================================================================ handwritten "For You"
def script_font(size):
    for name in ["segoesc.ttf", "Gabriola.ttf", "BRUSHSCI.TTF", "Apple Chancery.ttf", "SnellRoundhand.ttc",
                 "texgyrechorus-mediumitalic.otf", "DejaVuSerif-Italic.ttf", "LiberationSerif-Italic.ttf", "georgiai.ttf"]:
        for folder in ["", "C:/Windows/Fonts/", "/System/Library/Fonts/Supplemental/", "/Library/Fonts/",
                       "/usr/share/texmf/fonts/opentype/public/tex-gyre/", "/usr/share/fonts/truetype/dejavu/",
                       "/usr/share/fonts/truetype/liberation/"]:
            try:
                return ImageFont.truetype(folder + name, size)
            except OSError:
                pass
    return find_font(size)

_INK = None

def ink():
    """The text rendered once as a greyscale mask."""
    global _INK
    if _INK is None:
        m = Image.new("L", (W, 200))
        ImageDraw.Draw(m).text((W / 2, 100), MESSAGE, font=script_font(92), fill=255, anchor="mm")
        a = np.asarray(m, np.float32) / 255
        cols = np.flatnonzero(a.max(0) > 0.1)
        _INK = (a, cols.min(), cols.max())
    return _INK

def handwrite(im, t):
    if t < T_WORD: return
    a, x0, x1 = ink()
    p = float(smooth(T_WORD, T_WORD + 2.4, t))
    edge = x0 + (x1 - x0 + 30) * p
    xs = np.arange(W)
    shown = a * np.clip((edge - xs) / 14, 0, 1)[None, :]
    beat = 1 + 0.08 * math.sin(6 * t) * float(smooth(T_WORD + 2.4, T_WORD + 3, t))
    layer = shown[..., None] * (np.array([1.0, 0.22, 0.32]) * beat)
    if p < 1:                                             # the glowing pen tip
        c = int(min(edge, x1))
        ys = np.flatnonzero(a[:, max(c - 3, 0):c + 1].max(1) > 0.3)
        if len(ys):
            yy, r = ys.mean(), 7
            Y, X = np.ogrid[:200, :W]
            layer += (np.exp(-((X - c) ** 2 + (Y - yy) ** 2) / (2 * r * r)))[..., None] * np.array([1.0, 0.8, 0.85])
    blur = np.asarray(Image.fromarray((np.clip(layer, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)), np.float32) / 255
    y0 = 800
    base = np.asarray(im, np.float32)[y0:y0 + 200] / 255
    out = 1 - (1 - base) * (1 - np.clip(layer + 1.2 * blur, 0, 1))
    im.paste(Image.fromarray((out * 255).astype(np.uint8)), (0, y0))

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
    root.title("For You")
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
