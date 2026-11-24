"""
Nzsqh.py — a street lamp in the rain at night, in 3D ASCII.
The lamp flickers on, the puddles catch its light and the neon sign, and the city appears behind.

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
    lines = lines[next(i for i, l in enumerate(lines) if l.startswith(marker)):]
    out, doc = [], False
    for l in lines:
        s = l.strip()
        if s.startswith('"""'):
            if not (s.endswith('"""') and len(s) > 3):
                doc = not doc
            continue
        if doc or s.startswith("#"):
            continue
        l = re.sub(r"\s{2,}# .*$", "", l)
        if l.strip() or (out and out[-1].strip()):
            out.append(l)
    return out


# ================================================================ signed distance toolkit
def smin(a, b, k=0.06):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

def ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / r, axis=-1)
    return k0 * (k0 - 1) / (k1 + 1e-9)

def limb(p, a, b, ra, rb):
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba + 1e-12), 0, 1)
    return np.linalg.norm(pa - h[:, None] * ba, axis=-1) - (ra + (rb - ra) * h)

def hash3(p):
    h = np.sin(p[..., 0] * 127.1 + p[..., 1] * 311.7 + p[..., 2] * 74.7) * 43758.5453
    return h - np.floor(h)

def vnoise(p):
    i = np.floor(p); f = p - i; u = f * f * (3 - 2 * f)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1]) * (u[..., 2] if dz else 1 - u[..., 2])
                out = out + w * hash3(i + np.array([dx, dy, dz]))
    return out

def blend(parts, p, k=0.05, labels=False):
    ds = np.stack([f(p) for _, f in parts])
    d = ds[0]
    for x in ds[1:]:
        d = smin(d, x, k)
    return (d, ds.argmin(0)) if labels else d

class View:
    def __init__(self, ppu, cx, cy, yaw, pitch):
        self.ppu, self.cx, self.cy = ppu, cx, cy
        self.R = rot_x(pitch) @ rot_y(yaw)
        yy, xx = np.mgrid[0:ROWS, 0:COLS]
        self.X, self.Y = OX + (xx + 0.5) * CW, (yy + 0.5) * CH

    def to_screen(self, P):
        Q = np.atleast_2d(P) @ self.R.T
        return self.cx + Q[:, 0] * self.ppu, self.cy - Q[:, 1] * self.ppu, Q[:, 2]

    def rays(self, box):
        iy, ix = np.nonzero(box & (self.Y < 955))
        u = (self.X[iy, ix] - self.cx) / self.ppu
        v = (self.cy - self.Y[iy, ix]) / self.ppu
        o = np.stack([u, v, np.full(len(u), 4.0)], 1) @ self.R
        d = np.tile(np.array([0, 0, -1.0]) @ self.R, (len(u), 1))
        return iy, ix, o, d

    def box(self, x0, x1, y0, y1):
        u = (self.X - self.cx) / self.ppu
        v = (self.cy - self.Y) / self.ppu
        return (u > x0) & (u < x1) & (v > y0) & (v < y1)

def march(sdf, o, d, steps=70, eps=0.002, tmax=8.0):
    T = np.zeros(len(o)); hit = np.zeros(len(o), bool); alive = np.ones(len(o), bool)
    for _ in range(steps):
        idx = np.flatnonzero(alive)
        if not len(idx): break
        dist = sdf(o[idx] + d[idx] * T[idx, None])
        T[idx] += dist
        h = dist < eps
        hit[idx[h]] = True
        alive[idx[h | (T[idx] > tmax)]] = False
    return hit, T

def normals(sdf, P, e=0.003):
    return normalize(np.stack([sdf(P + np.eye(3)[a] * e) - sdf(P - np.eye(3)[a] * e) for a in range(3)], 1))

def studio_light(n, D, R, sdf, P, key=(-0.45, 0.65, 0.6), fill=(0.8, 0.1, 0.3),
                 KEY=np.array([1.0, 0.86, 0.70]), FILL=np.array([0.35, 0.45, 0.80]), RIM=np.array([0.65, 0.75, 1.0]),
                 rim=0.75, ambient=0.16):
    L1, L2 = normalize(np.array(key)) @ R, normalize(np.array(fill)) @ R
    ndv = np.clip(-(n * D).sum(1), 0, 1)
    ao = np.clip(sdf(P + n * 0.04) / 0.04, 0.25, 1)
    diffuse = KEY * np.clip(n @ L1, 0, 1)[:, None] ** 1.1 + FILL * 0.5 * np.clip(n @ L2, 0, 1)[:, None] + ambient
    return diffuse * ao[:, None], RIM * ((1 - ndv) ** 3 * rim * ao)[:, None]

def surface_letters(chars, cols, iy, ix, col, noise, gain=1.25, gamma=0.7):
    lum = np.clip(col.max(1), 0, 1.4)
    lvl = np.clip((lum ** gamma * (len(RAMP) - 1) + (noise[iy, ix] - 0.5) * 8).astype(int), 6, len(RAMP) - 1)
    chars[iy, ix] = RAMP_IDX[lvl]
    cols[iy, ix] = col * gain

def put(chars, cols, x, y, ch, col):
    c_, r_ = cells(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)))
    ok = (r_ >= 0) & (r_ < ROWS) & (c_ >= 0) & (c_ < COLS)
    chars[r_[ok], c_[ok]] = np.broadcast_to(ch, ok.shape)[ok]
    cols[r_[ok], c_[ok]] = np.broadcast_to(np.asarray(col, float), ok.shape + (3,))[ok]


# ================================================================ the scene
DURATION, TITLE = 20.0, "street lamp"
CY = 478.0
LAMP = np.array([0.8, 2.86, -0.5])
PINK = np.array([1.0, 0.22, 0.62])
WARM = np.array([1.0, 0.62, 0.30])
LAMPC = np.array([1.0, 0.80, 0.55])
TOWERS = [(-12, -7, -30, -22, 18), (-6, -1, -34, -26, 26), (0, 4, -28, -22, 14),
          (5, 11, -36, -26, 22), (-20, -13, -26, -18, 12), (12, 18, -24, -18, 15), (-30, -21, -40, -30, 20)]
LETTERS = {
    "R": [[(0, 0), (0, 1), (0.7, 1), (0.9, 0.85), (0.9, 0.65), (0.7, 0.5), (0, 0.5)], [(0.4, 0.5), (0.9, 0)]],
    "A": [[(0, 0), (0.45, 1), (0.9, 0)], [(0.2, 0.42), (0.7, 0.42)]],
    "M": [[(0, 0), (0, 1), (0.45, 0.45), (0.9, 1), (0.9, 0)]],
    "E": [[(0.9, 1), (0, 1), (0, 0), (0.9, 0)], [(0, 0.5), (0.65, 0.5)]],
    "N": [[(0, 0), (0, 1), (0.9, 0), (0.9, 1)]],
}

def neon_segments():
    A_, B_ = [], []
    for i, ch in enumerate("RAMEN"):
        x0 = -2.25 + i * 0.36
        for stroke in LETTERS[ch]:
            pts = [np.array([x0 + u * 0.27, 1.93 + v * 0.30, -2.93]) for u, v in stroke]
            for a, b in zip(pts[:-1], pts[1:]):
                A_.append(a); B_.append(b)
    return np.array(A_), np.array(B_)

NA, NB = neon_segments()

def power(t):
    lamp = 0.0 if t < 1.6 else 1.0
    if 1.6 <= t < 2.7:
        lamp = float(hash3(np.array([[math.floor(t * 16), 3.0, 1.0]]))[0] > 0.4) * (0.6 + 0.4 * (t - 1.6))
    neon = smooth(0.0, 0.1, t) * (0.25 + 0.75 * float(hash3(np.array([[math.floor(t * 9), 7.0, 2.0]]))[0] < 0.93))
    return lamp, neon

def box(p, lo, hi):
    q = np.maximum(lo - p, p - hi)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)

def segdist(p, a, b):
    pa = p[:, None, :] - a[None]
    ba = (b - a)[None]
    h = np.clip((pa * ba).sum(2) / ((ba * ba).sum(2) + 1e-9), 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=2).min(1)

def neon(p):
    bd = box(p, np.array([-2.35, 1.85, -3.0]), np.array([-0.45, 2.30, -2.85]))
    d = bd + 0.02
    m = bd < 0.05
    if m.any():
        d[m] = segdist(p[m], NA, NB) - 0.017
    return d

def grid(u, y, w, y0, y1, cw=0.8, ch=1.05, hw=0.2, hh=0.27):
    uc = np.mod(u, cw) - cw / 2
    vc = np.mod(y - y0, ch) - ch / 2
    q = np.stack([np.abs(uc) - hw, np.abs(vc) - hh, np.abs(w - 0.02) - 0.06], 1)
    d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return np.where((y > y0 - 0.2) & (y < y1), d, 9.0)

def buildings(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    facade = box(p, np.array([-7.0, -1, -8.0]), np.array([7.0, 3.4, -3.0]))
    facade = np.maximum(facade, -grid(x + 0.5, y, -(z + 3.0), 2.4, 3.1, 1.0, 1.0, 0.3, 0.3))
    facade = np.maximum(facade, -box(p, np.array([-2.4, 0.15, -3.12]), np.array([-0.3, 1.75, -2.9])))
    facade = np.maximum(facade, -box(p, np.array([0.25, 0.0, -3.15]), np.array([1.05, 1.95, -2.9])))
    d = facade
    for x0, x1, z0, z1, h in TOWERS:
        tw = box(p, np.array([x0, -1, z0]), np.array([x1, h, z1]))
        tw = np.maximum(tw, -grid(x - x0, y, z1 - z, 1.0, h - 0.6))
        d = np.minimum(d, tw)
    return d

def lamp_parts(p):
    metal = limb(p, np.array([0.8, 0.0, -0.5]), np.array([0.8, 0.35, -0.5]), 0.10, 0.065)
    metal = np.minimum(metal, limb(p, np.array([0.8, 0.35, -0.5]), np.array([0.8, 2.68, -0.5]), 0.05, 0.04))
    metal = np.minimum(metal, ellipsoid(p, np.array([0.8, 3.05, -0.5]), np.array([0.2, 0.05, 0.2])))
    metal = np.minimum(metal, np.linalg.norm(p - [0.8, 3.14, -0.5], axis=1) - 0.035)
    metal = np.minimum(metal, ellipsoid(p, np.array([0.8, 2.68, -0.5]), np.array([0.1, 0.03, 0.1])))
    glass = ellipsoid(p, LAMP, np.array([0.13, 0.17, 0.13]))
    hyd = limb(p, np.array([-0.4, 0.0, 0.3]), np.array([-0.4, 0.42, 0.3]), 0.095, 0.085)
    hyd = np.minimum(hyd, ellipsoid(p, np.array([-0.4, 0.45, 0.3]), np.array([0.09, 0.07, 0.09])))
    hyd = np.minimum(hyd, limb(p, np.array([-0.52, 0.28, 0.3]), np.array([-0.28, 0.28, 0.3]), 0.04, 0.04))
    hyd = np.minimum(hyd, limb(p, np.array([-0.4, 0.28, 0.3]), np.array([-0.4, 0.28, 0.43]), 0.045, 0.045))
    return [("metal", metal), ("glass", glass), ("hydrant", hyd)]

def scene(p, labels=False):
    parts = [("wall", buildings(p)), ("neon", neon(p))] + lamp_parts(p)
    ds = np.stack([d for _, d in parts])
    d = ds.min(0)
    return (d, [n for n, _ in parts], ds.argmin(0)) if labels else d

def sky(d, t):
    y = d[:, 1]
    el = np.clip(y, 0, 1)
    col = np.array([0.30, 0.13, 0.20]) * np.exp(-el * 5)[:, None] + np.array([0.02, 0.02, 0.05])
    uv = d[:, [0, 2]] / (np.abs(y)[:, None] + 0.3)
    cl = vnoise(np.stack([uv[:, 0] * 1.5 + t * 0.05, uv[:, 1] * 3, np.zeros(len(d))], 1))
    return col + np.clip(cl - 0.45, 0, 1)[:, None] * np.array([0.25, 0.12, 0.18]) * (1 - el[:, None] * 0.6)

def puddle(x, z):
    m = vnoise(np.stack([x * 1.3, z * 1.1, np.zeros_like(x)], 1))
    m = m + 0.6 * np.exp(-((x - 0.9) ** 2 / 0.8 + (z - 0.6) ** 2 / 0.9))
    return smooth(0.55, 0.62, m)

def ripples(x, z, t, drops):
    h = np.zeros_like(x)
    for (dx, dz, per, ph) in drops:
        tau = (t + ph) % per
        r = np.hypot(x - dx, z - dz)
        c = 0.32 * tau
        h = h + 0.004 * np.sin(55 * (r - c)) * np.exp(-((r - c) / 0.035) ** 2) * math.exp(-2.5 * tau)
    return h

def lights(t):
    lamp, ne = power(t)
    return [(LAMP, LAMPC * 2.6 * lamp, 2.8, True),
            (np.array([-1.35, 2.07, -2.75]), PINK * 1.1 * ne, 1.2, False),
            (np.array([-1.35, 0.9, -2.7]), WARM * 0.8, 1.3, False),
            (np.array([0.65, 1.0, -2.8]), WARM * 0.3, 0.8, False),
            (np.array([0.0, 7.0, -20.0]), np.array([0.6, 0.35, 0.5]) * 0.5, 9.0, False)]

def light_at(P, n, L):
    c = np.zeros((len(P), 3))
    for lp, lc, r, cone in L:
        v = lp - P
        dist = np.linalg.norm(v, axis=1) + 1e-6
        ndl = np.ones(len(P)) if n is None else np.clip((n * v).sum(1) / dist, 0, 1)
        k = 1.0
        if cone:
            k = 0.12 + 0.88 * smooth(0.5, 0.8, -v[:, 1] / dist)
        c = c + lc[None] * (k * ndl / (1 + (dist / r) ** 2))[:, None]
    return c

def shade_solid(P, n, D, t, L):
    d, names, lab = scene(P, labels=True)
    kind = np.array(names)[lab]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    lamp, ne = power(t)
    alb = np.full((len(P), 3), 0.06)
    emit = np.zeros((len(P), 3))
    wall = kind == "wall"
    fac = wall & (z > -8.1)
    row = np.floor(y / 0.085)
    brick = (np.mod(x * 4 + 0.5 * np.mod(row, 2), 1.0) > 0.07) & (np.mod(y, 0.085) > 0.014)
    alb[fac] = (np.array([0.40, 0.19, 0.14]) * np.where(brick, 1.0, 0.7)[:, None])[fac]
    alb[wall & ~fac] = np.array([0.10, 0.11, 0.14])
    deep = wall & (n[:, 2] > 0.7)
    shopw = deep & (x > -2.4) & (x < -0.3) & (y > 0.15) & (y < 1.75) & (z < -3.05)
    emit[shopw] = WARM * ((0.55 + 0.25 * vnoise(P[shopw] * 5)) * np.where(np.mod(y[shopw], 0.14) < 0.03, 0.45, 1.0))[:, None]
    door = deep & (x > 0.25) & (x < 1.05) & (y < 1.95) & (z < -3.05)
    emit[door] = WARM * 0.25
    u = np.where(fac, x + 0.5, x)
    up_ = deep & (y > 1.0) & ~shopw & ~door
    cell = np.stack([np.floor(u / np.where(fac, 1.0, 0.8)), np.floor((y - 1.0) / np.where(fac, 1.0, 1.05)), np.floor(z)], 1)
    hw = hash3(cell)
    inwin = up_ & ((z < -3.05) & fac | (~fac & (np.abs(np.mod(x, 1.0) - 0.5) < 0.21)))
    wc = np.where((hw > 0.5)[:, None], np.array([1.0, 0.7, 0.38]) * (0.5 + 0.6 * hash3(cell + 3))[:, None],
                  np.where((hw > 0.44)[:, None], np.array([0.4, 0.55, 1.0]) * (0.55 + 0.3 * math.sin(9 * t)), np.array([0.02, 0.025, 0.04])))
    emit[inwin] = wc[inwin]
    nm = kind == "neon"
    emit[nm] = PINK * (0.08 + 2.6 * ne)
    gl = kind == "glass"
    emit[gl] = LAMPC * (0.12 + 2.4 * lamp)
    alb[kind == "metal"] = np.array([0.06, 0.08, 0.07])
    alb[kind == "hydrant"] = np.array([0.55, 0.06, 0.05])
    c = alb * (light_at(P, n, L) * 1.5 + 0.012) + emit
    if D is not None:
        ndv = np.clip(-(n * D).sum(1), 0, 1)
        Rf = D + 2 * ndv[:, None] * n
        spec = np.zeros((len(P), 3))
        for lp, lc, r, cone in L:
            v = lp - P
            dist = np.linalg.norm(v, axis=1) + 1e-6
            spec = spec + lc[None] * (np.clip((Rf * v).sum(1) / dist, 0, 1) ** 30 / (1 + (dist / r) ** 2))[:, None]
        shiny = np.isin(kind, ["metal", "hydrant"])
        c = c + spec * np.where(shiny, 1.2, 0.25)[:, None] * (~(nm | gl))[:, None]
    return c

def march_scene(o, d, steps=110, tmax=60.0):
    T = np.full(len(o), 0.01); hit = np.zeros(len(o), bool); alive = np.ones(len(o), bool)
    for _ in range(steps):
        idx = np.flatnonzero(alive)
        if not len(idx): break
        dist = scene(o[idx] + d[idx] * T[idx, None])
        T[idx] += dist * 0.9
        h = dist < 0.0015 * (1 + T[idx])
        hit[idx[h]] = True
        alive[idx[h | (T[idx] > tmax)]] = False
    return hit, T

def camera(t):
    keys = [(0.0, (1.55, 0.14, 1.90), (0.75, 0.55, -1.0)),
            (4.5, (1.50, 0.55, 1.45), (0.80, 2.30, -0.5)),
            (8.5, (-0.10, 2.25, 0.95), (0.80, 2.65, -0.5)),
            (12.0, (-1.10, 1.40, 2.70), (0.20, 1.50, -2.0)),
            (16.0, (-0.30, 3.00, 6.00), (0.00, 3.20, -12.0)),
            (20.0, (0.00, 3.80, 7.60), (0.00, 4.60, -20.0))]
    ts = [k[0] for k in keys]
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(keys) - 2))
    s = (t - ts[i]) / (ts[i + 1] - ts[i])
    s = s * s * (3 - 2 * s)
    eye = np.array(keys[i][1]) * (1 - s) + np.array(keys[i + 1][1]) * s
    tgt = np.array(keys[i][2]) * (1 - s) + np.array(keys[i + 1][2]) * s
    f = normalize(tgt - eye)
    r = normalize(np.cross(f, [0, 1.0, 0]))
    return eye, f, r, np.cross(r, f), 477 / math.tan(math.radians(27))

class Renderer:
    def __init__(self):
        rng = np.random.default_rng(29)
        yy, xx = np.mgrid[0:ROWS, 0:COLS]
        self.iy, self.ix = np.nonzero((yy + 0.5) * CH < 955)
        self.u = OX + (self.ix + 0.5) * CW - W / 2
        self.v = CY - (self.iy + 0.5) * CH
        self.noise = rng.uniform(0, 1, (ROWS, COLS))
        self.drops = [(rng.uniform(-3, 3), rng.uniform(-2.9, 2.5), rng.uniform(0.5, 1.4), rng.uniform(0, 2)) for _ in range(70)]
        self.rain = np.stack([rng.uniform(-2.8, 3.2, 1100), rng.uniform(0, 6, 1100), rng.uniform(-2.9, 2.2, 1100)], 1)
        self.code = source_from("def neon_segments")

    def image(self, t):
        eye, f, r, up, F = camera(t)
        d = normalize(f[None] + (r[None] * self.u[:, None] + up[None] * self.v[:, None]) / F)
        o = np.tile(eye, (len(d), 1))
        L = lights(t)
        lamp, ne = power(t)
        col = sky(d, t)
        depth = np.full(len(d), np.inf)
        hit, T = march_scene(o, d)
        T[~hit] = np.inf
        tg = np.where(d[:, 1] < -1e-4, eye[1] / np.maximum(-d[:, 1], 1e-4), np.inf)
        ground = np.isfinite(tg) & (tg < T)
        solid = hit & ~ground
        haze = np.array([0.05, 0.03, 0.05])
        if solid.any():
            k = np.flatnonzero(solid)
            P = o[k] + d[k] * T[k, None]
            n = normals(scene, P, 0.002)
            c = shade_solid(P, n, d[k], t, L)
            fog = np.exp(-T[k] * 0.03)[:, None]
            col[k] = c * fog + haze * (1 - fog)
            depth[k] = T[k]
        if ground.any():
            k = np.flatnonzero(ground)
            P = o[k] + d[k] * tg[k, None]
            x, z = P[:, 0], P[:, 2]
            pm = puddle(x, z)
            e = 0.004
            hx = (ripples(x + e, z, t, self.drops) - ripples(x - e, z, t, self.drops)) / (2 * e)
            hz = (ripples(x, z + e, t, self.drops) - ripples(x, z - e, t, self.drops)) / (2 * e)
            gx = vnoise(np.stack([x * 30 + 0.1, z * 30, np.zeros_like(x)], 1)) - vnoise(np.stack([x * 30 - 0.1, z * 30, np.zeros_like(x)], 1))
            gz = vnoise(np.stack([x * 30, z * 30 + 0.1, np.ones_like(x)], 1)) - vnoise(np.stack([x * 30, z * 30 - 0.1, np.ones_like(x)], 1))
            rough = 1 - pm
            n = normalize(np.stack([-hx * pm - gx * rough * 0.6, np.ones_like(x), -hz * pm - gz * rough * 0.18], 1))
            D = d[k]
            ndv = np.clip(-(n * D).sum(1), 0, 1)
            Fr = (0.35 + 0.65 * (1 - ndv) ** 5) * rough + (0.8 + 0.2 * (1 - ndv) ** 5) * pm
            R = D + 2 * ndv[:, None] * n
            R[:, 1] = np.abs(R[:, 1])
            O = P + R * 0.02
            rh, rT = march_scene(O, R, steps=80, tmax=40.0)
            refl = sky(R, t)
            if rh.any():
                Q = O[rh] + R[rh] * rT[rh, None]
                nq = normals(scene, Q, 0.003)
                rc = shade_solid(Q, nq, None, t, L)
                fog = np.exp(-rT[rh] * 0.03)[:, None]
                refl[rh] = rc * fog + haze * (1 - fog)
            tile = (np.mod(x * 2.0, 1.0) < 0.03) | (np.mod(z * 2.0, 1.0) < 0.03)
            base = np.array([0.06, 0.06, 0.065]) * (0.7 + 0.5 * vnoise(np.stack([x * 8, z * 8, np.zeros_like(x)], 1)))[:, None] * np.where(tile, 0.5, 1.0)[:, None]
            c = base * (light_at(P, np.tile([0, 1.0, 0], (len(P), 1)), L) * 1.6 + 0.012) * (1 - Fr[:, None] * 0.6) + Fr[:, None] * refl
            fog = np.exp(-tg[k] * 0.03)[:, None]
            col[k] = c * fog + haze * (1 - fog)
            depth[k] = tg[k]

        if lamp > 0:
            tmax = np.minimum(np.where(np.isfinite(depth), depth, 12.0), 12.0)
            acc = np.zeros(len(d))
            for s in np.linspace(0.02, 1.0, 26):
                Pm = o + d * (tmax * s)[:, None]
                v = Pm - LAMP
                dist = np.linalg.norm(v, axis=1) + 1e-6
                cone = smooth(0.72, 0.9, -v[:, 1] / dist)
                acc += cone / (1 + (dist / 1.1) ** 2) * tmax / 26
            col = col + LAMPC[None] * (acc * 0.32 * lamp)[:, None]

        Lb = np.zeros((ROWS, COLS, 3)); Lb[self.iy, self.ix] = col
        Z = np.full((ROWS, COLS), np.inf); Z[self.iy, self.ix] = depth

        rp = self.rain.copy()
        rp[:, 1] = np.mod(rp[:, 1] - t * 8.5, 6.0)
        rp[:, 0] = rp[:, 0] + 0.06 * rp[:, 1]
        vv = LAMP - rp
        dd = np.linalg.norm(vv, axis=1) + 1e-6
        lc = LAMPC[None] * (lamp * 3.0 * smooth(0.72, 0.9, vv[:, 1] / dd) / (1 + (dd / 1.4) ** 2))[:, None] + light_at(rp, None, L[1:4]) * 1.2
        RAIN = np.zeros((ROWS, COLS)); RC = np.zeros((ROWS, COLS, 3))
        for s in np.linspace(0, 0.25, 8):
            q = rp + np.array([-0.06 * s, s, 0]) - eye
            z = q @ f
            ok = z > 0.3
            px, py = W / 2 + F * (q[ok] @ r) / z[ok], CY - F * (q[ok] @ up) / z[ok]
            cx, cy = cells(px, py)
            okk = (cx >= 0) & (cx < COLS) & (cy >= 0) & (cy < ROWS) & (py < 955)
            dist = np.linalg.norm(q[ok], axis=1)
            okk[okk] &= dist[okk] < Z[cy[okk], cx[okk]]
            np.maximum.at(RAIN, (cy[okk], cx[okk]), (lc[ok].max(1) * (1 - s * 2))[okk])
            RC[cy[okk], cx[okk]] = lc[ok][okk]

        lum = Lb.max(2)
        lvl = np.clip((np.clip(lum, 0, 1.4) ** 0.7 * (len(RAMP) - 1) + (self.noise - 0.5) * 2).astype(int), 0, len(RAMP) - 1)
        solidc = np.isfinite(Z)
        chars = RAMP_IDX[lvl]
        faint = lum < 0.05
        chars = np.where(faint, np.where(self.noise > 0.65, IDX["."], 0), chars)
        cols = (Lb * 1.5).astype(np.float32)
        cols[faint] = np.maximum(cols[faint], 0.03)
        rain_on = (RAIN > 0.22) & (RAIN > lum * 0.7)
        chars = np.where(rain_on, np.where(self.noise > 0.5, IDX["/"], IDX["|"]), chars)
        cols[rain_on] = (RC[rain_on] / (RC[rain_on].max(1, keepdims=True) + 1e-6) * np.clip(RAIN[rain_on], 0.15, 0.95)[:, None] * 1.6).astype(np.float32)
        chars[int(955 / CH):] = 0
        cols *= smooth(0, 1.0, t)
        im = Image.fromarray((glow(to_pixels(chars, cols), 0.85, 5) * 255).astype(np.uint8))
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
