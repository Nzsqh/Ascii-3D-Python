"""
Nzsqh.py — a firefly in a meadow at night, in 3D ASCII.
It takes off from a blade of grass, paints a butterfly of light in the air, and the butterfly flies away.

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
DURATION, TITLE = 19.0, "firefly"
T_OFF, T_DRAW, T_DONE, T_ALIVE, T_HOME = 4.8, 6.3, 12.4, 12.9, 15.6
S = 0.1
FPOS = np.array([0.03, 0.611, 0.0])
CELL, FIELD = 0.085, 4.0
GLOWC = np.array([0.72, 1.0, 0.30])
MOON = normalize(np.array([-0.55, 0.42, -0.75]))
MOONC = np.array([0.55, 0.65, 0.95])
WING_C = np.array([0.0, 1.02, -0.35])
WING_S = 0.5
CY = 478.0

def butterfly():
    a = np.linspace(0, 2 * np.pi, 1800)
    m = np.abs(np.angle(np.exp(1j * a)))
    r = 1.05 * np.exp(-((m - 0.95) / 0.5) ** 2) + 0.68 * np.exp(-((m - 2.3) / 0.42) ** 2) + 0.05
    r = r * (1 + 0.05 * np.sin(a * 16))
    outer = np.stack([r * np.sin(a), r * np.cos(a)], 1)
    inner = outer[::-1] * 0.58
    body = np.concatenate([np.linspace([0, 0.05], [0, -0.42], 120), np.linspace([0, -0.42], [0, 0.30], 160)])
    k = np.linspace(0, 1, 160)[:, None]
    ant = np.concatenate([[0.0, 0.30] + k * [0.22, 0.42] + np.sin(k * np.pi) * [0.05, 0.0],
                          ([0.0, 0.30] + k * [0.22, 0.42])[::-1],
                          [0.0, 0.30] + k * [-0.22, 0.42] + np.sin(k * np.pi) * [-0.05, 0.0]])
    pts = np.concatenate([outer, inner, body, ant])
    return np.concatenate([pts, np.zeros((len(pts), 1))], 1) * WING_S

def bezier(a, b, c, d, s):
    s = np.clip(s, 0, 1)
    return ((1 - s) ** 3) * a + 3 * (1 - s) ** 2 * s * b + 3 * (1 - s) * s * s * c + s ** 3 * d

def firefly_path(t, curve):
    start, end = curve[0] + WING_C, curve[-1] + WING_C
    if t < T_OFF:
        return FPOS + [0, 0.002 * math.sin(3 * t), 0]
    if t < T_DRAW:
        s = smooth(T_OFF, T_DRAW, t)
        return bezier(FPOS, FPOS + [0.30, 0.18, 0.05], start + [0.3, -0.1, 0.25], start, s)
    if t < T_DONE:
        i = (t - T_DRAW) / (T_DONE - T_DRAW) * (len(curve) - 1)
        k = int(i); f = i - k
        return (curve[k] * (1 - f) + curve[min(k + 1, len(curve) - 1)] * f) + WING_C
    s = smooth(T_DONE, T_HOME, t)
    return bezier(end, end + [0.5, 0.25, 0.6], FPOS + [-0.35, 0.22, 0.25], FPOS, s)

def heading(t, curve):
    if t < T_OFF or t > T_HOME + 0.05:
        return np.array([1.0, 0, 0])
    v = firefly_path(t + 0.03, curve) - firefly_path(t - 0.03, curve)
    v = v / (np.linalg.norm(v) + 1e-9)
    v[1] *= 0.25
    return normalize(v * 0.8 + np.array([1.0, 0, 0]) * 0.2 * smooth(T_HOME - 0.6, T_HOME, t))

def lantern(t):
    if T_OFF < t < T_HOME:
        return 1.0
    ph = (t % 2.1) / 2.1
    return 0.3 + 0.7 * math.exp(-((ph - 0.35) / 0.14) ** 2)

def rot_axis(ax, a):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K

def insect(t, flying):
    tw = 0.06 * math.sin(5 * t) * (1 - flying)
    e = flying
    P = []
    P.append(("pronotum", lambda q: ellipsoid(q, np.array([0.30, 0.05, 0]), np.array([0.17, 0.065, 0.165]))))
    P.append(("head", lambda q: ellipsoid(q, np.array([0.40, -0.03, 0]), np.array([0.08, 0.07, 0.085]))))
    for z in (0.065, -0.065):
        P.append(("eye", lambda q, c=np.array([0.43, -0.03, z]): np.linalg.norm(q - c, axis=1) - 0.045))
        a0 = np.array([0.46, -0.01, z * 0.6])
        a1 = np.array([0.64, 0.10 + tw, z * 2.0])
        a2 = np.array([0.86, 0.16 + 2 * tw, z * 3.6 + 0.4 * tw])
        P.append(("leg", lambda q, a=a0, b=a1: limb(q, a, b, 0.026, 0.02)))
        P.append(("leg", lambda q, a=a1, b=a2: limb(q, a, b, 0.02, 0.014)))
    P.append(("belly", lambda q: ellipsoid(q, np.array([-0.12, -0.01, 0]), np.array([0.40, 0.065, 0.125]))))
    P.append(("lantern", lambda q: ellipsoid(q, np.array([-0.40, -0.035, 0]), np.array([0.17, 0.05, 0.10]))))
    for side in (1, -1):
        hinge = np.array([0.18, 0.08, 0.02 * side])
        Rm = rot_axis(np.array([1.0, 0, 0]), -side * 0.95 * e) @ rot_axis(np.array([0, 0, 1.0]), 0.35 * e)
        P.append(("elytron", lambda q, h=hinge, R=Rm, s=side: ellipsoid((q - h) @ R + h, np.array([-0.13, 0.065, 0.072 * s]), np.array([0.34, 0.04, 0.08]))))
        for k, hx in enumerate((0.27, 0.17, 0.06)):
            hip = np.array([hx, -0.04, 0.05 * side])
            knee = np.array([hx + 0.04 * (k - 1), -0.07, 0.20 * side]) * (1 - e) + (hip + [-0.05, -0.04, 0.06 * side]) * e
            foot = np.array([hx + 0.12 * (k - 1), -0.19, 0.22 * side]) * (1 - e) + (hip + [-0.15, -0.07, 0.05 * side]) * e
            P.append(("leg", lambda q, a=hip, b=knee: limb(q, a, b, 0.03, 0.022)))
            P.append(("leg", lambda q, a=knee, b=foot: limb(q, a, b, 0.022, 0.015)))
    return P

def main_blade(p):
    pts = [np.array(v) for v in ((-0.22, 0.0, 0.05), (-0.15, 0.33, 0.03), (-0.07, 0.55, 0.01), (0.06, 0.588, 0.0), (0.20, 0.566, -0.01))]
    rads = (0.012, 0.010, 0.0085, 0.0075, 0.006, 0.0015)
    d = np.full(len(p), 9.0)
    for i in range(4):
        d = np.minimum(d, limb(p, pts[i], pts[i + 1], rads[i], rads[i + 1]))
    return d

def grass(p, t):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    gx, gz = np.floor(x / CELL), np.floor(z / CELL)
    hs = lambda k: hash3(np.stack([gx, gz, np.full_like(gx, k)], 1))
    ox = (gx + 0.5 + (hs(1) - 0.5) * 0.45) * CELL
    oz = (gz + 0.5 + (hs(2) - 0.5) * 0.45) * CELL
    h = 0.18 + 0.33 * hs(3) ** 1.4
    near = np.hypot(ox - 0.03, oz) < 0.2
    h = np.where(near, 0.12, h)
    ang = hs(4) * 2 * np.pi
    lean = 0.03 * hs(5) + 0.012 * np.sin(1.3 * t + ox * 2.1 + hs(6) * 6)
    s = np.clip(y / h, 0, 1)
    cx = ox + np.cos(ang) * lean * s * s
    cz = oz + np.sin(ang) * lean * s * s
    rad = 0.0075 * (1 - s) ** 0.8 + 0.0007
    dxz = np.hypot(x - cx, z - cz)
    d = np.where(y > h, np.hypot(dxz, y - h) - 0.0007, dxz - rad) * 0.85
    return d, s, hs(7), gx, gz

def grass_step(o, d, T, t):
    p = o + d * T[:, None]
    db, _, _, gx, gz = grass(p, t)
    tx = np.where(d[:, 0] > 0, (gx + 1) * CELL - p[:, 0], gx * CELL - p[:, 0]) / np.where(np.abs(d[:, 0]) < 1e-6, 1e-6, d[:, 0])
    tz = np.where(d[:, 2] > 0, (gz + 1) * CELL - p[:, 2], gz * CELL - p[:, 2]) / np.where(np.abs(d[:, 2]) < 1e-6, 1e-6, d[:, 2])
    step = np.minimum(db, np.minimum(np.abs(tx), np.abs(tz)) + 0.002)
    step = np.minimum(step, main_blade(p))
    step = np.minimum(step, p[:, 1])
    far = np.hypot(p[:, 0], p[:, 2]) > FIELD
    step = np.where(far, p[:, 1] - 0.22, step)
    step = np.where(p[:, 1] > 0.6, np.maximum(p[:, 1] - 0.59, 0.002), step)
    return step

def sky(d, t):
    y = d[:, 1]
    el = np.clip(y, 0, 1)
    col = np.array([0.10, 0.11, 0.20]) * (1 - el[:, None] ** 0.35) + np.array([0.008, 0.012, 0.04]) * el[:, None] ** 0.35
    band = np.exp(-((d @ normalize(np.array([0.6, 0.2, 0.75]))) / 0.22) ** 2) * vnoise(d * 7) * np.clip(y * 3, 0, 1)
    col = col + band[:, None] * np.array([0.08, 0.07, 0.13])
    st = hash3(np.floor(d * 180))
    star = (st > 0.989) & (y > 0.04)
    col = col + star[:, None] * (0.35 + 0.5 * np.sin(3 * t + st * 300) ** 2)[:, None] * np.array([0.9, 0.92, 1.0])
    cm = d @ MOON
    col = col + MOONC * (0.15 * np.exp((cm - 1) * 40))[:, None]
    disk = cm > math.cos(0.04)
    crescent = disk & ((d @ normalize(MOON + np.array([0.02, 0.025, 0.0]))) < math.cos(0.032))
    col[crescent] = np.array([1.0, 0.95, 0.82]) * 1.2
    az = np.arctan2(d[:, 0], -d[:, 2])
    tree = 0.02 + 0.06 * vnoise(np.stack([az * 6, np.zeros_like(az), np.zeros_like(az)], 1)) ** 2 + 0.02 * vnoise(np.stack([az * 50, np.ones_like(az), np.zeros_like(az)], 1))
    col[y < tree] = np.array([0.012, 0.016, 0.025])
    sky.star = star & (y >= tree)
    return col

def camera(t):
    keys = [(0.0, (0.175, 0.712, 0.085), (0.025, 0.607, 0.0)),
            (4.6, (0.11, 0.712, 0.155), (0.025, 0.609, 0.0)),
            (6.3, (0.55, 0.78, 2.30), (0.0, 0.95, -0.30)),
            (12.4, (-0.35, 0.80, 2.45), (0.0, 0.98, -0.30)),
            (15.2, (-0.25, 0.95, 2.70), (0.10, 1.45, -1.2)),
            (17.2, (0.10, 0.725, 0.20), (0.025, 0.609, 0.0)),
            (19.0, (0.17, 0.713, 0.09), (0.025, 0.607, 0.0))]
    ts = [k[0] for k in keys]
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(keys) - 2))
    s = (t - ts[i]) / (ts[i + 1] - ts[i])
    s = s * s * (3 - 2 * s)
    eye = np.array(keys[i][1]) * (1 - s) + np.array(keys[i + 1][1]) * s
    tgt = np.array(keys[i][2]) * (1 - s) + np.array(keys[i + 1][2]) * s
    f = normalize(tgt - eye)
    r = normalize(np.cross(f, [0, 1.0, 0]))
    return eye, f, r, np.cross(r, f), 477 / math.tan(math.radians(24))

class Renderer:
    def __init__(self):
        rng = np.random.default_rng(5)
        yy, xx = np.mgrid[0:ROWS, 0:COLS]
        self.iy, self.ix = np.nonzero((yy + 0.5) * CH < 955)
        self.u = OX + (self.ix + 0.5) * CW - W / 2
        self.v = CY - (self.iy + 0.5) * CH
        self.noise = rng.uniform(0, 1, (ROWS, COLS))
        self.curve = butterfly()
        self.bokeh = np.stack([rng.uniform(-3.5, 3.5, 70), rng.uniform(0.15, 1.6, 70), rng.uniform(-5, 0.6, 70)], 1)
        self.bphase = rng.uniform(0, 10, 70)
        self.spark = rng.uniform(0, 1, (400, 3))
        self.code = source_from("def butterfly")

    def wings_now(self, t):
        k = int(np.clip((t - T_DRAW) / (T_DONE - T_DRAW), 0, 1) * (len(self.curve) - 1))
        pts = self.curve[:k + 1].copy()
        if t < T_ALIVE or not len(pts):
            return pts + WING_C, np.linspace(0.5, 1, len(pts))
        a = t - T_ALIVE
        flap = 1.05 * math.sin(2 * math.pi * 1.3 * a) * smooth(0, 0.8, a) - 0.25
        x = pts[:, 0]
        pts[:, 2] = np.abs(x) * math.sin(flap)
        pts[:, 0] = x * math.cos(flap)
        lift = smooth(0.3, 3.6, a) ** 1.4
        pts = pts * (1 - 0.7 * lift) @ rot_y(0.5 * lift).T
        return pts + WING_C + np.array([0.4, 1.8, -3.0]) * lift, np.ones(len(pts)) * (1 - smooth(2.8, 3.8, a))

    def image(self, t):
        eye, f, r, up, F = camera(t)
        d = normalize(f[None] + (r[None] * self.u[:, None] + up[None] * self.v[:, None]) / F)
        o = np.tile(eye, (len(d), 1))
        n_ = len(d)
        col = sky(d, t)
        STAR = np.zeros((ROWS, COLS), bool); STAR[self.iy, self.ix] = sky.star
        BUG = np.zeros((ROWS, COLS), bool)
        depth = np.full(n_, np.inf)
        flying = smooth(T_OFF - 0.3, T_OFF + 0.2, t) * (1 - smooth(T_HOME - 0.4, T_HOME, t))
        pos = firefly_path(t, self.curve)
        fw = heading(t, self.curve)
        fw = normalize(fw + np.array([0, 0.2, 0]) * flying)
        sd = normalize(np.cross(fw, [0, 1.0, 0]))
        uv = np.cross(sd, fw)
        Mw = np.stack([fw, uv, sd], 1)
        lamp_w = pos + Mw @ np.array([-0.42, -0.04, 0]) * S
        I = lantern(t)
        wpts, wlum = self.wings_now(t)
        lights = [(lamp_w, GLOWC * I * 1.0, 0.07)]
        if len(wpts) > 8:
            for p_ in wpts[:: max(1, len(wpts) // 8)][:8]:
                lights.append((p_, GLOWC * 0.25 * float(wlum.mean()), 0.25))

        T = np.zeros(n_)
        alive = d[:, 1] < 0.25
        hit = np.zeros(n_, bool)
        start = np.where(eye[1] > 0.6, (eye[1] - 0.6) / np.maximum(-d[:, 1], 1e-4), 0)
        T[alive] = start[alive]
        alive &= T < 40
        for _ in range(170):
            idx = np.flatnonzero(alive)
            if not len(idx): break
            st = grass_step(o[idx], d[idx], T[idx], t)
            T[idx] += st
            h = st < 0.0008 * (1 + 2 * T[idx])
            hit[idx[h]] = True
            alive[idx[h | (T[idx] > 40)]] = False
        if hit.any():
            k = np.flatnonzero(hit)
            P = o[k] + d[k] * T[k, None]
            db, s, hv, _, _ = grass(P, t)
            mb = main_blade(P)
            isg = P[:, 1] < 0.004
            isfar = np.hypot(P[:, 0], P[:, 2]) > FIELD
            ismain = (mb < db + 0.001) & ~isg
            e = 0.002
            gsdf = lambda q: np.minimum(grass(q, t)[0], main_blade(q))
            n = normals(gsdf, P, e)
            n[isg | isfar] = [0, 1.0, 0]
            n = n * np.sign(-(n * d[k]).sum(1) + 1e-6)[:, None]
            s = np.where(ismain, np.clip(P[:, 1] / 0.6, 0, 1), s)
            alb = np.array([0.05, 0.11, 0.04])[None] * (1 - s[:, None]) + np.array([0.20, 0.36, 0.11])[None] * s[:, None]
            alb = alb * (0.7 + 0.6 * hv[:, None]) + np.array([0.06, 0.02, 0.0]) * (hv[:, None] > 0.8)
            alb[isg] = np.array([0.02, 0.035, 0.015])
            ao = 0.25 + 0.75 * s ** 0.8
            ao[ismain] = 0.9
            ambient = np.array([0.13, 0.17, 0.30]) * ao[:, None]
            moon = MOONC * 0.8 * (np.abs(n @ MOON) * 0.6 + 0.4)[:, None] * ao[:, None]
            ndv = np.clip(-(n * d[k]).sum(1), 0, 1)
            moon = moon + MOONC * 0.5 * ((1 - ndv) ** 3 * ao)[:, None]
            gl = np.zeros((len(k), 3))
            for lp, lc, rr in lights:
                L = lp - P
                dist = np.linalg.norm(L, axis=1)
                wrap = np.abs((n * L).sum(1) / (dist + 1e-9)) * 0.6 + 0.4
                gl = gl + lc[None] * (wrap / (1 + (dist / rr) ** 2))[:, None]
            c = alb * (ambient + moon + gl * 3.2) + gl * 0.06
            fog = np.exp(-T[k] * 0.22)[:, None]
            focus = np.linalg.norm(eye - FPOS)
            macro = 1 - smooth(0.35, 0.7, focus)
            c = c * (1 - 0.6 * macro * np.clip((T[k] - focus - 0.06) / 0.25, 0, 1))[:, None]
            c = c * fog + np.array([0.05, 0.06, 0.11]) * (1 - fog)
            c[isfar] = (np.array([0.05, 0.08, 0.06]) * (0.6 + 0.8 * hash3(np.floor(P[isfar] * 60))[:, None])) * fog[isfar] + np.array([0.05, 0.06, 0.11]) * (1 - fog[isfar])
            col[k] = c
            depth[k] = T[k]

        ch = np.flatnonzero(np.linalg.norm(np.cross(pos - o, d), axis=1) < 0.12)
        if len(ch):
            parts = insect(t, flying)
            def fsdf(p):
                q = ((p - pos) @ Mw) / S
                return blend(parts, q, 0.02) * S
            hf, Tf = march(fsdf, o[ch], d[ch], steps=70, eps=0.0004, tmax=12)
            m = hf & (Tf < depth[ch])
            if m.any():
                k = ch[m]
                P = o[k] + d[k] * Tf[m, None]
                n = normals(fsdf, P, 0.0006)
                q = ((P - pos) @ Mw) / S
                _, lab = blend(parts, q, 0.02, labels=True)
                kind = np.array([parts[i][0] for i in lab])
                alb = np.full((len(k), 3), 0.16)
                pr = kind == "pronotum"
                rr = np.hypot(q[:, 0] - 0.30, q[:, 2])
                pc = np.where((rr < 0.05)[:, None], np.array([0.05, 0.03, 0.03]),
                              np.where((rr < 0.11)[:, None], np.array([0.85, 0.30, 0.28]), np.array([0.95, 0.78, 0.30])))
                alb[pr] = pc[pr]
                ely = kind == "elytron"
                edge = np.abs(q[:, 2]) > 0.125
                alb[ely] = np.where(edge[ely, None], np.array([0.9, 0.78, 0.38]), np.array([0.34, 0.22, 0.13]) * (0.75 + 0.5 * vnoise(q[ely] * 30))[:, None])
                alb[kind == "belly"] = np.array([0.45, 0.33, 0.18])
                lam = kind == "lantern"
                alb[lam] = np.array([0.55, 0.58, 0.32])
                ndv = np.clip(-(n * d[k]).sum(1), 0, 1)
                Rf = d[k] + 2 * ndv[:, None] * n
                fill = normalize(-f + up * 0.6 + r * 0.4)
                lit = np.array([0.12, 0.14, 0.24]) + MOONC * 0.8 * np.clip(n @ MOON, 0, 1)[:, None] + np.array([0.88, 0.88, 0.95]) * 1.3 * np.clip(n @ fill, 0, 1)[:, None]
                Ld = lamp_w - P
                dl = np.linalg.norm(Ld, axis=1)
                lit = lit + GLOWC * (I * 0.5 * np.clip((n * Ld).sum(1) / (dl + 1e-9) * 0.5 + 0.5, 0, 1) / (1 + (dl / 0.03) ** 2))[:, None]
                spec = (np.clip(Rf @ MOON, 0, 1) ** 40 * 0.9 + np.clip(Rf @ fill, 0, 1) ** 30 * 0.6 + (1 - ndv) ** 4 * 0.5)[:, None] * MOONC
                gloss = np.isin(kind, ["eye", "elytron", "head"])
                c = alb * lit + spec * np.where(gloss, 1.0, 0.35)[:, None]
                c[lam] = c[lam] + GLOWC * (I * 2.6 * (0.6 + 0.4 * ndv[lam]))[:, None]
                col[k] = c
                depth[k] = Tf[m]
                BUG[self.iy[k], self.ix[k]] = True

        L = np.zeros((ROWS, COLS, 3)); L[self.iy, self.ix] = col
        Z = np.full((ROWS, COLS), np.inf); Z[self.iy, self.ix] = depth

        def splat(P, c, rad_world=0.0):
            c = np.broadcast_to(c, (len(P), 3))
            q = P - eye
            z = q @ f
            ok = z > 0.03
            if not ok.any(): return
            q, z, c = q[ok], z[ok], c[ok]
            px, py = W / 2 + F * (q @ r) / z, CY - F * (q @ up) / z
            rad = np.maximum(rad_world * F / z / CW, 0.6)
            span = int(min(6, np.ceil(rad.max() * 2)))
            cx0, cy0 = (px - OX) / CW, py / CH
            for dy in range(-span, span + 1):
                for dx in range(-span, span + 1):
                    cx, cy = np.floor(cx0).astype(int) + dx, np.floor(cy0).astype(int) + dy
                    okk = (cx >= 0) & (cx < COLS) & (cy >= 0) & (cy < ROWS) & (cy * CH < 955)
                    w_ = np.exp(-((cx + 0.5 - cx0) ** 2 + ((cy + 0.5 - cy0) * CH / CW) ** 2) / (2 * rad ** 2))
                    okk &= w_ > 0.02
                    okk[okk] &= z[okk] < Z[cy[okk], cx[okk]] + 0.01
                    np.add.at(L, (cy[okk], cx[okk]), c[okk] * w_[okk, None])

        blink = np.clip(np.sin(t * 1.7 + self.bphase) * 3 - 2, 0, 1)
        splat(self.bokeh, GLOWC[None] * 0.9 * blink[:, None], 0.012)
        if len(wpts):
            age = np.linspace(0.55, 1.0, len(wpts)) if t < T_ALIVE else np.ones(len(wpts))
            splat(wpts, GLOWC[None] * (0.55 * age * wlum)[:, None], 0.0)
            splat(wpts[::6], np.array([0.85, 1.0, 0.55])[None] * (0.25 * wlum[::6])[:, None], 0.02)
            if t > T_ALIVE:
                a = t - T_ALIVE
                pick = (self.spark[:, 0] * (len(wpts) - 1)).astype(int)
                fall = ((a * 0.6 + self.spark[:, 1]) % 1.0)
                sp = wpts[pick] + np.stack([np.zeros(400), -fall * 0.5, (self.spark[:, 2] - 0.5) * 0.1], 1)
                splat(sp, GLOWC[None] * (0.8 * (1 - fall) * smooth(0, 1, a) * (1 - smooth(2.8, 3.8, a)))[:, None], 0.0)
                if a < 3.8:
                    body = WING_C + np.stack([np.zeros(60), np.linspace(-0.42, 0.30, 60) * WING_S, np.zeros(60)], 1)
                    lift = smooth(0.3, 3.6, a) ** 1.4
                    body = (body - WING_C) * (1 - 0.7 * lift) + WING_C + np.array([0.4, 1.8, -3.0]) * lift
                    splat(body, np.array([0.95, 1.0, 0.7])[None] * 0.9 * smooth(0, 0.6, a) * (1 - smooth(2.8, 3.8, a)), 0.004)
        splat(lamp_w[None], GLOWC[None] * 1.1 * I, 0.015 + 0.02 * I)
        if flying > 0.05:
            phase = np.linspace(-1, 1, 90)
            rad_ = np.linspace(0.1, 0.55, 12)
            A, Rr = np.meshgrid(phase, rad_)
            for side in (1, -1):
                loc = np.stack([0.12 - Rr.ravel() * 0.35, 0.08 + Rr.ravel() * np.sin(A.ravel() * 1.1) * 0.9, side * (0.05 + Rr.ravel() * np.cos(A.ravel() * 1.1))], 1)
                wp = pos + (loc * S) @ Mw.T
                splat(wp, np.array([0.55, 0.58, 0.7])[None] * 0.06 * flying * np.ones((len(wp), 1)), 0.0)

        lum = L.max(2)
        lvl = np.clip((np.clip(lum, 0, 1.4) ** 0.6 * (len(RAMP) - 1) + (self.noise - 0.5) * 6).astype(int), 0, len(RAMP) - 1)
        solid = np.isfinite(Z)
        lvl = np.where(solid, np.maximum(lvl, 4), lvl)
        lvl = np.where(BUG, np.maximum(lvl, 16), lvl)
        chars = RAMP_IDX[lvl]
        faint = (~solid) & (lum < 0.07)
        chars = np.where(faint, np.where(self.noise > 0.6, IDX["."], 0), chars)
        star = STAR & ~solid
        chars = np.where(star, np.where(lum > 0.45, IDX["+"], IDX["·"]), chars)
        cols = (L * 1.6).astype(np.float32)
        cols[faint] = np.maximum(cols[faint], 0.04)
        chars[int(955 / CH):] = 0
        cols *= smooth(0, 1.2, t)
        im = Image.fromarray((glow(to_pixels(chars, cols), 0.9, 5) * 255).astype(np.uint8))
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
