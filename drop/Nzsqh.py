"""
Nzsqh.py — a single drop of water, in 3D ASCII.
It swells at the tip of a leaf, falls in slow motion and splashes into a pond under the moon.

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
DURATION, TITLE = 18.0, "drop"
T_REL, T_IMP = 5.2, 7.0
G = 0.78
TIP = np.array([0.0, 1.42, 0.0])
MOON = normalize(np.array([0.08, 0.20, -1.0]))
FILL = normalize(np.array([0.5, 0.7, 0.8]))
MOONC = np.array([1.0, 0.93, 0.80])
LEAF_U = normalize(np.array([-0.85, 0.40, -0.40]))
LEAF_W = normalize(np.cross(LEAF_U, [0, 1.0, 0]))
LEAF_V = np.cross(LEAF_W, LEAF_U)
LEAF_L = 1.7
PADS = [(-0.78, -0.62, 0.27, 0.6), (0.92, -0.95, 0.23, 2.3), (-0.62, 0.70, 0.21, 4.1), (1.25, 0.35, 0.18, 5.0)]
DEW = [(0.35, 0.05, 0.022), (0.62, -0.08, 0.018), (0.80, 0.07, 0.026), (1.05, -0.04, 0.016), (0.48, -0.03, 0.012)]
CY = 478.0

def water_height(x, z, t):
    r = np.hypot(x, z)
    h = 0.003 * np.sin(x * 9 + t * 1.3) * np.sin(z * 7 - t * 1.1) + 0.002 * np.sin((x + z) * 23 - t * 2)
    for t0, amp in ((T_IMP, 0.034), (T_IMP + 2.82, 0.016)):
        s = t - t0
        if s > 0:
            c, w = 0.42 * s, 0.09 + 0.16 * s
            h = h + amp * math.exp(-0.11 * s) * np.sin(17 * (r - c)) * np.exp(-((r - c) / w) ** 2) / (1 + 1.4 * r)
    s = t - T_IMP
    if 0 < s < 0.9:
        k = math.sin(math.pi * s / 0.9)
        h = h - 0.07 * k * np.exp(-(r / 0.13) ** 2)
    return h

def splash(t):
    s = t - T_IMP
    jet = 0.42 * math.sin(math.pi * np.clip((s - 0.45) / 1.4, 0, 1)) if s > 0.45 else 0.0
    crown = 0.13 * math.sin(math.pi * s / 0.8) if 0 < s < 0.8 else 0.0
    ring = 0.08 + 0.22 * math.sqrt(max(s, 0) / 0.8)
    return s, jet, crown, ring

def water_sdf(p, t):
    d = (p[:, 1] - water_height(p[:, 0], p[:, 2], t)) * 0.5
    s, jet, crown, ring = splash(t)
    extra = np.full(len(p), 9.0)
    if jet > 0.01:
        extra = np.minimum(extra, limb(p, np.array([0, -0.06, 0]), np.array([0, jet, 0]), 0.075, 0.036))
        extra = np.minimum(extra, np.linalg.norm(p - [0, jet + 0.01, 0], axis=1) - 0.048 * (s < 1.0))
    if crown > 0.005:
        r = np.hypot(p[:, 0], p[:, 2])
        q = np.stack([r - ring, p[:, 1] - np.clip(p[:, 1], 0, crown)], 1)
        extra = np.minimum(extra, np.linalg.norm(q, axis=1) - 0.011)
        for k in range(14):
            a = 2 * math.pi * k / 14 + 0.3 * math.sin(k * 7.1)
            c = np.array([ring * 1.05 * math.cos(a), crown + 0.025 + 0.02 * math.sin(k * 3.3), ring * 1.05 * math.sin(a)])
            extra = np.minimum(extra, np.linalg.norm(p - c, axis=1) - 0.017)
    return smin(d, extra, 0.05), extra

def drop_state(t):
    r = 0.035 + 0.072 * smooth(0, 4.6, t)
    if t < T_REL:
        wob = 0.05 * math.sin(9 * t) * smooth(3.5, 5.0, t)
        sag = 0.012 * smooth(4.0, T_REL, t)
        c = TIP + [0, -(r * 1.12) - 0.012 - sag, 0]
        return c, np.array([r * (1 - wob), r * (1.12 + wob + sag * 3), r * (1 - wob)])
    s = t - T_REL
    c = TIP + [0, -(r * 1.12) - 0.024 - 0.5 * G * s * s, 0]
    a = 0.14 * math.sin(15 * s) * math.exp(-1.6 * s)
    return c, np.array([r * (1 + a), r * (1 - a), r * (1 + a)])

def leaf_sdf(p):
    q = p - TIP
    u, w, v = q @ LEAF_U, q @ LEAF_W, q @ LEAF_V
    x = np.clip(u / LEAF_L, 0, 1)
    vc = 0.10 * np.sin(math.pi * x) + 0.10 * np.abs(w)
    half = 0.20 * np.sin(math.pi * x) ** 0.75
    plan = np.maximum(np.abs(w) - half, np.maximum(-u, u - LEAF_L))
    thick = np.abs(v - vc) - 0.006
    out = np.linalg.norm(np.maximum(np.stack([plan, thick], 1), 0), axis=1) + np.minimum(np.maximum(plan, thick), 0)
    return out * 0.7

def leaf_point(u, w, lift=0.0):
    x = u / LEAF_L
    vc = 0.10 * math.sin(math.pi * x) + 0.10 * abs(w)
    return TIP + LEAF_U * u + LEAF_W * w + LEAF_V * (vc + lift)

def pad_sdf(p, t, cx, cz, R, rot):
    y0 = float(water_height(np.array([cx]), np.array([cz]), t)[0]) + 0.004
    q = p - [cx, y0, cz]
    c, s = math.cos(rot), math.sin(rot)
    lx, lz = q[:, 0] * c + q[:, 2] * s, -q[:, 0] * s + q[:, 2] * c
    rr = np.hypot(lx, lz)
    disc = np.maximum(rr - R, np.abs(q[:, 1] - 0.012 * (rr / R) ** 4) - 0.006)
    notch = (np.abs(lz) - lx * 0.28) / 1.04
    return np.maximum(disc, -notch)

def object_parts(t):
    c, rad = drop_state(t)
    parts = [("leaf", leaf_sdf)]
    neck = 0.017 * (1 - smooth(T_REL - 0.5, T_REL, t))
    if neck > 0.002:
        parts.append(("drop", lambda p: smin(ellipsoid(p, c, rad), limb(p, TIP + [0, 0.005, 0], c, 0.012, neck), 0.03)))
    else:
        parts.append(("drop", lambda p: ellipsoid(p, c, rad)))
        r2 = 0.012 + 0.023 * smooth(T_REL + 1.5, DURATION, t)
        parts.append(("drop", lambda p: ellipsoid(p, TIP + [0, -r2 * 1.1 - 0.01, 0], np.array([r2, r2 * 1.1, r2]))))
    for u, w, r in DEW:
        parts.append(("drop", lambda p, c=leaf_point(u, w, r * 0.6), r=r: ellipsoid(p, c, np.array([r, r * 0.7, r]))))
    s = t - T_IMP - 1.0
    if 0 < s < 1.82:
        y = 0.43 + 0.5 * s - 0.5 * G * s * s
        a = 0.12 * math.sin(18 * s)
        parts.append(("drop", lambda p, c=np.array([0, y, 0]): ellipsoid(p, c, np.array([0.05 * (1 + a), 0.05 * (1 - a), 0.05 * (1 + a)]))))
    for cx, cz, R, rot in PADS:
        parts.append(("pad", lambda p, a=(cx, cz, R, rot): pad_sdf(p, t, *a)))
    return parts

def sky(d, t, rng_noise=None):
    y = d[:, 1]
    el = np.clip(y, 0, 1)
    col = np.array([0.30, 0.36, 0.52]) * (1 - el[:, None] ** 0.4) + np.array([0.06, 0.08, 0.17]) * el[:, None] ** 0.4
    uv = d[:, [0, 2]] / (np.abs(y)[:, None] + 0.25)
    cl = vnoise(np.stack([uv[:, 0] * 2 + t * 0.03, uv[:, 1] * 5, np.zeros(len(d))], 1)) * 0.65
    cl = cl + vnoise(np.stack([uv[:, 0] * 6, uv[:, 1] * 13, np.ones(len(d))], 1)) * 0.35
    cm = d @ MOON
    cloud = np.clip(cl - 0.48, 0, 1) * 2.2 * smooth(0.02, 0.2, np.abs(y))
    col = col + cloud[:, None] * (np.array([0.22, 0.24, 0.32]) + MOONC * 0.9 * np.exp((cm - 1) * 6)[:, None])
    star = hash3(np.floor(d * 160)) > 0.993
    col = col + (star & (y > 0.05))[:, None] * 0.55 * (1 - cloud[:, None])
    col = col + MOONC * (0.35 * np.exp((cm - 1) * 25) + 0.6 * np.exp((cm - 1) * 300))[:, None]
    disk = cm > math.cos(0.05)
    craters = 0.75 + 0.25 * vnoise(d * 90)
    col[disk] = MOONC * 1.4 * craters[disk, None]
    az = np.arctan2(d[:, 0], -d[:, 2])
    tree = 0.008 + 0.04 * vnoise(np.stack([az * 7, np.zeros_like(az), np.zeros_like(az)], 1)) ** 2 + 0.014 * vnoise(np.stack([az * 70, np.ones_like(az), np.zeros_like(az)], 1))
    col[y < tree] = np.array([0.03, 0.045, 0.075]) * (0.7 + 0.6 * hash3(np.floor(d * 300))[y < tree, None])
    sky.star = star & (y > 0.05) & (cloud < 0.3) & (y >= tree)
    return col

def world_behind(d, t):
    up = d.copy()
    up[:, 1] = np.abs(up[:, 1])
    return sky(up, t) * np.where(d[:, 1] < 0, 0.55, 1.0)[:, None]

def refract(d, n, eta):
    c = -(d * n).sum(1)
    k = 1 - eta * eta * (1 - c * c)
    out = eta * d + (eta * c - np.sqrt(np.clip(k, 0, None)))[:, None] * n
    return np.where((k < 0)[:, None], d + 2 * c[:, None] * n, out)

def water_drop_shade(P, n, D, t, gain=1.0):
    ndv = np.clip(-(n * D).sum(1), 0, 1)
    F = 0.03 + 0.97 * (1 - ndv) ** 5
    R = D + 2 * ndv[:, None] * n
    d1 = refract(D, n, 1 / 1.33)
    d2 = refract(d1, -n, 1.33)
    d2 = normalize(d2 * np.array([-1, 1, -1]) * 0.6 + d1 * 0.4)
    col = F[:, None] * world_behind(R, t) + (1 - F[:, None]) * world_behind(d2, t) * 0.95
    glint = np.clip(R @ MOON, 0, 1) ** 300 * 6 + np.clip(R @ FILL, 0, 1) ** 60 * 1.2
    return (col + MOONC * glint[:, None]) * gain

def camera(t):
    keys = [(0.0, (0.20, 1.60, 0.70), (0.0, 1.33, 0.0)),
            (4.9, (0.10, 1.50, 0.52), (0.0, 1.31, 0.0)),
            (6.0, (0.42, 0.95, 0.95), (0.0, 0.80, 0.0)),
            (7.0, (0.50, 0.30, 0.95), (0.0, 0.15, 0.0)),
            (11.0, (-0.40, 0.33, 1.05), (0.0, 0.14, 0.0)),
            (16.6, (0.15, 1.05, 2.45), (0.0, 0.45, -0.6)),
            (18.0, (0.17, 1.08, 2.55), (0.0, 0.46, -0.6))]
    ts = [k[0] for k in keys]
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(keys) - 2))
    s = (t - ts[i]) / (ts[i + 1] - ts[i])
    s = s * s * (3 - 2 * s)
    eye = np.array(keys[i][1]) * (1 - s) + np.array(keys[i + 1][1]) * s
    tgt = np.array(keys[i][2]) * (1 - s) + np.array(keys[i + 1][2]) * s
    follow = smooth(T_REL - 0.1, T_REL + 0.4, t) * (1 - smooth(T_IMP - 0.4, T_IMP + 0.3, t))
    c, _ = drop_state(t)
    tgt = tgt * (1 - follow) + (c - [0, 0.05, 0]) * follow
    f = normalize(tgt - eye)
    r = normalize(np.cross(f, [0, 1.0, 0]))
    return eye, f, r, np.cross(r, f), 477 / math.tan(math.radians(23))

def march_from(sdf, o, d, t0, steps=110, tmax=30.0):
    T = t0.copy(); hit = np.zeros(len(o), bool); alive = np.isfinite(T)
    T[~alive] = 0
    for _ in range(steps):
        idx = np.flatnonzero(alive)
        if not len(idx): break
        dist = sdf(o[idx] + d[idx] * T[idx, None])
        T[idx] += dist
        h = dist < 0.0012 * (1 + T[idx])
        hit[idx[h]] = True
        alive[idx[h | (T[idx] > tmax)]] = False
    return hit, T

class Renderer:
    def __init__(self):
        rng = np.random.default_rng(31)
        yy, xx = np.mgrid[0:ROWS, 0:COLS]
        self.keep = (yy + 0.5) * CH < 955
        self.iy, self.ix = np.nonzero(self.keep)
        self.u = OX + (self.ix + 0.5) * CW - W / 2
        self.v = CY - (self.iy + 0.5) * CH
        self.noise = rng.uniform(0, 1, (ROWS, COLS))
        self.spray = [(rng.uniform(0, 2 * np.pi, 46), rng.uniform(0.25, 0.65, 46), rng.uniform(0.45, 1.05, 46)),
                      (rng.uniform(0, 2 * np.pi, 16), rng.uniform(0.12, 0.3, 16), rng.uniform(0.3, 0.6, 16))]
        self.code = source_from("def water_height")

    def image(self, t):
        eye, f, r, up, F = camera(t)
        d = normalize(f[None] + (r[None] * self.u[:, None] + up[None] * self.v[:, None]) / F)
        o = np.tile(eye, (len(d), 1))
        col = sky(d, t)
        STAR = np.zeros((ROWS, COLS), bool); STAR[self.iy, self.ix] = sky.star
        depth = np.full(len(d), np.inf)
        s, jet, crown, ring = splash(t)
        ymax = 0.06 + max(jet + 0.06, crown + 0.06, 0.0)
        down = d[:, 1] < -1e-4
        t0 = np.where(eye[1] > ymax, np.where(down, (eye[1] - ymax) / np.maximum(-d[:, 1], 1e-4), np.inf), 0.0)
        wsdf = lambda p: water_sdf(p, t)[0]
        hw, Tw = march_from(wsdf, o, d, t0)
        parts = object_parts(t)
        osdf = lambda p: blend(parts, p, 1e-4)
        ho, To = march_from(osdf, o, d, np.zeros(len(d)), steps=90, tmax=12.0)
        flat = ~hw & down
        hw = hw | flat
        Tw[flat] = eye[1] / np.maximum(-d[flat, 1], 1e-4)
        Tw[~hw] = np.inf; To[~ho] = np.inf
        use_o = ho & (To < Tw)
        use_w = hw & ~use_o
        if use_w.any():
            k = np.flatnonzero(use_w)
            P, D = o[k] + d[k] * Tw[k, None], d[k]
            n = normals(wsdf, P, 0.004)
            n[flat[k]] = [0, 1.0, 0]
            far = np.clip((Tw[k] - 2.5) / 5, 0, 1)[:, None]
            n = normalize(n * (1 - far) + np.array([0, 1.0, 0]) * far)
            thin = water_sdf(P, t)[1] < 0.02
            ndv = np.clip(-(n * D).sum(1), 0, 1)
            Fr = 0.02 + 0.98 * (1 - ndv) ** 5
            R = D + 2 * ndv[:, None] * n
            R[:, 1] = np.abs(R[:, 1])
            deep = np.array([0.03, 0.07, 0.09])
            wc = Fr[:, None] * sky(R, t) * 1.15 + (1 - Fr[:, None]) * deep
            wc = wc + MOONC * (np.clip(R @ MOON, 0, 1) ** 900 * 5)[:, None]
            dist = np.hypot(P[:, 0], P[:, 2])
            wc = wc + np.array([0.5, 0.55, 0.7]) * (np.clip(R @ FILL, 0, 1) ** 40 * 0.5)[:, None]
            if thin.any():
                wc[thin] = water_drop_shade(P[thin], n[thin], D[thin], t, 1.1)
            col[k] = wc
            depth[k] = Tw[k]
        if use_o.any():
            k = np.flatnonzero(use_o)
            P, D = o[k] + d[k] * To[k, None], d[k]
            n = normals(osdf, P, 0.0015)
            _, lab = blend(parts, P, 1e-4, labels=True)
            kind = np.array([parts[i][0] for i in lab])
            oc = np.zeros((len(k), 3))
            m = kind == "drop"
            if m.any():
                oc[m] = water_drop_shade(P[m], n[m], D[m], t, 1.15)
            for name in ("leaf", "pad"):
                m = kind == name
                if not m.any(): continue
                nn = n[m] * np.sign(-(n[m] * D[m]).sum(1))[:, None]
                if name == "leaf":
                    q = P[m] - TIP
                    uu, ww = q @ LEAF_U, q @ LEAF_W
                    vein = np.exp(-(ww / 0.008) ** 2) * 0.8 + 0.35 * (np.sin((uu * 1.0 + np.abs(ww) * 1.6) * 38) > 0.82)
                    alb = np.array([0.16, 0.50, 0.10]) * (0.75 + 0.5 * vnoise(P[m] * 40))[:, None] + np.array([0.18, 0.30, 0.08]) * vein[:, None]
                    glow_ = np.array([0.55, 0.85, 0.20]) * (np.clip(-(nn @ MOON), 0, 1) ** 1.5 * 0.0 + np.clip(D[m] @ MOON, 0, 1) ** 6)[:, None] * 0.7
                else:
                    q = P[m]
                    alb = np.array([0.14, 0.42, 0.16]) * (0.8 + 0.4 * vnoise(q * 25))[:, None]
                    glow_ = 0.0
                diff = np.clip(nn @ MOON, 0, 1)[:, None] * MOONC * 1.2 + np.clip(nn @ FILL, 0, 1)[:, None] * np.array([0.55, 0.62, 0.8]) + np.array([0.12, 0.14, 0.22])
                ndv = np.clip(-(nn * D[m]).sum(1), 0, 1)
                R = D[m] + 2 * ndv[:, None] * nn
                sheen = (0.04 + 0.5 * (1 - ndv) ** 5)[:, None] * sky(R, t) + MOONC * (np.clip(R @ MOON, 0, 1) ** 120 * 1.5)[:, None]
                oc[m] = alb * diff + glow_ + sheen
            col[k] = oc
            depth[k] = To[k]
        L = np.zeros((ROWS, COLS, 3)); L[self.iy, self.ix] = col
        Z = np.full((ROWS, COLS), np.inf); Z[self.iy, self.ix] = depth
        for (ang, vh, vu), t0_ in zip(self.spray, (T_IMP, T_IMP + 2.82)):
            ss = t - t0_
            if not 0 < ss < 2.0: continue
            y = 0.02 + vu * ss - 0.5 * G * 1.6 * ss * ss
            P = np.stack([np.cos(ang) * (0.1 + vh * ss), y, np.sin(ang) * (0.1 + vh * ss)], 1)[y > 0]
            if not len(P): continue
            q = P - eye
            z = q @ f
            px, py = W / 2 + F * (q @ r) / z, CY - F * (q @ up) / z
            cx, cy = cells(px, py)
            ok = (z > 0.05) & (cx >= 0) & (cx < COLS) & (cy >= 0) & (cy < ROWS) & (py < 955)
            ok[ok] &= z[ok] < Z[cy[ok], cx[ok]]
            L[cy[ok], cx[ok]] = np.array([0.85, 0.95, 1.0]) * 1.3
        lum = L.max(2)
        chars = np.zeros((ROWS, COLS), int)
        lvl = np.clip((np.clip(lum, 0, 1.4) ** 0.62 * (len(RAMP) - 1) + (self.noise - 0.5) * 6).astype(int), 0, len(RAMP) - 1)
        solid = np.isfinite(Z)
        lvl = np.where(solid, np.maximum(lvl, 5), lvl)
        chars = RAMP_IDX[lvl]
        faint = (~solid) & (lum < 0.09)
        chars = np.where(faint, np.where(self.noise > 0.55, IDX["."], 0), chars)
        star = STAR & ~solid
        chars = np.where(star, np.where(lum > 0.5, IDX["+"], IDX["·"]), chars)
        cols = (L * 1.6).astype(np.float32)
        cols[faint] = np.maximum(cols[faint], 0.05)
        chars[int(955 / CH):] = 0
        cols *= smooth(0, 1.2, t) * (1 - 0.0 * smooth(DURATION - 0.6, DURATION, t))
        im = Image.fromarray((glow(to_pixels(chars, cols), 0.75, 4) * 255).astype(np.uint8))
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
