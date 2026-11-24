# ASCII 3D in Python

Animations made only with characters, numpy and Pillow — the ones from my TikTok.
Every animation is a single file, `Nzsqh.py`. Pick a folder, run it, watch it.

## Animations

| Folder | What it does |
|---|---|
| [`letter/`](letter) | An envelope appears, a cursor clicks the heart seal, the letter opens and a red rose blooms above "For You" |
| [`drop/`](drop) | A drop of water swells on a leaf, falls in slow motion and splashes into a pond under the moon (ray-marched 3D) |
| [`firefly/`](firefly) | A firefly takes off from a blade of grass and paints a butterfly of light that flies away (ray-marched 3D) |
| [`street-lamp/`](street-lamp) | A street lamp flickers on in the rain, a neon sign glows and the city appears behind (ray-marched 3D) |
| [`dna/`](dna) | A DNA double helix builds itself: two backbones twist upwards and every base pair (A-T, G-C) snaps into place |

More coming.

The ray-marched ones are heavier: they pre-render for longer before playing (or use `--save` and watch the video).

## How to run

You need Python 3.9+ (Windows, macOS or Linux).

```bash
pip install numpy pillow
cd letter
python Nzsqh.py
```

It pre-renders for a few seconds, then plays smoothly in a window.

To export a video instead:

```bash
pip install imageio-ffmpeg
python Nzsqh.py --save
```

This writes `Nzsqh.mp4` next to the script.

## Customize

Open `Nzsqh.py` and edit the settings at the top: the message under the rose (`MESSAGE`), the signature (`SIGNATURE`), the duration, the size.

## License

MIT — do whatever you want with it, a credit is always appreciated.
