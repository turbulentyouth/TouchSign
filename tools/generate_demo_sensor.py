"""Generate a demo tactile-sensor frame for the TouchSign UI mockup.

This is a visual asset generator, not part of the runtime recognition pipeline.
It renders an imaginary gel-sensor surface with an embossed zodiac seal.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo" / "tactile-rat.png"

W, H = 1600, 1000
CX, CY = W / 2, H / 2
FONT_PATH = Path(r"C:\Windows\Fonts\STKAITI.TTF")
SERIF_PATH = Path(r"C:\Windows\Fonts\NotoSerifSC-VF.ttf")
RNG = np.random.default_rng(20261030)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blur_mask(mask: Image.Image, radius: float) -> np.ndarray:
    return np.asarray(mask.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[..., None] if t.ndim == 2 else t
    return a * (1.0 - t) + b * t


# ---------------------------------------------------------------------------
# Gel surface base
# ---------------------------------------------------------------------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
nx = (xx - CX) / (W * 0.56)
ny = (yy - CY) / (H * 0.56)
radius = np.sqrt(nx**2 + ny**2)

center_color = np.array([12, 54, 46], dtype=np.float32)
mid_color = np.array([5, 30, 26], dtype=np.float32)
edge_color = np.array([2, 9, 8], dtype=np.float32)

base = np.empty((H, W, 3), dtype=np.float32)
inner_t = np.clip(radius / 0.72, 0.0, 1.0)[..., None]
base[:] = center_color * (1.0 - inner_t) + mid_color * inner_t
outer_t = np.clip((radius - 0.68) / 0.6, 0.0, 1.0)[..., None]
base = base * (1.0 - outer_t) + edge_color * outer_t

# Slow organic ripples of the illuminated gel.
ripple = (
    np.sin(nx * 8.7 + ny * 3.2) * 0.5
    + np.sin(nx * 3.2 - ny * 7.1 + 1.7) * 0.35
    + np.sin((nx + ny) * 12.0) * 0.15
)
base += (ripple * 4.2)[..., None] * np.array([0.35, 0.85, 0.75], dtype=np.float32)

# Soft mottled lighting.
noise_small = Image.fromarray(
    np.clip(RNG.normal(0.5, 0.18, (H // 4, W // 4)) * 255, 0, 255).astype(np.uint8), "L"
).resize((W, H), Image.Resampling.BICUBIC)
mottle = np.asarray(noise_small.filter(ImageFilter.GaussianBlur(20)), dtype=np.float32) / 255.0
base += (mottle[..., None] - 0.5) * np.array([0.0, 11.0, 9.0], dtype=np.float32)

# Soft central illumination / pressure bloom.
bloom = np.exp(-((radius / 0.7) ** 2) * 2.3)
base += bloom[..., None] * np.array([2.0, 17.0, 14.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Ornament masks: seal disc, rings, 12 earthly branches, and the 鼠 glyph
# ---------------------------------------------------------------------------
ornament = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(ornament)

disc_r = 416
d.ellipse((CX - disc_r, CY - disc_r, CX + disc_r, CY + disc_r), outline=255, width=3)
for r, width in ((384, 2), (438, 1), (458, 1), (472, 4)):
    d.ellipse((CX - r, CY - r, CX + r, CY + r), outline=255, width=width)

# Fine radial ticks: 60 minor, 12 major.
for i in range(60):
    ang = math.tau * i / 60 - math.pi / 2
    major = i % 5 == 0
    r0 = 423 if major else 427
    r1 = 445 if major else 435
    width = 3 if major else 1
    x0, y0 = CX + math.cos(ang) * r0, CY + math.sin(ang) * r0
    x1, y1 = CX + math.cos(ang) * r1, CY + math.sin(ang) * r1
    d.line((x0, y0, x1, y1), fill=255, width=width)

# The twelve earthly branches around the rim.
branch_font = ImageFont.truetype(str(SERIF_PATH), 24)
branches = "子丑寅卯辰巳午未申酉戌亥"
for i, ch in enumerate(branches):
    ang = math.tau * i / 12 - math.pi / 2
    r = 400
    x = CX + math.cos(ang) * r
    y = CY + math.sin(ang) * r
    box = d.textbbox((0, 0), ch, font=branch_font)
    tw, th = box[2] - box[0], box[3] - box[1]
    d.text((x - tw / 2, y - th / 2 - 2), ch, font=branch_font, fill=255)

# Tiny ornamental diamonds between the branches.
for i in range(12):
    ang = math.tau * (i + 0.5) / 12 - math.pi / 2
    x, y = CX + math.cos(ang) * 438, CY + math.sin(ang) * 438
    s = 4
    d.polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], fill=255)

# The large zodiac glyph.
glyph_font = ImageFont.truetype(str(FONT_PATH), 500)
glyph_text = "鼠"
gb = d.textbbox((0, 0), glyph_text, font=glyph_font)
gw, gh = gb[2] - gb[0], gb[3] - gb[1]
gx = CX - gw / 2 - gb[0]
gy = CY - gh / 2 - gb[1] - 6
glyph = Image.new("L", (W, H), 0)
ImageDraw.Draw(glyph).text((gx, gy), glyph_text, font=glyph_font, fill=255)

# Faint square seal frame around the glyph.
pad = 36
seal_l, seal_t = gx + gb[0] - pad, gy + gb[1] - pad
seal_r, seal_b = gx + gb[2] + pad, gy + gb[3] + pad
seal_mask = Image.new("L", (W, H), 0)
sd = ImageDraw.Draw(seal_mask)
sd.rounded_rectangle((seal_l, seal_t, seal_r, seal_b), radius=18, outline=255, width=3)
sd.rectangle((seal_l - 8, seal_t - 8, seal_r + 8, seal_b + 8), outline=255, width=1)

ornament_mask = np.asarray(ornament, dtype=np.float32) / 255.0
glyph_mask = np.asarray(glyph, dtype=np.float32) / 255.0
seal_mask_f = np.asarray(seal_mask, dtype=np.float32) / 255.0

# Height field used for plausible relief lighting.
height = np.zeros((H, W), dtype=np.float32)
height += np.exp(-((radius / 0.74) ** 2) * 2.5) * 18.0

disc_alpha = 1.0 - smoothstep(disc_r - 2, disc_r + 16, np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2))
height += disc_alpha * 5.0

glyph_soft = np.asarray(
    Image.fromarray((glyph_mask * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(10)),
    dtype=np.float32,
) / 255.0
height += glyph_soft * 30.0
height += ornament_mask * 7.0
height += seal_mask_f * 9.0

height_img = Image.fromarray(np.clip(height, 0, 255).astype(np.uint8), "L").filter(
    ImageFilter.GaussianBlur(3.2)
)
height = np.asarray(height_img, dtype=np.float32)

# Surface normal -> embossed directional light.
gy, gx = np.gradient(height)
scale = 0.14
normal = np.dstack((-gx * scale, -gy * scale, np.ones_like(height)))
normal /= np.linalg.norm(normal, axis=2, keepdims=True)
light = np.array([-0.55, -0.65, 0.82], dtype=np.float32)
light /= np.linalg.norm(light)
diffuse = np.clip((normal * light).sum(axis=2), 0.0, 1.0)
specular = np.clip(diffuse, 0.0, 1.0) ** 18
rim_light = np.clip((normal * np.array([0.55, 0.35, 0.76], dtype=np.float32)).sum(axis=2), 0.0, 1.0)

base *= (0.78 + 0.42 * diffuse)[..., None]

jade_light = np.array([154, 235, 206], dtype=np.float32)
jade_deep = np.array([27, 120, 101], dtype=np.float32)
gold = np.array([231, 194, 124], dtype=np.float32)
warm_white = np.array([244, 233, 206], dtype=np.float32)

# Token face has a faintly lighter jade glaze.
token_glaze = disc_alpha * (0.10 + 0.14 * bloom)
base = mix(base, jade_deep, np.clip(token_glaze, 0, 0.28))

# Embossed character, with dark recessed edge and mint-lit face.
glyph_edge = np.clip(
    np.asarray(
        Image.fromarray((glyph_mask * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(4)),
        dtype=np.float32,
    ) / 255.0
    - glyph_mask,
    0,
    1,
)
base = mix(base, np.array([4, 19, 17], dtype=np.float32), glyph_edge * 0.45)
base = mix(base, jade_light, np.clip(glyph_soft * 1.1, 0, 1) * (0.34 + 0.42 * diffuse))
base += (specular * glyph_soft * 1.4)[..., None] * warm_white * 0.22
base += (rim_light * glyph_soft * 0.2)[..., None] * gold

# Gold-on-jade engraved rings and rim characters.
base = mix(base, gold, ornament_mask * (0.28 + 0.25 * diffuse))
base += (ornament_mask * specular * 0.9)[..., None] * gold * 0.28
base = mix(base, gold * 0.65, seal_mask_f * (0.10 + 0.12 * diffuse))

# Pressure halo around the token.
r_px = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
halo = np.exp(-((r_px - 430) / 55) ** 2) * 0.20
base += halo[..., None] * np.array([46, 128, 101], dtype=np.float32)
print_ring = np.exp(-((r_px - 360) / 38) ** 2) * 0.08
base += print_ring[..., None] * np.array([80, 150, 115], dtype=np.float32)

# Topographic contours in the contact region.
contour = np.sin(height * 0.16 + 0.6) * 0.5 + 0.5
contour = (contour**10) * 0.05 * (1.0 - smoothstep(0.55, 1.0, radius))
base += contour[..., None] * np.array([72, 158, 135], dtype=np.float32)

# Glow bloom from the bright glyph and ornamental ring.
glow_src = Image.fromarray(
    np.clip((glyph_mask * 0.75 + ornament_mask * 0.12) * 255, 0, 255).astype(np.uint8), "L"
).filter(ImageFilter.GaussianBlur(36))
glow = np.asarray(glow_src, dtype=np.float32) / 255.0
base += glow[..., None] * np.array([55, 180, 155], dtype=np.float32) * 0.30

# ---------------------------------------------------------------------------
# Surface marker dots (fine reference dots embedded in the gel)
# ---------------------------------------------------------------------------
dots = Image.new("RGBA", (W, H), (0, 0, 0, 0))
dd = ImageDraw.Draw(dots)
for y0 in range(28, H + 28, 44):
    row = (y0 - 28) // 44
    for x0 in range(28 + (row % 2) * 22, W + 28, 44):
        dx = x0 - CX
        dy = y0 - CY
        dist = math.hypot(dx, dy)
        if dist < disc_r - 18:
            # Markers inside the token are faintly magnified by contact.
            rad = 2.2 + 1.1 * math.exp(-((dist / 360) ** 2))
            alpha = 54
        else:
            rad = 1.8
            alpha = 34
        if dist > 760:
            continue
        dd.ellipse((x0 - rad, y0 - rad, x0 + rad, y0 + rad), fill=(188, 244, 225, alpha))
dots_np = np.asarray(dots, dtype=np.float32)
dot_a = (dots_np[..., 3:4] / 255.0) * 0.65
base = base * (1 - dot_a) + dots_np[..., :3] * dot_a

# ---------------------------------------------------------------------------
# Finishing: vignette, grain, color grade
# ---------------------------------------------------------------------------
vignette = 1.0 - 0.55 * smoothstep(0.62, 1.45, radius)
base *= vignette[..., None]

# Slight teal-to-gold color grade in the highlights.
luma = base.mean(axis=2, keepdims=True) / 255.0
base += (np.clip(luma - 0.35, 0, 1) ** 1.5) * np.array([8, 14, 0], dtype=np.float32)

# Fine camera grain.
grain = RNG.normal(0, 2.5, (H, W, 1)).astype(np.float32)
base += grain

# Rounded outer framing shadow, matching a camera view.
frame = Image.new("L", (W, H), 0)
ImageDraw.Draw(frame).rounded_rectangle((10, 10, W - 10, H - 10), radius=26, outline=255, width=2)
frame = frame.filter(ImageFilter.GaussianBlur(1))
frame_np = np.asarray(frame, dtype=np.float32) / 255.0
base = mix(base, np.array([3, 12, 10], dtype=np.float32), frame_np * 0.5)

base = np.clip(base, 0, 255).astype(np.uint8)
image = Image.fromarray(base, "RGB")

# Very subtle chromatic edge bloom.
blur = image.filter(ImageFilter.GaussianBlur(18))
image = Image.blend(image, Image.blend(blur, image, 0.66), 0.18)

OUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUT, quality=95)
print(OUT)
