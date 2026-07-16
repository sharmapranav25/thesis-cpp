"""Side-by-side: real eval vs PHP-MyBB synth. Picks 12 of each at random."""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    real_dir = Path("data/eval")
    php_dir = Path("data/synth_mybb_test")
    rng = random.Random(0)
    real = rng.sample(sorted(real_dir.glob("*.png")) + sorted(real_dir.glob("*.PNG")), 12)
    php = rng.sample(sorted(php_dir.glob("*.png")), min(12, len(list(php_dir.glob("*.png")))))
    W, H, pad, label_h = 200, 60, 6, 18
    canvas = Image.new("RGB", (W * 2 + pad * 3, label_h + (H + pad) * 12 + pad), (235, 235, 235))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    d.text((pad, 2), "REAL", fill=(0, 0, 0), font=font)
    d.text((pad * 2 + W, 2), "PHP synth", fill=(0, 0, 0), font=font)
    for i in range(12):
        y = label_h + i * (H + pad) + pad
        canvas.paste(Image.open(real[i]).convert("RGB").resize((W, H), Image.BICUBIC), (pad, y))
        if i < len(php):
            canvas.paste(Image.open(php[i]).convert("RGB").resize((W, H), Image.BICUBIC), (pad * 2 + W, y))
    big = canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST)
    out = Path("data/_php_vs_real.png")
    big.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
