#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "demo_blisters"


def blister_image(seed: int, defect: str | None, rows: int, cols: int) -> Image.Image:
    rng = random.Random(seed)
    width, height = 768, 512
    image = Image.new("RGB", (width, height), (236, 239, 239))
    draw = ImageDraw.Draw(image)

    for _ in range(900):
        x = rng.randrange(width)
        y = rng.randrange(height)
        value = rng.randrange(216, 251)
        draw.point((x, y), fill=(value, value, value))

    blister = (54, 46, width - 54, height - 46)
    draw.rounded_rectangle(blister, radius=18, fill=(207, 214, 212), outline=(147, 160, 160), width=4)

    cell_w = (blister[2] - blister[0]) / cols
    cell_h = (blister[3] - blister[1]) / rows
    missing_index = rng.randrange(rows * cols) if defect == "missing" else -1
    spot_index = rng.randrange(rows * cols) if defect == "spot" else -1
    broken_index = rng.randrange(rows * cols) if defect == "broken" else -1

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            cx = int(blister[0] + (col + 0.5) * cell_w + rng.randrange(-3, 4))
            cy = int(blister[1] + (row + 0.5) * cell_h + rng.randrange(-3, 4))
            rx = int(cell_w * 0.28)
            ry = int(cell_h * 0.30)
            cavity = (cx - rx - 10, cy - ry - 10, cx + rx + 10, cy + ry + 10)
            draw.ellipse(cavity, fill=(184, 194, 194), outline=(127, 139, 139), width=3)
            draw.ellipse((cavity[0] + 8, cavity[1] + 8, cavity[2] - 8, cavity[3] - 8), fill=(221, 226, 226))
            if index == missing_index:
                draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(166, 178, 178))
                continue

            tablet_fill = (244 + rng.randrange(-5, 4), 245 + rng.randrange(-4, 5), 239 + rng.randrange(-4, 5))
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=tablet_fill, outline=(185, 190, 186), width=2)
            draw.arc((cx - rx + 10, cy - ry + 8, cx + rx - 8, cy + ry - 6), 205, 330, fill=(255, 255, 255), width=3)
            if index == spot_index:
                sx = cx + rng.randrange(-rx // 2, rx // 2)
                sy = cy + rng.randrange(-ry // 2, ry // 2)
                draw.ellipse((sx - 9, sy - 9, sx + 9, sy + 9), fill=(92, 84, 79))
            if index == broken_index:
                draw.line((cx - rx + 8, cy - 6, cx + rx - 6, cy + 10), fill=(118, 125, 126), width=5)
                draw.line((cx - 4, cy - ry + 4, cx + 6, cy + ry - 6), fill=(151, 157, 158), width=3)

    if defect == "hair":
        points = []
        start_x = rng.randrange(120, width - 160)
        start_y = rng.randrange(90, height - 120)
        for i in range(8):
            points.append((start_x + i * 34, start_y + int(18 * rng.uniform(-1, 1))))
        draw.line(points, fill=(42, 38, 35), width=3, joint="curve")

    if defect == "foil_tear":
        x = rng.randrange(120, width - 140)
        y = rng.randrange(90, height - 130)
        tear = [(x, y), (x + 46, y + 15), (x + 21, y + 52), (x + 74, y + 78), (x + 8, y + 70)]
        draw.polygon(tear, fill=(98, 111, 112))
        draw.line(tear + [tear[0]], fill=(68, 79, 82), width=3)

    return image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic blister images for local demos.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--good", type=int, default=16)
    parser.add_argument("--defect", type=int, default=10)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    good_dir = args.out / "good"
    defect_dir = args.out / "defect"
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"rows": args.rows, "cols": args.cols, "good": [], "defect": []}
    for index in range(args.good):
        path = good_dir / f"good_{index + 1:03d}.png"
        blister_image(args.seed + index, None, args.rows, args.cols).save(path)
        manifest["good"].append(str(path.relative_to(args.out)))

    defect_types = ["missing", "spot", "broken", "hair", "foil_tear"]
    for index in range(args.defect):
        defect = defect_types[index % len(defect_types)]
        path = defect_dir / f"{defect}_{index + 1:03d}.png"
        blister_image(args.seed + 1000 + index, defect, args.rows, args.cols).save(path)
        manifest["defect"].append({"path": str(path.relative_to(args.out)), "defect": defect})

    with (args.out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Generated {args.good} good and {args.defect} defective synthetic blister images in {args.out}")


if __name__ == "__main__":
    main()

