"""Post-ready clips: the first N seconds of each model's best-fight video, with a name label.

    uv run --with pillow --with imageio-ffmpeg python scripts/make_clips.py --fonts <dir with Cinzel.ttf>

Reads each bench's best_videos.json (the recordings live on Windows, under
%USERPROFILE%\\Videos\\SoulsBench), overlays a small dark tag with the model name at the top
centre, and re-encodes to H.264/AAC 1080p60 with faststart (what Twitter/X plays natively; the
OBS originals are ~45 Mbps HEVC). ffmpeg comes from the imageio-ffmpeg wheel, no system install.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

CLIPS = [  # bench, adapter key in best_videos.json, label (same names as the chart)
    ("full50-hosted", "openai", "OpenAI GPT-6 Luna"),
    ("full50-hosted", "jev", "TypeSafe Jev"),
    ("full50-hosted", "pplx", "Perplexity Decider"),
    ("full50-kev", "kev", "Kev 0.8B"),
    ("full50-laya-typed", "laya-typed", "Laya 210M"),
]
TITLE = (0xE6, 0xC9, 0x8F, 255)
PANEL = (0x14, 0x11, 0x0E, 200)
EDGE = (0x9C, 0x8B, 0x6E, 150)


def windows_to_wsl(path: str) -> Path:
    drive, rest = path.split(":", 1)
    return Path(f"/mnt/{drive.lower()}") / rest.replace("\\", "/").lstrip("/")


def label_png(text: str, font_path: Path, out: Path) -> None:
    font = ImageFont.truetype(str(font_path), 46)
    pad_x, pad_y = 34, 16
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 2 * pad_x, box[3] - box[1] + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=PANEL, outline=EDGE, width=2)
    d.text((pad_x - box[0], pad_y - box[1]), text, font=font, fill=TITLE)
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--seconds", type=float, default=30.0, help="0 = the whole video")
    ap.add_argument("--file", help="one Windows video path instead of the benches' best videos")
    ap.add_argument("--label", help="label for --file")
    ap.add_argument("--out", type=Path, help="default: a clips/ folder next to the source videos")
    args = ap.parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if args.file:
        jobs = [(windows_to_wsl(args.file), args.label or Path(args.file).stem)]
    else:
        jobs = [
            (
                windows_to_wsl(
                    json.loads((args.runs / f"bench_{bench}" / "best_videos.json").read_text())[
                        key
                    ]["path"]
                ),
                label,
            )
            for bench, key, label in CLIPS
        ]
    out = args.out or jobs[0][0].parent.parent / "clips"  # ...\\Videos\\SoulsBench\\clips
    out.mkdir(parents=True, exist_ok=True)
    for src, label in jobs:
        tag = out / f".{src.stem}_label.png"
        label_png(label, args.fonts / "Cinzel.ttf", tag)
        length = f"{int(args.seconds)}s" if args.seconds else "full"
        dest = out / f"{src.stem}_{length}.mp4"
        trim = ["-t", str(args.seconds)] if args.seconds else []
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                *trim,
                "-i",
                str(src),
                "-i",
                str(tag),
                "-filter_complex",
                "[0:v][1:v]overlay=(W-w)/2:40,format=yuv420p[v]",
                "-map",
                "[v]",
                "-map",
                "0:a?",
                # capped at 20 Mbps: X recommends <= 25 Mbps for 1080p uploads
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "19",
                "-maxrate",
                "20M",
                "-bufsize",
                "40M",
                "-profile:v",
                "high",
                "-r",
                "60",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=True,
        )
        tag.unlink()
        print(f"{label:<20} {dest}  ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
