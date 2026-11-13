"""Exercise the complete pipeline on explicitly synthetic images, not WikiArt."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_process import prepare, write_json


def create_images(destination, count=80):
    destination.mkdir(parents=True)
    rng = np.random.default_rng(123)
    for index in range(count):
        width, height = int(rng.integers(70, 130)), int(rng.integers(70, 130))
        background = tuple(int(v) for v in rng.integers(0, 256, 3))
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        for _ in range(5):
            x0, x1 = sorted(rng.integers(0, width, 2).tolist())
            y0, y1 = sorted(rng.integers(0, height, 2).tolist())
            draw.rectangle((x0, y0, x1, y1), fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
        image.save(destination / f"synthetic_{index:03d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" /
                        datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--skip-inception", action="store_true")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new smoke-test directory: {destination}")
    create_images(destination / "synthetic_input")
    prepare(destination / "synthetic_input", destination / "cache",
            source_kind="synthetic_smoke_test", source_url="not applicable: synthetic rectangles")
    command = [sys.executable, str(ROOT / "src/train.py"),
               "--cache-dir", str(destination / "cache"), "--output-dir", str(destination / "run"),
               "--epochs", "2", "--batch-size", "8", "--latent-dim", "16", "--width", "8",
               "--device", "cpu", "--eval-images", "8", "--eval-batch-size", "4", "--threads", "4"]
    if args.skip_inception:
        command.append("--skip-eval")
    subprocess.run(command, check=True, cwd=ROOT)
    checkpoint = destination / "run/checkpoints/latest.pt"
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["epoch"] != 2 or saved["config"]["source_kind"] != "synthetic_smoke_test":
        raise AssertionError("Unexpected checkpoint metadata")
    summary = {"result": "passed", "source_kind": "synthetic_smoke_test", "epochs": 2,
               "inception_evaluation": not args.skip_inception,
               "note": "Pipeline validation only. Not HW2 WikiArt training or report results.",
               "run_directory": str(destination / "run")}
    write_json(destination / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
