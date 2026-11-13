"""Prepare WikiArt, train/evaluate a DCGAN, and build the assignment report."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data_process import prepare
from train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="WikiArt image directory or ZIP")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, help="Optional seeded random image subset")
    parser.add_argument("--prep-workers", type=int, default=4,
                        help="Parallel image-decoding threads while preparing the cache")
    parser.add_argument("--skip-undecodable", action="store_true",
                        help="Leave out images that cannot be decoded (listed in the cache "
                             "manifest) instead of stopping preparation")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-images", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()
    if not (args.cache_dir / "manifest.json").exists():
        if not args.data_root:
            parser.error("WikiArt is not cached. Download the assigned dataset, then supply "
                         "--data-root /path/to/wikiart.zip (or its image directory).")
        prepare(args.data_root, args.cache_dir, max_images=args.max_images, seed=args.seed,
                workers=args.prep_workers, skip_undecodable=args.skip_undecodable)
    elif args.max_images is not None:
        manifest = json.loads((args.cache_dir / "manifest.json").read_text())
        if manifest["max_images"] != args.max_images:
            parser.error("The existing cache was prepared with a different --max-images setting. "
                         "Choose a separate --cache-dir for this subset.")
    run_dir = train(args)
    if not args.skip_eval and not args.skip_report:
        from report import build_report
        build_report(run_dir)


if __name__ == "__main__":
    main()
