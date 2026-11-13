"""Cache WikiArt from a directory or ZIP directly as 64 x 64 RGB images."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

HW2_DIR = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_URL = "https://www.kaggle.com/datasets/steubk/wikiart"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def is_image(name):
    path = PurePosixPath(name)
    return (path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(part.startswith(".") or part == "__MACOSX" for part in path.parts))


def decode_image(data):
    """EXIF orientation, RGB, bilinear shorter edge 64, center crop: uint8 (3, 64, 64)."""
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = TF.center_crop(TF.resize(image, 64, Image.Resampling.BILINEAR), 64)
        return np.asarray(image).transpose(2, 0, 1)


@contextmanager
def image_source(path):
    """Yield sorted names, a reader returning one image's bytes, and a sort key for the
    cheapest reading order (ZIP members by position in the archive). Nothing is extracted."""
    path = Path(path)
    if path.is_dir():
        names = sorted(p.relative_to(path).as_posix() for p in path.rglob("*")
                       if p.is_file() and is_image(p.relative_to(path).as_posix()))
        yield names, lambda name: (path / name).read_bytes(), lambda name: name
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist() if is_image(info.filename)]
            infos = {info.filename: info for info in members}
            if len(infos) != len(members):
                raise ValueError("ZIP has duplicate image member names")
            yield (sorted(infos), lambda name: archive.read(infos[name]),
                   lambda name: infos[name].header_offset)
    else:
        raise FileNotFoundError(f"WikiArt image directory or ZIP not found: {path}")


def prepare(data_root, output_dir, max_images=None, seed=42, test_fraction=0.2,
            source_kind="wikiart", source_url=SOURCE_URL, workers=1, skip_undecodable=False):
    """Resize shorter edge to 64, then center crop. Store uint8, not float32.

    `workers` threads decode in parallel (Pillow releases the GIL while decoding and
    resizing). Images are read in archive order but stored in sorted name order, so the
    cache depends on neither. With skip_undecodable, images that cannot be decoded are
    left out and listed in the manifest instead of stopping preprocessing."""
    data_root, output_dir = Path(data_root).resolve(), Path(output_dir).resolve()
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if max_images is not None and max_images < 4:
        raise ValueError("max_images must be at least 4")
    if workers < 1:
        raise ValueError("workers must be positive")
    if source_kind not in {"wikiart", "synthetic_smoke_test"}:
        raise ValueError("Unknown source_kind")
    if output_dir == data_root or (data_root.is_dir() and data_root in output_dir.parents):
        raise ValueError("Put the processed cache outside the input image directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images.npy", "images.tmp.npy", "images.kept.npy", "splits.npz",
                 "manifest.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"{output_dir / name} exists; choose a new cache directory")
    start = time.monotonic()
    with image_source(data_root) as (names, read_image, position):
        source_count = len(names)
        if max_images is not None and max_images < len(names):
            # Random subset across all styles, rather than the first style alphabetically.
            order = np.random.default_rng(seed).choice(len(names), max_images, replace=False)
            names = [names[i] for i in sorted(order.tolist())]
        if len(names) < 4:
            raise ValueError(f"Need at least four images, found {len(names)} in {data_root}")
        slot = {name: index for index, name in enumerate(names)}
        images = np.lib.format.open_memmap(output_dir / "images.tmp.npy", mode="w+",
                                           dtype=np.uint8, shape=(len(names), 3, 64, 64))
        decoded = np.zeros(len(names), dtype=bool)
        skipped = []
        finished = 0

        def store(name, future):
            nonlocal finished
            try:
                images[slot[name]] = future.result()
                decoded[slot[name]] = True
            except Exception as error:
                if not skip_undecodable:
                    raise ValueError(f"Cannot decode {name}; cache is incomplete") from error
                skipped.append({"path": name, "error": f"{type(error).__name__}: {error}"})
            finished += 1
            if finished % 1000 == 0 or finished == len(names):
                print(f"Cached {finished:,}/{len(names):,} paintings, {len(skipped)} skipped "
                      f"({time.monotonic() - start:.1f}s)", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            for name in sorted(names, key=position):
                pending.append((name, pool.submit(decode_image, read_image(name))))
                if len(pending) >= 4 * workers:
                    store(*pending.popleft())
            while pending:
                store(*pending.popleft())
        images.flush()
        del images
    kept = np.flatnonzero(decoded)
    if len(kept) < 4:
        raise ValueError(f"Only {len(kept)} images could be decoded; need at least four")
    if skipped:
        full = np.load(output_dir / "images.tmp.npy", mmap_mode="r")
        compact = np.lib.format.open_memmap(output_dir / "images.kept.npy", mode="w+",
                                            dtype=np.uint8, shape=(len(kept), 3, 64, 64))
        for first in range(0, len(kept), 4096):
            compact[first:first + 4096] = full[kept[first:first + 4096]]
        compact.flush()
        del compact, full
        (output_dir / "images.tmp.npy").unlink()
        (output_dir / "images.kept.npy").replace(output_dir / "images.npy")
        names = [names[i] for i in kept]
        skipped.sort(key=lambda entry: entry["path"])
    else:
        (output_dir / "images.tmp.npy").replace(output_dir / "images.npy")
    order = np.random.default_rng(seed).permutation(len(names))
    test_count = max(2, min(len(names) - 2, int(round(len(names) * test_fraction))))
    np.savez(output_dir / "splits.npz", train=order[test_count:], test=order[:test_count])
    manifest = {
        "format_version": 1, "source_kind": source_kind, "source_url": source_url,
        "source_path": str(data_root), "source_image_count": source_count,
        "image_count": len(names), "train_count": len(names) - test_count,
        "test_count": test_count, "seed": seed, "test_fraction": test_fraction,
        "max_images": max_images, "shape": [len(names), 3, 64, 64], "dtype": "uint8",
        "preprocessing": "EXIF orientation, RGB, bilinear shorter edge 64, center crop 64",
        "augmentation": "none", "image_paths": names, "skipped_images": skipped,
        "images_sha256": sha256_file(output_dir / "images.npy"),
        "splits_sha256": sha256_file(output_dir / "splits.npz"),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"Ready: {manifest['train_count']:,} train / {test_count:,} held out; {output_dir}",
          flush=True)
    return manifest


class CachedImageDataset(Dataset):
    """Memory-mapped RGB images; return float32 tensors in [-1, 1]."""

    def __init__(self, cache_dir, split="train"):
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        with np.load(self.cache_dir / "splits.npz", allow_pickle=False) as splits:
            train, test = splits["train"].copy(), splits["test"].copy()
        count = self.manifest["image_count"]
        combined = np.concatenate((train, test))
        if len(combined) != count or not np.array_equal(np.sort(combined), np.arange(count)):
            raise ValueError("Train/test indices must partition the cache without overlap")
        if len(train) != self.manifest["train_count"] or len(test) != self.manifest["test_count"]:
            raise ValueError("Manifest and split sizes disagree")
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be train, test, or all")
        self.indices = {"train": train, "test": test, "all": np.arange(count)}[split]
        self.images = np.load(self.cache_dir / "images.npy", mmap_mode="r", allow_pickle=False)
        if self.images.shape != (count, 3, 64, 64) or self.images.dtype != np.uint8:
            raise ValueError("Cache must contain uint8 images of shape (N, 3, 64, 64)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = np.array(self.images[int(self.indices[index])], copy=True)
        return torch.from_numpy(image).float().div_(127.5).sub_(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory of WikiArt images, or the downloaded ZIP")
    parser.add_argument("--output-dir", type=Path, default=HW2_DIR / "data/processed")
    parser.add_argument("--max-images", type=int, help="Optional seeded random subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4, help="Parallel decoding threads")
    parser.add_argument("--skip-undecodable", action="store_true",
                        help="Leave out images that cannot be decoded and list them in the "
                             "manifest, instead of stopping")
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
