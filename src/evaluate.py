"""Measure Ignite tutorial IS/FID on disjoint, held-out WikiArt image sets."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from data_process import CachedImageDataset, HW2_DIR, sha256_file, write_json
from model import Generator
from train import choose_device, versions


def prepare_inception(images):
    """Undo tanh normalization before PIL conversion, bilinear resize to 299."""
    pixels = images.detach().cpu().add(1).mul(127.5).clamp(0, 255).round().to(torch.uint8)
    prepared = []
    for pixel in pixels:
        image = Image.fromarray(pixel.permute(1, 2, 0).numpy())
        image = image.resize((299, 299), Image.Resampling.BILINEAR)
        prepared.append(pil_to_tensor(image).float().div_(255))
    return torch.stack(prepared)


def compute_scores(fake, real, control):
    """Use precomputed Inception probabilities so all metrics share one model."""
    from ignite.metrics import FID, InceptionScore

    scores = {}
    diagnostic_warnings = []
    for name, probabilities in (("generated", fake), ("real_control", control)):
        fid = FID(num_features=1000, feature_extractor=torch.nn.Identity(), device="cpu")
        inception = InceptionScore(num_features=1000, feature_extractor=torch.nn.Identity(),
                                  device="cpu")
        fid.update((probabilities, real))
        inception.update(probabilities)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fid_value = float(fid.compute())
        diagnostic_warnings.extend(str(w.message) for w in captured)
        is_value = float(inception.compute())
        if not np.isfinite(fid_value) or not np.isfinite(is_value):
            raise FloatingPointError(f"Non-finite {name} IS/FID")
        scores[name] = {"fid": fid_value, "inception_score": is_value}
    return scores, sorted(set(diagnostic_warnings))


@torch.no_grad()
def evaluate(checkpoint_path, cache_dir=None, output_dir=None, device="auto",
             n_images=2000, batch_size=32, threads=4):
    from ignite.metrics.gan.utils import InceptionModel

    if n_images < 2 or batch_size < 1 or threads < 1:
        raise ValueError("Need at least two evaluation images and positive batch size/threads")
    torch.set_num_threads(threads)
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    device = choose_device(device)
    cache_dir = Path(cache_dir or config["cache_dir"]).resolve()
    dataset = CachedImageDataset(cache_dir, "test")
    if sha256_file(cache_dir / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("Evaluation dataset does not match the trained model")
    for filename, key in (("images.npy", "images_sha256"), ("splits.npz", "splits_sha256")):
        if sha256_file(cache_dir / filename) != dataset.manifest[key]:
            raise ValueError(f"Evaluation cache {filename} has changed")
    if 2 * n_images > len(dataset):
        raise ValueError(f"Need {2 * n_images:,} held-out images for disjoint reference/control "
                         f"sets; found {len(dataset):,}. Use --n-images <= {len(dataset) // 2}.")
    output_dir = Path(output_dir or checkpoint_path.parent.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config["seed"] + 1000
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=rng)
    reference_indices, control_indices = order[:n_images], order[n_images:2 * n_images]
    noise_rng = torch.Generator().manual_seed(seed + 1)
    generator = Generator(config["latent_dim"], config["width"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    inception = InceptionModel(return_features=False, device=device)
    features = {name: [] for name in ("generated", "reference", "control")}
    for start in range(0, n_images, batch_size):
        end = min(n_images, start + batch_size)
        fake = generator(torch.randn(end - start, config["latent_dim"], 1, 1,
                                     generator=noise_rng).to(device)).cpu()
        reference = torch.stack([dataset[int(i)] for i in reference_indices[start:end]])
        control = torch.stack([dataset[int(i)] for i in control_indices[start:end]])
        for name, images in (("generated", fake), ("reference", reference), ("control", control)):
            probabilities = inception(prepare_inception(images).to(device)).cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Inception probabilities are not finite")
            features[name].append(probabilities)
        print(f"Inception features: {end:,}/{n_images:,} images per set", flush=True)
    features = {name: torch.cat(batches) for name, batches in features.items()}
    scores, diagnostic_warnings = compute_scores(features["generated"], features["reference"],
                                                features["control"])
    results = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": checkpoint["epoch"], "source_kind": config["source_kind"],
        "source_url": config["source_url"], "seed": seed, "device": str(device),
        "n_images_per_set": n_images, "batch_size": batch_size,
        "real_split": "held-out test; never used for adversarial training",
        "manifest_sha256": config["manifest_sha256"],
        "reference_cache_indices": dataset.indices[reference_indices.numpy()].tolist(),
        "control_cache_indices": dataset.indices[control_indices.numpy()].tolist(),
        "fake_sampling": f"CPU torch.Generator seed {seed + 1}; standard-normal z",
        "protocol": {
            "implementation": "pytorch-ignite FID and InceptionScore",
            "feature_extractor": "ImageNet-pretrained torchvision Inception-v3 class probabilities",
            "feature_dimensions": 1000,
            "image_preprocessing": "[-1,1] -> [0,1], uint8 round, PIL bilinear 299x299, float [0,1]",
            "normalization": "No additional mean/std normalization; Ignite tutorial [0,1] inputs",
            "fid_note": "Tutorial probability-space FID; not standard 2048-dimensional pool3 FID",
            "is_note": "Ignite aggregate Inception Score over all generated images; no split averaging",
            "tutorial": "https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/",
        },
        "scores": scores, "numerical_warnings": diagnostic_warnings,
        "versions": versions(), "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    np.savez_compressed(output_dir / "inception_probabilities.npz",
                        **{name: value.numpy() for name, value in features.items()})
    write_json(output_dir / "metrics.json", results)
    write_json(output_dir / "status.json", {"phase": "evaluated", "epoch": checkpoint["epoch"],
               "evaluation_complete": True, "source_kind": config["source_kind"]})
    print(json.dumps(scores, indent=2), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--n-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.cache_dir, args.output_dir, args.device,
             args.n_images, args.batch_size, args.threads)


if __name__ == "__main__":
    main()
