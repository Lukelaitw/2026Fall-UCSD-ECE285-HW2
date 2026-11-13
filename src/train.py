"""Train a WikiArt DCGAN, preserving losses, fixed samples, and restart state."""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data_process import CachedImageDataset, HW2_DIR, sha256_file, write_json
from model import make_models

HISTORY_FIELDS = ["epoch", "steps", "images", "g_loss", "d_loss", "d_real", "d_fake", "seconds"]


def choose_device(name="auto"):
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process; use a GPU-enabled terminal or CPU")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")
    return torch.device(name)


def setup(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in
               ("torch", "torchvision", "numpy", "Pillow", "pytorch-ignite", "scipy")}}


def adversarial_step(generator, discriminator, real, g_optimizer, d_optimizer, noise):
    """One discriminator update, then one non-saturating generator update."""
    criterion = nn.BCEWithLogitsLoss()
    generator.train()
    discriminator.train()
    d_optimizer.zero_grad(set_to_none=True)
    g_optimizer.zero_grad(set_to_none=True)
    fake = generator(noise)
    real_logits, fake_logits = discriminator(real), discriminator(fake.detach())
    d_loss = (criterion(real_logits, torch.ones_like(real_logits))
              + criterion(fake_logits, torch.zeros_like(fake_logits)))
    if not torch.isfinite(d_loss):
        raise FloatingPointError("Discriminator loss is not finite")
    d_loss.backward()
    d_optimizer.step()
    discriminator.requires_grad_(False)
    try:
        generated_logits = discriminator(fake)
        g_loss = criterion(generated_logits, torch.ones_like(generated_logits))
        if not torch.isfinite(g_loss):
            raise FloatingPointError("Generator loss is not finite")
        g_loss.backward()
        g_optimizer.step()
    finally:
        discriminator.requires_grad_(True)
    return {"g_loss": float(g_loss.detach()), "d_loss": float(d_loss.detach()),
            "d_real": float(real_logits.detach().sigmoid().mean()),
            "d_fake": float(fake_logits.detach().sigmoid().mean())}


def save_checkpoint(path, generator, discriminator, g_optimizer, d_optimizer,
                    epoch, global_step, config, fixed_noise, shuffle_rng, noise_rng):
    checkpoint = {
        "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(), "d_optimizer": d_optimizer.state_dict(),
        "epoch": epoch, "global_step": global_step, "config": config,
        "fixed_noise": fixed_noise.cpu(), "torch_rng": torch.get_rng_state(),
        "shuffle_rng": shuffle_rng.get_state(), "noise_rng": noise_rng.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        checkpoint["mps_rng"] = torch.mps.get_rng_state()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_checkpoint(checkpoint, generator, discriminator, g_optimizer, d_optimizer,
                       shuffle_rng, noise_rng, device):
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    for optimizer in (g_optimizer, d_optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
    torch.set_rng_state(checkpoint["torch_rng"])
    shuffle_rng.set_state(checkpoint["shuffle_rng"])
    noise_rng.set_state(checkpoint["noise_rng"])
    random.setstate(checkpoint["python_rng"])
    if device.type == "cuda" and "cuda_rng" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if device.type == "mps" and "mps_rng" in checkpoint:
        torch.mps.set_rng_state(checkpoint["mps_rng"])


@torch.no_grad()
def save_samples(generator, fixed_noise, device, destination):
    was_training = generator.training
    generator.eval()
    try:
        images = generator(fixed_noise.to(device)).cpu()
        if not torch.isfinite(images).all():
            raise FloatingPointError("Generated samples contain non-finite pixels")
        save_image((images + 1) / 2, destination, nrow=8, padding=2)
    finally:
        generator.train(was_training)


def plot_history(csv_path, destination, figsize=(9.0, 2.8)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(csv_path).open(newline="") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise ValueError("No training history to plot")
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    for field, label in (("g_loss", "Generator"), ("d_loss", "Discriminator")):
        axes[0].plot(epochs, [float(r[field]) for r in history], label=label)
    for field, label in (("d_real", "D(real)"), ("d_fake", "D(fake), before D update")):
        axes[1].plot(epochs, [float(r[field]) for r in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="BCE loss")
    axes[1].set(xlabel="Epoch", ylabel="Mean probability", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def train(args):
    for name in ("epochs", "batch_size", "latent_dim", "width", "threads", "save_every",
                 "eval_images", "eval_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    device = choose_device(args.device)
    setup(args.seed, args.threads)
    dataset = CachedImageDataset(args.cache_dir, "train")
    cache_dir = dataset.cache_dir
    for filename, key in (("images.npy", "images_sha256"), ("splits.npz", "splits_sha256")):
        if sha256_file(cache_dir / filename) != dataset.manifest[key]:
            raise ValueError(f"{filename} differs from its preparation manifest")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        output_dir = Path(args.resume).resolve().parent.parent
        if args.output_dir and args.output_dir.resolve() != output_dir:
            raise ValueError("Resume writes to the checkpoint's original run directory")
        config = dict(checkpoint["config"])
        current_hash = sha256_file(cache_dir / "manifest.json")
        if config["manifest_sha256"] != current_hash:
            raise ValueError("Resume cache does not match the original dataset")
        if args.epochs <= checkpoint["epoch"]:
            raise ValueError("--epochs is the total target and must exceed the resumed epoch")
        if (output_dir / "metrics.json").exists():
            # Preserve previously measured results under their actual checkpoint epoch.
            previous = json.loads((output_dir / "metrics.json").read_text())
            write_json(output_dir / f"metrics_epoch_{previous['epoch']:03d}.json", previous)
            (output_dir / "metrics.json").unlink()
        with (output_dir / "history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != checkpoint["epoch"] or int(rows[-1]["epoch"]) != checkpoint["epoch"]:
            raise ValueError("History is ahead of or behind this checkpoint; resume latest.pt")
        config.update(epochs=args.epochs, device=str(device), threads=args.threads,
                      cache_dir=str(cache_dir))
    else:
        output_dir = args.output_dir or HW2_DIR / "outputs" / datetime.now().strftime("dcgan_%Y%m%d_%H%M%S_%f")
        output_dir = Path(output_dir).resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "latent_dim": args.latent_dim, "width": args.width, "image_size": 64,
            "learning_rate": args.learning_rate, "betas": [0.5, 0.999],
            "seed": args.seed, "device": str(device), "threads": args.threads,
            "save_every": args.save_every, "eval_images": args.eval_images,
            "eval_batch_size": args.eval_batch_size, "cache_dir": str(cache_dir),
            "train_count": len(dataset), "test_count": dataset.manifest["test_count"],
            "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "source_kind": dataset.manifest["source_kind"],
            "source_url": dataset.manifest["source_url"],
            "source_image_count": dataset.manifest["source_image_count"],
            "selected_image_count": dataset.manifest["image_count"],
            "data_seed": dataset.manifest["seed"],
            "test_fraction": dataset.manifest["test_fraction"],
            "preprocessing": dataset.manifest["preprocessing"],
            "augmentation": dataset.manifest["augmentation"],
            "objective": "BCEWithLogits: D = BCE(real, 1) + BCE(fake, 0); G = BCE(fake, 1)",
            "started_utc": datetime.now(timezone.utc).isoformat(), "versions": versions(),
        }
    if not args.skip_eval and (config["eval_images"] < 2
                              or 2 * config["eval_images"] > config["test_count"]):
        raise ValueError(f"Evaluation needs two disjoint held-out sets. "
                         f"Use --eval-images between 2 and {config['test_count'] // 2}, "
                         "or --skip-eval.")
    for folder in ("checkpoints", "samples"):
        (output_dir / folder).mkdir(exist_ok=True)
    generator, discriminator = make_models(config["latent_dim"], config["width"], device)
    config["generator_parameters"] = sum(p.numel() for p in generator.parameters())
    config["discriminator_parameters"] = sum(p.numel() for p in discriminator.parameters())
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config["learning_rate"],
                                   betas=tuple(config["betas"]))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"],
                                   betas=tuple(config["betas"]))
    shuffle_rng = torch.Generator().manual_seed(config["seed"] + 1)
    noise_rng = torch.Generator().manual_seed(config["seed"] + 2)
    fixed_noise = torch.randn(64, config["latent_dim"], 1, 1,
                             generator=torch.Generator().manual_seed(config["seed"] + 3))
    start_epoch, global_step = 1, 0
    if checkpoint:
        restore_checkpoint(checkpoint, generator, discriminator, g_optimizer, d_optimizer,
                           shuffle_rng, noise_rng, device)
        fixed_noise = checkpoint["fixed_noise"]
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
    else:
        save_samples(generator, fixed_noise, device, output_dir / "samples/epoch_000.png")
        real = torch.stack([dataset[i] for i in range(min(64, len(dataset)))])
        save_image((real + 1) / 2, output_dir / "real_images.png", nrow=8, padding=2)
        with (output_dir / "history.csv").open("w", newline="") as stream:
            csv.writer(stream).writerow(HISTORY_FIELDS)
    write_json(output_dir / "config.json", config)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=0, generator=shuffle_rng, drop_last=False,
                        pin_memory=device.type == "cuda")
    print(f"Training {len(dataset):,} images on {device}; {len(loader):,} steps/epoch\n"
          f"Run directory: {output_dir}", flush=True)
    if dataset.manifest["source_kind"] == "synthetic_smoke_test":
        print("SYNTHETIC SMOKE TEST: these are not WikiArt experiment results.", flush=True)
    write_json(output_dir / "status.json", {"phase": "training", "target_epochs": args.epochs})
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.monotonic()
            totals = dict.fromkeys(("g_loss", "d_loss", "d_real", "d_fake"), 0.0)
            seen = 0
            for step, real in enumerate(loader, 1):
                real = real.to(device)
                noise = torch.randn(len(real), config["latent_dim"], 1, 1,
                                    generator=noise_rng).to(device)
                metrics = adversarial_step(generator, discriminator, real,
                                           g_optimizer, d_optimizer, noise)
                for key, value in metrics.items():
                    totals[key] += value * len(real)
                seen += len(real)
                global_step += 1
                if step % 100 == 0:
                    print(f"Epoch {epoch}/{args.epochs}, step {step}/{len(loader)}: "
                          f"G {metrics['g_loss']:.4f}, D {metrics['d_loss']:.4f}", flush=True)
            elapsed = time.monotonic() - started
            row = {"epoch": epoch, "steps": global_step, "images": seen,
                   **{key: value / seen for key, value in totals.items()}, "seconds": elapsed}
            save_samples(generator, fixed_noise, device,
                         output_dir / f"samples/epoch_{epoch:03d}.png")
            with (output_dir / "history.csv").open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(row)
            save_checkpoint(output_dir / "checkpoints/latest.pt", generator, discriminator,
                            g_optimizer, d_optimizer, epoch, global_step, config, fixed_noise,
                            shuffle_rng, noise_rng)
            if epoch % config["save_every"] == 0 or epoch == args.epochs:
                save_checkpoint(output_dir / f"checkpoints/epoch_{epoch:03d}.pt", generator,
                                discriminator, g_optimizer, d_optimizer, epoch, global_step,
                                config, fixed_noise, shuffle_rng, noise_rng)
            plot_history(output_dir / "history.csv", output_dir / "loss.png")
            write_json(output_dir / "status.json", {"phase": "training", "epoch": epoch,
                       "target_epochs": args.epochs, "latest_checkpoint": "checkpoints/latest.pt"})
            eta_hours = elapsed * (args.epochs - epoch) / 3600
            print(f"Epoch {epoch}/{args.epochs}: G={row['g_loss']:.4f}, D={row['d_loss']:.4f}, "
                  f"{elapsed:.1f}s; remaining approximately {eta_hours:.2f}h", flush=True)
        write_json(output_dir / "status.json", {"phase": "trained", "epoch": args.epochs,
                   "latest_checkpoint": "checkpoints/latest.pt", "evaluation_complete": False})
        if not args.skip_eval:
            from evaluate import evaluate
            evaluate(output_dir / f"checkpoints/epoch_{args.epochs:03d}.pt", cache_dir, output_dir,
                     device=str(device), n_images=config["eval_images"],
                     batch_size=config["eval_batch_size"], threads=args.threads)
        print(f"Training finished: {output_dir}", flush=True)
    except BaseException as error:
        write_json(output_dir / "status.json", {"phase": "interrupted" if isinstance(error, KeyboardInterrupt)
                   else "failed", "error": str(error), "latest_checkpoint": "checkpoints/latest.pt"})
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=HW2_DIR / "data/processed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-images", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
