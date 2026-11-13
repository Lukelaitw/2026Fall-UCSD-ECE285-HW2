"""Checks for data leakage, adversarial updates, and exact CPU continuation."""

import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_process import CachedImageDataset, prepare, sha256_file
from evaluate import prepare_inception
from model import make_models
from report import build_report, checkpoint_evaluations, measured_checkpoint
from train import adversarial_step, plot_history, train


@pytest.fixture
def raw_images(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    for index in range(20):
        folder = root / ("style_a" if index % 2 else "style_b")
        folder.mkdir(exist_ok=True)
        image = Image.new("RGB", (100 + index, 80), (index * 12, 40, 255 - index * 12))
        image.save(folder / f"{index:03d}.png")
    metadata = root / "__MACOSX"
    metadata.mkdir()
    (metadata / "._ignored.jpg").write_bytes(b"not an image")
    (root / "labels.csv").write_text("ignored metadata")
    return root


def test_directory_zip_equivalence_and_partition(raw_images, tmp_path):
    archive = tmp_path / "wikiart.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for path in raw_images.rglob("*"):
            if path.is_file():
                stream.write(path, path.relative_to(raw_images))
    a = prepare(raw_images, tmp_path / "folder_cache", max_images=16)
    b = prepare(archive, tmp_path / "zip_cache", max_images=16)
    assert a["image_paths"] == b["image_paths"]
    assert a["images_sha256"] == b["images_sha256"]
    training = CachedImageDataset(tmp_path / "folder_cache", "train")
    test = CachedImageDataset(tmp_path / "folder_cache", "test")
    assert not set(training.indices) & set(test.indices)
    assert len(training) + len(test) == 16
    assert training[0].shape == (3, 64, 64)
    assert -1 <= training[0].min() <= training[0].max() <= 1
    assert a["source_image_count"] == 20
    with pytest.raises(FileExistsError):
        prepare(raw_images, tmp_path / "folder_cache")


def test_corrupt_image_cannot_produce_completed_cache(raw_images, tmp_path):
    (raw_images / "broken.png").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Cannot decode"):
        prepare(raw_images, tmp_path / "cache")
    assert not (tmp_path / "cache/manifest.json").exists()


def test_parallel_decoding_of_reverse_ordered_zip_matches_folder(raw_images, tmp_path):
    # Members are stored in reverse name order, so the archive's physical order differs
    # from the sorted order in which the cache stores images.
    archive = tmp_path / "reversed.zip"
    files = sorted(path for path in raw_images.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "w") as stream:
        for path in reversed(files):
            stream.write(path, path.relative_to(raw_images).as_posix())
    folder = prepare(raw_images, tmp_path / "folder")
    zipped = prepare(archive, tmp_path / "zip", workers=3)
    assert zipped["image_paths"] == folder["image_paths"]
    assert zipped["images_sha256"] == folder["images_sha256"]


def test_undecodable_images_can_be_skipped_and_are_recorded(raw_images, tmp_path):
    clean = prepare(raw_images, tmp_path / "clean")
    (raw_images / "style_a" / "broken.jpg").write_bytes(b"not a jpeg")
    lenient = prepare(raw_images, tmp_path / "lenient", workers=2, skip_undecodable=True)
    assert lenient["image_paths"] == clean["image_paths"]
    assert lenient["images_sha256"] == clean["images_sha256"]
    assert lenient["splits_sha256"] == clean["splits_sha256"]
    assert [entry["path"] for entry in lenient["skipped_images"]] == ["style_a/broken.jpg"]
    assert (lenient["source_image_count"], lenient["image_count"]) == (21, 20)


def test_overlapping_splits_are_rejected(raw_images, tmp_path):
    cache = tmp_path / "cache"
    prepare(raw_images, cache)
    with np.load(cache / "splits.npz") as arrays:
        training, test = arrays["train"].copy(), arrays["test"].copy()
    test[0] = training[0]
    np.savez(cache / "splits.npz", train=training, test=test)
    with pytest.raises(ValueError, match="without overlap"):
        CachedImageDataset(cache)


def test_both_adversaries_update_and_generation_is_bounded():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    generator, discriminator = make_models(latent_dim=16, width=8)
    g_before = [p.detach().clone() for p in generator.parameters()]
    d_before = [p.detach().clone() for p in discriminator.parameters()]
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    noise = torch.randn(4, 16, 1, 1)
    metrics = adversarial_step(generator, discriminator, torch.rand(4, 3, 64, 64) * 2 - 1,
                               g_optimizer, d_optimizer, noise)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(g_before, generator.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(d_before, discriminator.parameters()))
    result = generator(noise).detach()
    assert result.shape == (4, 3, 64, 64)
    assert -1 <= result.min() <= result.max() <= 1
    assert all(p.requires_grad for p in discriminator.parameters())


def train_args(cache_dir, output_dir, epochs, resume=None):
    return argparse.Namespace(cache_dir=cache_dir, output_dir=output_dir,
        resume=resume, epochs=epochs, batch_size=4, latent_dim=16, width=8,
        learning_rate=0.0002, seed=42, device="cpu", threads=2,
        save_every=10, eval_images=2, eval_batch_size=2, skip_eval=True)


def test_cpu_resume_matches_uninterrupted_training(raw_images, tmp_path):
    cache = tmp_path / "cache"
    prepare(raw_images, cache, source_kind="synthetic_smoke_test", source_url="test fixture")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(train_args(cache, full, 2))
    train(train_args(cache, resumed, 1))
    checkpoint = resumed / "checkpoints/latest.pt"
    train(train_args(cache, resumed, 2, resume=checkpoint))
    complete = torch.load(full / "checkpoints/latest.pt", weights_only=True)
    continued = torch.load(checkpoint, weights_only=True)
    assert complete["epoch"] == continued["epoch"] == 2
    assert complete["global_step"] == continued["global_step"]
    for name in ("generator", "discriminator"):
        for parameter in complete[name]:
            assert torch.equal(complete[name][parameter], continued[name][parameter]), parameter
    with (resumed / "history.csv").open() as stream:
        assert [row["epoch"] for row in csv.DictReader(stream)] == ["1", "2"]


def test_metric_preprocessing_restores_rgb_range():
    image = torch.empty(2, 3, 64, 64)
    image[0].fill_(-1)
    image[1].fill_(1)
    prepared = prepare_inception(image)
    assert prepared.shape == (2, 3, 299, 299)
    assert torch.equal(prepared[0], torch.zeros_like(prepared[0]))
    assert torch.equal(prepared[1], torch.ones_like(prepared[1]))


def test_report_refuses_synthetic_results(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"source_kind": "synthetic_smoke_test"}))
    with pytest.raises(ValueError, match="synthetic smoke test"):
        build_report(run_dir, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_report_finds_the_evaluated_checkpoint_in_a_copied_run(tmp_path):
    # metrics.json records the absolute checkpoint path on the training server.
    run_dir = tmp_path / "copied_run"
    checkpoint = run_dir / "checkpoints/epoch_002.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trained weights")
    metrics = {"checkpoint": str(tmp_path / "server/run/checkpoints/epoch_002.pt"),
               "checkpoint_sha256": sha256_file(checkpoint)}
    assert measured_checkpoint(run_dir, metrics) == checkpoint
    checkpoint.write_bytes(b"retrained weights")
    with pytest.raises(ValueError, match="changed since evaluation"):
        measured_checkpoint(run_dir, metrics)


def scored(epoch, fid, reference=(1, 2)):
    return {"epoch": epoch, "manifest_sha256": "m", "seed": 1042, "n_images_per_set": 2,
            "reference_cache_indices": list(reference), "control_cache_indices": [3, 4],
            "fake_sampling": "CPU torch.Generator seed 1043; standard-normal z",
            "protocol": {"feature_dimensions": 1000}, "numerical_warnings": [],
            "scores": {"generated": {"fid": fid, "inception_score": 2.0},
                       "real_control": {"fid": 0.01, "inception_score": 3.0}}}


def test_report_includes_checkpoints_scored_with_the_final_protocol(tmp_path):
    run_dir = tmp_path / "run"
    final = scored(30, 0.03)
    for epoch, fid in ((20, 0.04), (10, 0.09)):
        path = run_dir / f"eval_by_epoch/epoch_{epoch:03d}/metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(scored(epoch, fid)))
    evaluations = checkpoint_evaluations(run_dir, final)
    assert [(m["epoch"], m["scores"]["generated"]["fid"]) for m in evaluations] == [
        (10, 0.09), (20, 0.04), (30, 0.03)]
    # Scores against a different reference set cannot share a table with the final ones.
    path.write_text(json.dumps(scored(10, 0.09, reference=(5, 6))))
    with pytest.raises(ValueError, match="protocol"):
        checkpoint_evaluations(run_dir, final)


def test_loss_plot_can_be_drawn_at_its_printed_size(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text("epoch,steps,images,g_loss,d_loss,d_real,d_fake,seconds\n"
                       "1,2,8,1.5,1.2,0.6,0.4,1.0\n2,4,16,1.4,1.1,0.6,0.4,1.0\n")
    plot_history(history, tmp_path / "loss.png", figsize=(7.0, 1.65))
    with Image.open(tmp_path / "loss.png") as image:
        assert image.size == (1400, 330)
