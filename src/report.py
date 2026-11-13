"""Build the two-page NeurIPS report from a completed, measured WikiArt run."""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess

from data_process import HW2_DIR, sha256_file
from train import plot_history


def tex_escape(value):
    replacements = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
    return "".join(replacements.get(character, character) for character in str(value))


def measured_checkpoint(run_dir, metrics):
    """Return the evaluated checkpoint, also when the run was copied from the training machine."""
    recorded = Path(metrics["checkpoint"])
    path = recorded if recorded.is_file() else Path(run_dir) / "checkpoints" / recorded.name
    if sha256_file(path) != metrics["checkpoint_sha256"]:
        raise ValueError("The measured checkpoint has changed since evaluation")
    return path


# Fields that make two evaluations' scores comparable: same data, sets, noise, and metric.
PROTOCOL_KEYS = ("manifest_sha256", "seed", "n_images_per_set", "reference_cache_indices",
                 "control_cache_indices", "fake_sampling", "protocol")


def checkpoint_evaluations(run_dir, final):
    """Earlier checkpoints scored into eval_by_epoch/ (evaluate.py --output-dir), then the final."""
    earlier = [json.loads(path.read_text())
               for path in Path(run_dir).glob("eval_by_epoch/*/metrics.json")]
    for metrics in earlier:
        if any(metrics[key] != final[key] for key in PROTOCOL_KEYS):
            raise ValueError(f"Epoch {metrics['epoch']} was not scored with the final "
                             "evaluation's protocol")
    return sorted(earlier, key=lambda metrics: metrics["epoch"]) + [final]


def build_report(run_dir, output_dir=None):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    if config["source_kind"] != "wikiart":
        raise ValueError("A synthetic smoke test cannot be used for the HW2 report")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if metrics["source_kind"] != "wikiart" or metrics["epoch"] != config["epochs"]:
        raise ValueError("Evaluation must describe the final WikiArt checkpoint")
    if metrics["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Training and evaluation datasets differ")
    measured_checkpoint(run_dir, metrics)
    evaluations = checkpoint_evaluations(run_dir, metrics)
    with (run_dir / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in history] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Training history must contain every completed epoch exactly once")
    if metrics["protocol"]["feature_dimensions"] != 1000:
        raise ValueError("This report template describes the Ignite tutorial probability-space FID")
    for executable in ("pdflatex", "pdfinfo", "pdftoppm"):
        if not shutil.which(executable):
            raise RuntimeError(f"{executable} is required to compile and verify the report")
    output_dir = Path(output_dir or HW2_DIR / "report").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    for source, target in (("real_images.png", "real.png"),
                           (f"samples/epoch_{config['epochs']:03d}.png", "generated.png")):
        shutil.copy2(run_dir / source, figures / target)
    # Redraw the losses at their printed size (text width), so labels stay legible.
    plot_history(run_dir / "history.csv", figures / "loss.png", figsize=(7.0, 1.65))
    style = HW2_DIR / "report/neurips_2025.sty"
    if style.resolve() != (output_dir / style.name).resolve():
        shutil.copy2(style, output_dir / style.name)
    last = history[-1]
    scores = metrics["scores"]
    subset = ("All discovered images were selected." if config["source_image_count"]
              == config["selected_image_count"] else
              f"A seed-{config['data_seed']} random subset of {config['selected_image_count']:,} "
              f"out of {config['source_image_count']:,} discovered images was selected.")
    warning = ("SciPy emitted a covariance warning; the FID values should be treated as approximate."
               if any(evaluation["numerical_warnings"] for evaluation in evaluations) else
               "")
    generated = [evaluation["scores"]["generated"] for evaluation in evaluations]
    values = {
        "EPOCHS": str(config["epochs"]), "TRAIN": f"{config['train_count']:,}",
        "TEST": f"{config['test_count']:,}", "LATENT": str(config["latent_dim"]),
        "WIDTH": str(config["width"]), "BATCH": str(config["batch_size"]),
        "LR": f"{config['learning_rate']:g}", "SEED": str(config["seed"]),
        "DATASEED": str(config["data_seed"]), "REALGRID": str(min(64, config["train_count"])),
        "DEVICE": tex_escape(config["device"]), "SUBSET": subset,
        "GPARAMS": f"{config['generator_parameters']:,}",
        "DPARAMS": f"{config['discriminator_parameters']:,}",
        "GCHANNELS": r"\!\to\!".join(str(config["width"] * factor) for factor in (8, 4, 2, 1)),
        "GLAST": f"{float(last['g_loss']):.4f}", "DLAST": f"{float(last['d_loss']):.4f}",
        "DREAL": f"{float(last['d_real']):.4f}", "DFAKE": f"{float(last['d_fake']):.4f}",
        "STEPS": f"{int(last['steps']):,}", "N": f"{metrics['n_images_per_set']:,}",
        "FID": f"{scores['generated']['fid']:.4f}",
        "IS": f"{scores['generated']['inception_score']:.2f}",
        "REALFID": f"{scores['real_control']['fid']:.4f}",
        "REALIS": f"{scores['real_control']['inception_score']:.2f}",
        "CKPTCOLS": "c" * len(evaluations),
        "EPOCHROW": " & ".join(str(evaluation["epoch"]) for evaluation in evaluations),
        "FIDROW": " & ".join(f"{score['fid']:.4f}" for score in generated),
        "ISROW": " & ".join(f"{score['inception_score']:.2f}" for score in generated),
        "WARNING": warning, "IGNITE": tex_escape(metrics["versions"]["pytorch-ignite"]),
    }
    template = (HW2_DIR / "report/template.tex").read_text()
    for name, value in values.items():
        template = template.replace(f"@@{name}@@", value)
    if re.search(r"@@[A-Z]+@@", template):
        raise ValueError("Unfilled report template fields")
    destination = output_dir / "hw2_report.tex"
    destination.write_text(template)
    for _ in range(2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                 destination.name], cwd=output_dir, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Report compilation failed:\n{result.stdout[-6000:]}")
    pdf = output_dir / "hw2_report.pdf"
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    if not pages or pages.group(1) != "2":
        raise RuntimeError("Report must be exactly two pages; inspect and adjust its layout")
    preview = output_dir / "preview"
    preview.mkdir(exist_ok=True)
    subprocess.run(["pdftoppm", "-scale-to", "1600", "-png", str(pdf), str(preview / "page")],
                   check=True)
    print(f"Two-page report: {pdf}\nReview both rendered pages in {preview} before submission.")
    return pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build_report(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
