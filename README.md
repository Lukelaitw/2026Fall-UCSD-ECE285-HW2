# ECE 285 HW2: DCGAN on WikiArt

A DCGAN trained on the Kaggle [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart) at 64 × 64, scored with Inception Score and FID from pytorch-ignite. The two-page NeurIPS-format report is `report/hw2_report.pdf`.

The run in the report cached all 81,444 images (none undecodable), used 65,155 for training and held out 16,289, and trained for 50 epochs in 16 minutes (19.3 s per epoch) on an NVIDIA RTX PRO 6000 Blackwell MIG 1g.24gb slice on UCSD DataHub. Run outputs are not stored in git; the final checkpoint is attached to the [v1.0 release](https://github.com/Lukelaitw/2026Fall-UCSD-ECE285-HW2/releases/tag/v1.0) (see [Pretrained checkpoint](#pretrained-checkpoint)).

Ignite tutorial IS/FID (1,000-dimensional class probabilities, 2,000 images per set, the same held-out reference and generator noise for every checkpoint):

| Checkpoint | FID (lower is better) | IS (higher is better) |
|---|---|---|
| Epoch 10 | 0.0960 | 2.36 |
| Epoch 20 | 0.0370 | 2.66 |
| Epoch 30 | 0.0374 | 2.90 |
| Epoch 40 | 0.0361 | 2.92 |
| Epoch 50 (final) | 0.0283 | 2.91 |
| Disjoint real images (control) | 0.0125 | 3.69 |

## Run

Install the dependencies with `python -m pip install -r requirements.txt`, and run every command from the repository root.

Download WikiArt from the link above. The preprocessor accepts the downloaded ZIP directly, avoiding a second full-size extracted copy; it also accepts an existing image directory. It reads and resizes one image at a time, and stores a 64x64 uint8 cache (about 1 GB for 81,000 images). Keep the processed cache outside the raw image folder.

```bash
python run_hw2.py \
  --data-root /path/to/wikiart.zip \
  --output-dir outputs/dcgan_wikiart_e050
```

This prepares all images, trains for 50 epochs, evaluates 2,000 generated images against held-out images, and builds `report/hw2_report.pdf`. The report needs `pdflatex`, `pdfinfo`, and `pdftoppm` (TeX Live and Poppler); both rendered pages are written to `report/preview/` for checking. Add `--skip-report` for a machine without LaTeX/Poppler, or `--skip-eval` to train only.

`--device auto` selects CUDA, then MPS, then CPU. A GPU is preferable for the full dataset. Training prints the measured epoch time and remaining estimate. Checkpoints are resumable if training is stopped.

For a smaller explicitly selected experiment:

```bash
python run_hw2.py \
  --data-root /path/to/wikiart.zip --max-images 10000 \
  --cache-dir data/subset10000 --epochs 20 --eval-images 1000 \
  --output-dir outputs/dcgan_subset10000_e020
```

The subset is sampled across the complete filename list using seed 42. It is not the first 10,000 images of one alphabetically ordered style. Its report records the subset size.

`--prep-workers N` decodes images in N parallel threads. Images are read in archive order but stored in sorted name order, so the cache does not depend on N. `--skip-undecodable` leaves out images that cannot be decoded and lists them, with the error, under `skipped_images` in the cache manifest instead of stopping; without it, one bad image stops preprocessing.

### Training on UCSD DataHub

The Kaggle archive is 33.7 GB, larger than the roughly 30 GB per-user DataHub workspace quota. The archive is therefore downloaded to the server's local disk (`/tmp`), which does not count against the quota but is erased when the server stops, and the 64x64 cache is built next to it. Only run outputs are written to the workspace. The dataset is public, so no Kaggle token is needed:

```bash
mkdir -p /tmp/wikiart
curl -L -C - -o /tmp/wikiart/wikiart.zip \
  https://www.kaggle.com/api/v1/datasets/download/steubk/wikiart
python run_hw2.py --data-root /tmp/wikiart/wikiart.zip --cache-dir /tmp/wikiart/wikiart64 \
  --prep-workers 7 --skip-undecodable --output-dir outputs/dcgan_wikiart_e050 \
  --epochs 50 --threads 8 --skip-report
```

The archive downloaded on 2026-11-10 had 33,662,671,673 bytes and MD5 `82abbd043f481213d0d244b90210aee7` (as reported by Google Cloud Storage). DataHub sometimes assigns a Blackwell GPU slice, for which the course image's PyTorch 2.2 has no kernels; the runs used torch 2.7.1+cu128, torchvision 0.22.1+cu128, and pytorch-ignite 0.5.5. Build the report on a machine with LaTeX after copying the run directory back (`src/report.py --run-dir outputs/dcgan_wikiart_e050`). The checkpoint path recorded on the server does not exist locally, so the report uses the copy in the run's `checkpoints/` folder after checking its SHA-256.

## Implementation and settings

- `src/data_process.py`: recursively read folders or ZIP members, apply EXIF orientation, convert to RGB, resize the shorter edge to 64 with bilinear interpolation, and center crop. A seeded 80/20 image split separates training and evaluation. No labels or augmentation are used. The memory-mapped cache records image paths, split indices, and hashes; malformed images stop preprocessing.
- `src/model.py`: a DCGAN generator with 3,576,704 parameters and discriminator with 2,765,568 parameters at the default width. The generator uses latent dimension 100, transposed convolutions, batch normalization, ReLU, and tanh. The discriminator uses strided convolutions, batch normalization, LeakyReLU, and a scalar logit.
- `src/train.py`: one discriminator step followed by one non-saturating generator step, using `BCEWithLogitsLoss` and Adam (`lr=0.0002`, `betas=(0.5, 0.999)`). Defaults: batch size 128, 50 epochs, base width 64, seed 42. Real and generated tensors are both in `[-1,1]`. The last minibatch is retained.
- `src/evaluate.py`: compute FID and Inception Score with [PyTorch-Ignite's assignment tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/). A single pretrained Inception-v3 supplies the 1,000 class probabilities used by both metrics. Real reference and real control sets are disjoint held-out images. Normalize images back to `[0,1]` before PIL conversion, bilinearly resize to 299x299, and keep float inputs in `[0,1]`, following the tutorial's metric input convention. **This probability-space FID is not the usual 2,048-dimensional pool3 FID and must not be compared to published pool3 scores.**
- `src/report.py`: fill the two-page report only from a completed WikiArt run, check its measured checkpoint hash and full history, compile the PDF, enforce two pages, and render both pages. Synthetic smoke tests are rejected. Evaluations of earlier checkpoints in `eval_by_epoch/` join the results table only if they used the final evaluation's reference and control images, latent vectors, and metric protocol. The loss figure is redrawn at its printed width so its labels stay legible. Two sentences in `report/template.tex` describe this run's loss curves and samples (marked by comments); recheck them before building a report from another run.

The first evaluation downloads the pretrained Inception-v3 weights (about 104 MB). A report requires at least twice `--eval-images` held-out images for the disjoint reference/control sets. With the default 20% split, a 10,000-image subset supports at most 1,000 per set.

## Saved outputs

Each run has its own directory:

- `config.json`, `status.json`: settings, package versions, data provenance, and current phase.
- `history.csv`, `loss.png`: per-epoch mean losses and discriminator probabilities.
- `real_images.png`, `samples/epoch_000.png`, `samples/epoch_001.png`, etc.: real examples and 64 outputs from the same fixed latent vectors each epoch.
- `checkpoints/latest.pt`: atomic epoch-boundary checkpoint with both networks, Adam states, fixed noise, and random-generator states. At default width its trained size is roughly 76 MB. Numbered checkpoints are retained every 10 epochs and at the final epoch.
- `metrics.json`, `inception_probabilities.npz`: measured IS/FID, real control, exact checkpoint hash, held-out image indices, seeds, input protocol, and extracted probabilities. Numerical warnings are retained.

Resume a run to a **total** target of 100 epochs:

```bash
python run_hw2.py \
  --resume outputs/dcgan_wikiart_e050/checkpoints/latest.pt \
  --epochs 100
```

Use the same `--cache-dir` when resuming a custom cache. Network size, optimizer settings, batch size, seed, and evaluation sample count are restored from the checkpoint. Resume is allowed only when its saved epoch matches the end of the CSV history, so it needs the run's own `history.csv`; the released checkpoint alone cannot be resumed. Old evaluation results are preserved under their original epoch.

Evaluate an immutable numbered checkpoint separately:

```bash
python src/evaluate.py \
  --checkpoint outputs/dcgan_wikiart_e050/checkpoints/epoch_050.pt
python src/report.py \
  --run-dir outputs/dcgan_wikiart_e050
```

## Pretrained checkpoint

`epoch_050.pt` (76 MB) is the final checkpoint of the 50-epoch run and the one scored in the report. It is attached to the [v1.0 release](https://github.com/Lukelaitw/2026Fall-UCSD-ECE285-HW2/releases/tag/v1.0) rather than stored in git:

```bash
curl -L --create-dirs -o outputs/dcgan_wikiart_e050/checkpoints/epoch_050.pt \
  https://github.com/Lukelaitw/2026Fall-UCSD-ECE285-HW2/releases/download/v1.0/epoch_050.pt
```

It holds both networks (`generator`, `discriminator`), both Adam states, `epoch`, `global_step`, `config`, the 64 `fixed_noise` vectors used for every sample grid, and random-generator states. Its SHA-256 is `7393afc250d8a25a2dab1c444aae74dee34fcd58c55babac6bbef3adafa0ad48`. To redraw the generated grid of the report's Figure 2:

```python
import sys
import torch
from torchvision.utils import save_image

sys.path.insert(0, "src")
from model import Generator

ckpt = torch.load("outputs/dcgan_wikiart_e050/checkpoints/epoch_050.pt",
                  map_location="cpu", weights_only=True)
generator = Generator(ckpt["config"]["latent_dim"], ckpt["config"]["width"])
generator.load_state_dict(ckpt["generator"])
generator.eval()
with torch.no_grad():
    images = generator(ckpt["fixed_noise"])  # torch.randn(64, 100) gives new samples
save_image((images + 1) / 2, "samples.png", nrow=8, padding=2)
```

## Validation

```bash
python -m pytest tests -q
python scripts/smoke_test.py
```

Tests check folder/ZIP equivalence, disjoint splits, invalid input rejection, updates to both adversaries, pixel ranges, exact CPU continuation after a checkpoint, the report's checkpoint lookup in a copied run, its per-epoch score table, and the loss plot size. The smoke test performs two epochs and a real pretrained-Inception evaluation on a small synthetic image fixture. Its numbers demonstrate execution only and cannot be used as WikiArt scores.
