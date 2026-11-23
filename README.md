# Polyp Segmentation with Attention and Transformers

A PyTorch project that segments polyp regions in colonoscopy images using U-Net variants,
attention gates, and a Transformer bottleneck. Includes reproducible training, mask prediction,
visual overlays, and a paired comparison across three training seeds.

**Attention + Transformer U-Net: 88.24% mean test Dice
and 81.87% mean test IoU across three seeds.**

**Stack:** Python | PyTorch | OpenCV | NumPy | SciPy | Matplotlib | pytest

## Recorded results: three-seed study

Both architectures used the same 700/150/150 train/validation/test split, seeds 42, 123,
and 3407, and the matched extended-training recipe. Each checkpoint was selected by
validation Dice, then all six checkpoints were evaluated on the same 150 test images.
These are three training repetitions on one split, not three independent test datasets.

| Test metric | Transformer U-Net | Attention + Transformer U-Net |
|---|---:|---:|
| Dice | 87.74% ± 0.59 | 88.24% ± 0.39 |
| IoU | 81.18% ± 0.71 | 81.87% ± 0.30 |
| Precision | 91.23% ± 0.23 | 91.77% ± 0.47 |
| Recall | 88.32% ± 1.23 | 88.69% ± 0.52 |
| HD95 (lower is better) | 24.30 px ± 1.04 | 23.85 px ± 1.01 |
| Forward time per image | 1.25 ms ± 0.01 | 2.38 ms ± 0.01 |

Values are mean ± sample standard deviation across training seeds, not confidence intervals.
HD95 is in pixels at 256×256 resolution. Forward timing is per image in batches of four on
an RTX 4090, excluding preprocessing; it is not end-to-end application latency.

The combined model improved mean Dice by **0.50 percentage points** and won on
two of three seeds. It showed lower observed seed variability, with greater inference cost.
Three seeds do not establish statistical significance or universal superiority.

| Seed | Transformer Dice | Attention + Transformer Dice | Difference (points) |
|---|---:|---:|---:|
| 42 | 88.22% | 87.98% | -0.25 |
| 123 | 87.08% | 88.69% | +1.61 |
| 3407 | 87.91% | 88.06% | +0.15 |

![Three-seed test comparison](docs/images/seed_study_test_comparison.png)

## Evidence and reproducibility

- [Study plan and source fingerprints](results/seed-study-20261122-190743-088986/plan.json)
- [Frozen configurations](results/seed-study-20261122-190743-088986/configs)
- [Shared split](results/seed-study-20261122-190743-088986/split.csv)
- [Validation summary](results/seed-study-20261122-190743-088986/summary.json)
- [Test summary and checkpoint hashes](results/seed-study-20261122-190743-088986/test/summary.json)
- [Per-run test metrics and per-image scores](results/seed-study-20261122-190743-088986/test)
- [Comparison CSV](results/seed-study-20261122-190743-088986/test/comparison.csv)

The study used strict deterministic algorithms, the math attention backend, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Historical runs used different execution settings,
so their timings and seed-42 scores should not be treated as identical reproductions.

The test split had already been inspected in earlier experiments. This is an exploratory
portfolio benchmark, not a fully blinded final evaluation. Future tuning should use
validation data; external evaluation is needed before broader generalization claims.
This project is not a clinically validated diagnostic system.

## Earlier experiments and qualitative examples

[Previous results, learning curves, and segmentation examples](README_previous_results.md)
are preserved as historical evidence. Those images were generated from the earlier
single-seed checkpoints, not the checkpoints in the new study. The four-model historical
comparison is not a controlled comparison against this study.

## Setup and dataset

Use Python 3.10 or 3.11 and a suitable PyTorch installation:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Obtain [Kvasir-SEG from its official website](https://datasets.simula.no/kvasir-seg/).
Place images under `data/Kvasir-SEG/images/` and masks under `data/Kvasir-SEG/masks/`.
Dataset files and trained checkpoints are not included in this repository.

To reproduce the extended comparison, restore the exported split before training:

```bash
mkdir -p splits
cp -n results/attention-transformer-extended-20261116-200334/split.csv splits/kvasir_seed42.csv
```

If a split already exists, verify its image assignments match the exported split.
The combined configuration directly references the exported reference split.

## Train and evaluate

Run from the repository root, using a new output directory for every experiment:

```bash
python train.py --config configs/transformer_unet_extended.yaml --output runs/transformer_new
python train.py --config configs/proposed_model_extended.yaml --output runs/combined_new
python evaluate.py --checkpoint runs/combined_new/checkpoints/best.pt --split val
```

After selecting and freezing settings using validation data, evaluate the test split:

```bash
python evaluate.py --checkpoint runs/combined_new/checkpoints/best.pt --split test
```

Avoid repeatedly tuning against test scores. The recorded Transformer-only test result
was already known before the combined run; this is an exploratory portfolio comparison,
not a fully blinded final benchmark. Future improvements should use validation data,
include an untouched external evaluation dataset. The three-seed study above now measures variation across training seeds.

## Predict

```bash
python predict.py --checkpoint runs/combined_new/checkpoints/best.pt --image path/to/image.jpg --output predictions/mask.png
```

This exports a binary mask and a red overlay at the original image dimensions.

## Project structure

```text
configs/    Experiment settings
datasets/   Paired loading, augmentation, and splits
models/     U-Net, attention gates, and Transformer bottleneck
utils/      Training, losses, metrics, checkpoints
tests/      Model, metric, and training-safety tests
results/    Measured results, histories, and exported splits
docs/       Protocol and visualizations
```

## Limitations

One dataset and one seed per configuration do not establish clinical generalization or
reliable superiority. GPU nondeterminism warnings occurred during combined-model training.
This project is an engineering portfolio demonstration, not a clinical diagnostic system.

## License and dataset attribution

Code: [MIT license](LICENSE). Dataset: Kvasir-SEG, Jha et al. (2020),
[official source and citation](https://datasets.simula.no/kvasir-seg/).
Follow the dataset's research/education terms and attribution requirements; the code
license does not grant dataset redistribution rights.
