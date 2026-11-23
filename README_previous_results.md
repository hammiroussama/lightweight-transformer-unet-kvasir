# Polyp Segmentation with Attention and Transformers

A PyTorch computer-vision project for segmenting polyp regions in colonoscopy images.
I implemented and compared U-Net variants with attention gates and a Transformer bottleneck,
including reproducible training, evaluation, mask prediction, and visual comparisons.

**Attention + Transformer U-Net: 87.44% test Dice and 81.05% test IoU**
on 150 held-out Kvasir-SEG images.

**Stack:** Python | PyTorch | OpenCV | NumPy | SciPy | Matplotlib | pytest

## Recorded results: extended training

Both models used identical 700/150/150 train/validation/test splits, seed 42,
256 x 256 images, batch size four, and the same extended-training recipe.
Checkpoints were selected by validation Dice; test images were not used for gradient updates.

| Test metric | Transformer U-Net | Attention + Transformer U-Net |
|---|---:|---:|
| Dice | 87.23% | 87.44% |
| IoU | 80.42% | 81.05% |
| Precision | 89.24% | 91.59% |
| Recall | 90.32% | 87.31% |
| HD95, lower is better | 25.93 px | 25.16 px |
| Parameters | 11.97M | 12.06M |
| Forward time per image | 0.82 ms | 1.91 ms |

The combined model led by **0.21 percentage points in Dice**
and **0.63 points in IoU** in this single-seed comparison.
It improved precision and HD95, with lower recall and longer measured forward time.
These small gains do not establish consistent superiority across training seeds.

Timing is measured per image in batches of four on an RTX 4090, excluding preprocessing.
HD95 is measured in resized-image pixels. Variability across test images is recorded in
the JSON files; it is not uncertainty across repeated training runs.

![Extended test comparison](docs/images/extended_test_comparison.png)

## Training behavior

Transformer-only stopped at epoch 98 and selected epoch 73. The combined model completed
100 epochs and selected epoch 82. Both used a maximum of 100 epochs, five warmup epochs,
cosine decay, gradient clipping, and early-stopping patience of 25.

![Training and validation curves](docs/images/extended_learning_curves.png)

## Segmentation examples

These predictions come from the two selected checkpoints. The examples are the first four
validation filenames in sorted order, chosen independently of model performance.
They are illustrative validation examples, not additional test results.

![Current checkpoint predictions](docs/images/extended_segmentation_examples.png)

[Example selection and checkpoint paths](docs/images/extended_examples_manifest.json)

## Saved evidence

- [Transformer-only results](results/transformer-extended-20261115-164954/metrics.json)
- [Combined-model results](results/attention-transformer-extended-20261116-200334/metrics.json)
- [Combined-model configuration](results/attention-transformer-extended-20261116-200334/config.json)
- [Combined-model training history](results/attention-transformer-extended-20261116-200334/history.csv)
- [Shared split](results/attention-transformer-extended-20261116-200334/split.csv)

## Historical experiments

The original four-model experiment used a different training recipe. Its split has not
been verified against the extended runs, so these scores must not be interpreted as a
matched comparison against the new results. Historical files are preserved unchanged.

<details>
<summary>Original architecture comparison</summary>

| Model | Historical test Dice | Historical test IoU |
|---|---:|---:|
| U-Net | 76.35% | 66.60% |
| Attention U-Net | 82.31% | 73.89% |
| Transformer U-Net | 85.22% | 77.66% |
| Proposed | 84.84% | 76.84% |

"Proposed" is the legacy name for Attention + Transformer U-Net.
Transformer-only led on Dice and IoU in this historical experiment.

[Original metrics](results/summary.csv)

</details>

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
repeat both models across seeds, and include an untouched external evaluation dataset.

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
