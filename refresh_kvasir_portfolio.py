"""Refresh the portfolio from measured results; run from the repository root.

Default: build charts and actual CPU predictions, then update README.
--charts-only: build numeric charts only; leave README unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = ['transformer-extended-20261115-164954',
        'attention-transformer-extended-20261116-200334']
LABELS = ['Transformer U-Net', 'Attention + Transformer U-Net']
COLORS = ['#4b72b8', '#138575']


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_evidence(root):
    records = []
    for run in RUNS:
        folder = root / 'results' / run
        read_json = lambda p: json.loads((folder / p).read_text(encoding='utf-8'))
        metrics = read_json('metrics.json')
        assert metrics['split'] == 'test' and metrics['test_samples'] == 150
        records.append(dict(run=run, metrics=metrics, config=read_json('config.json'),
                            history=read_csv(folder / 'history.csv'),
                            split=read_csv(folder / 'split.csv')))
    normalize = lambda rows: sorted((r['image'].replace('\\', '/'),
                                    r['mask'].replace('\\', '/'), r['split']) for r in rows)
    assert normalize(records[0]['split']) == normalize(records[1]['split']), 'Splits differ'
    assert records[0]['config']['train'] == records[1]['config']['train'], 'Training recipes differ'
    assert records[0]['config']['seed'] == records[1]['config']['seed'] == 42
    return records


def charts(records, out):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(4)
    keys = ['dice_mean', 'iou_mean', 'precision_mean', 'recall_mean']
    for i, record in enumerate(records):
        values = [100 * record['metrics'][key] for key in keys]
        bars = ax.bar(x + (i - .5) * .36, values, width=.36, color=COLORS[i], label=LABELS[i])
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=4, fontsize=9)
    ax.set(xticks=x, xticklabels=['Dice', 'IoU', 'Precision', 'Recall'], ylim=(0, 105),
           ylabel='Mean per-image score (%)', title='Held-out test results | 150 images | seed 42')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045), ncol=2, frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(alpha=.18)
    fig.text(.5, .015, 'Same split and training recipe. One run per model; no cross-seed uncertainty estimate.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(out / 'extended_test_comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, record in enumerate(records):
        history = record['history']
        epochs = [int(r['epoch']) for r in history]
        best = max(history, key=lambda r: float(r['val_dice']))
        for ax, suffix, multiplier in [(axes[0], 'dice', 100), (axes[1], 'loss', 1)]:
            for phase, style in [('train', '--'), ('val', '-')]:
                ax.plot(epochs, [multiplier * float(r[f'{phase}_{suffix}']) for r in history],
                        style, color=COLORS[i], linewidth=1.5,
                        label=f'{LABELS[i]}: {phase}')
        axes[0].scatter(int(best['epoch']), 100 * float(best['val_dice']), color=COLORS[i], zorder=5)
    axes[0].set(title='Training and validation Dice', ylabel='Dice (%)', ylim=(0, 100))
    axes[1].set(title='Training and validation loss', ylabel='BCE + Dice loss')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(out / 'extended_learning_curves.png', dpi=180)
    plt.close(fig)


def examples(root, records, out, data_root):
    import cv2
    import torch
    sys.path.insert(0, str(root))
    from models import build_model
    from datasets.kvasir_dataset import MEAN, STD
    torch.set_num_threads(2)
    # Fixed selection independent of predictions or model performance.
    rows = sorted((r for r in records[0]['split'] if r['split'] == 'val'),
                  key=lambda r: r['image'])[:4]
    size = int(records[0]['config']['data']['image_size'])
    images, masks, tensors = [], [], []
    for row in rows:
        im = cv2.imread(str(data_root / row['image'].replace('\\', '/')))
        mask = cv2.imread(str(data_root / row['mask'].replace('\\', '/')), cv2.IMREAD_GRAYSCALE)
        if im is None or mask is None:
            raise FileNotFoundError(f'Missing image/mask for {row["image"]}')
        im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (size, size))
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 127
        tensor = ((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        images.append(im); masks.append(mask)
        tensors.append(torch.from_numpy(np.ascontiguousarray(tensor)))
    inputs = torch.stack(tensors)
    predictions = []
    for record in records:
        checkpoint_path = root / 'runs' / record['run'] / 'checkpoints/best.pt'
        # These are the user's own trusted training checkpoints.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['config']['model'] == record['config']['model']
        assert int(checkpoint['config']['data']['image_size']) == size
        assert checkpoint['config']['train'] == record['config']['train']
        model = build_model(checkpoint['config']['model']).eval()
        model.load_state_dict(checkpoint['model_state'])
        with torch.inference_mode():
            predictions.append((model(inputs).sigmoid()[:, 0] >=
                                checkpoint['config']['train'].get('threshold', .5)).numpy())
        del model, checkpoint
        print('Generated predictions:', record['run'], flush=True)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.6))
    titles = ['Input image', 'Ground truth', 'Transformer U-Net', 'Attention + Transformer']
    for i, row in enumerate(rows):
        for j, image in enumerate([images[i], masks[i], predictions[0][i], predictions[1][i]]):
            axes[i, j].imshow(image, cmap='gray', vmin=0, vmax=1 if j else 255)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
    fig.suptitle('Validation examples from the selected checkpoints', fontsize=14)
    fig.text(.5, .015, 'First four validation filenames in sorted order; selected without inspecting predictions. Masks at 256 x 256.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .96))
    fig.savefig(out / 'extended_segmentation_examples.png', dpi=160)
    plt.close(fig)
    manifest = {'selection': 'first four validation filenames in sorted order',
                'images': rows, 'checkpoints': [f'runs/{r}/checkpoints/best.pt' for r in RUNS]}
    (out / 'extended_examples_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def write_readme(root, records):
    a, b = [r['metrics'] for r in records]
    rows = []
    for title, key in [('Dice', 'dice_mean'), ('IoU', 'iou_mean'),
                       ('Precision', 'precision_mean'), ('Recall', 'recall_mean')]:
        rows.append(f'| {title} | {a[key]:.2%} | {b[key]:.2%} |')
    rows += [f"| HD95, lower is better | {a['hd95_pixels_mean']:.2f} px | {b['hd95_pixels_mean']:.2f} px |",
             f"| Parameters | {a['parameters']/1e6:.2f}M | {b['parameters']/1e6:.2f}M |",
             f"| Forward time per image | {a['inference_ms_per_image']:.2f} ms | {b['inference_ms_per_image']:.2f} ms |"]
    table = '\n'.join(rows)
    historical = read_csv(root / 'results/summary.csv')
    old_rows = '\n'.join(f"| {r['experiment_name']} | {float(r['dice_mean']):.2%} | {float(r['iou_mean']):.2%} |" for r in historical)
    text = f'''# Polyp Segmentation with Attention and Transformers

A PyTorch computer-vision project for segmenting polyp regions in colonoscopy images.
I implemented and compared U-Net variants with attention gates and a Transformer bottleneck,
including reproducible training, evaluation, mask prediction, and visual comparisons.

**Attention + Transformer U-Net: {b['dice_mean']:.2%} test Dice and {b['iou_mean']:.2%} test IoU**
on 150 held-out Kvasir-SEG images.

**Stack:** Python | PyTorch | OpenCV | NumPy | SciPy | Matplotlib | pytest

## Recorded results: extended training

Both models used identical 700/150/150 train/validation/test splits, seed 42,
256 x 256 images, batch size four, and the same extended-training recipe.
Checkpoints were selected by validation Dice; test images were not used for gradient updates.

| Test metric | Transformer U-Net | Attention + Transformer U-Net |
|---|---:|---:|
{table}

The combined model led by **{100*(b['dice_mean']-a['dice_mean']):.2f} percentage points in Dice**
and **{100*(b['iou_mean']-a['iou_mean']):.2f} points in IoU** in this single-seed comparison.
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

- [Transformer-only results](results/{RUNS[0]}/metrics.json)
- [Combined-model results](results/{RUNS[1]}/metrics.json)
- [Combined-model configuration](results/{RUNS[1]}/config.json)
- [Combined-model training history](results/{RUNS[1]}/history.csv)
- [Shared split](results/{RUNS[1]}/split.csv)

## Historical experiments

The original four-model experiment used a different training recipe. Its split has not
been verified against the extended runs, so these scores must not be interpreted as a
matched comparison against the new results. Historical files are preserved unchanged.

<details>
<summary>Original architecture comparison</summary>

| Model | Historical test Dice | Historical test IoU |
|---|---:|---:|
{old_rows}

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
cp -n results/{RUNS[1]}/split.csv splits/kvasir_seed42.csv
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
'''
    readme = root / 'README.md'
    backup = root / 'runs/portfolio-readme-backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup.mkdir(parents=True)
    if readme.exists():
        shutil.copy2(readme, backup / 'README.md')
    readme.write_text(text, encoding='utf-8')
    print('README updated; previous copy:', backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--charts-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    records = load_evidence(root)
    out = root / 'docs/images'
    out.mkdir(parents=True, exist_ok=True)
    charts(records, out)
    print('Updated numeric charts from saved evidence.', flush=True)
    if args.charts_only:
        return
    examples(root, records, out, args.data_root or root / 'data/Kvasir-SEG')
    write_readme(root, records)
    print('Done. No training, metric modification, Git commit, or push was performed.')


if __name__ == '__main__':
    main()
