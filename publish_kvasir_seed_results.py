"""Refresh portfolio documentation from all six saved test evaluations. No training or Git actions."""
import csv
import json
import statistics as stats
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY = 'seed-study-20261122-190743-088986'
MODELS = ['transformer_unet', 'proposed']
LABELS = ['Transformer U-Net', 'Attention + Transformer U-Net']
SEEDS = [42, 123, 3407]


def main():
    root = Path.cwd()
    study = root / 'results' / STUDY
    summary = json.loads((study / 'test/summary.json').read_text())
    assert summary['completed_runs'] == 6 and summary['split'] == 'test'
    assert {(r['model'], r['seed']) for r in summary['runs']} == {(m, s) for m in MODELS for s in SEEDS}
    records = {}
    for model in MODELS:
        for seed in SEEDS:
            p = study / 'test' / f'{model}-seed{seed}' / 'metrics.json'
            m = json.loads(p.read_text())
            assert m['split'] == 'test' and m['seed'] == seed and m['test_samples'] == 150
            row = next(r for r in summary['runs'] if r['model'] == model and r['seed'] == seed)
            for key in ['dice', 'iou']:
                assert abs(row[key] - m[key + '_mean']) < 1e-10
            records[model, seed] = m
    def values(model, key):
        return [records[model, seed][key] for seed in SEEDS]
    def mean(model, key):
        return stats.mean(values(model, key))
    images = root / 'docs/images'
    images.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, title in zip(axes, ['dice_mean', 'iou_mean'], ['Test Dice', 'Test IoU']):
        for i, model in enumerate(MODELS):
            v = [100*x for x in values(model, key)]
            ax.errorbar(i, stats.mean(v), yerr=stats.stdev(v), fmt='o', capsize=7,
                        color=['#4b72b8', '#138575'][i], markersize=9)
        for seed in SEEDS:
            ax.plot([0, 1], [100*records[m, seed][key] for m in MODELS],
                    '.--', alpha=.55, label=f'Seed {seed}')
        ax.set(xticks=[0, 1], xticklabels=['Transformer', 'Attention + Transformer'],
               ylabel='Score (%)', title=title, xlim=(-.4, 1.4))
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same 150 test images | three paired training seeds')
    fig.text(.5, .025, 'Large points: mean; error bars: sample SD across seeds (not confidence intervals).', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .94])
    fig.savefig(images / 'seed_study_test_comparison.png', dpi=170)
    plt.close(fig)
    with (study / 'test/comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'seed', 'dice', 'iou', 'precision', 'recall', 'hd95_pixels', 'forward_ms_per_image'])
        for (model, seed), m in records.items():
            writer.writerow([model, seed] + [m[k] for k in ['dice_mean', 'iou_mean', 'precision_mean', 'recall_mean', 'hd95_pixels_mean', 'inference_ms_per_image']])
    table = []
    for label, key, scale, unit in [('Dice', 'dice_mean', 100, '%'), ('IoU', 'iou_mean', 100, '%'),
                                  ('Precision', 'precision_mean', 100, '%'), ('Recall', 'recall_mean', 100, '%'),
                                  ('HD95 (lower is better)', 'hd95_pixels_mean', 1, ' px'),
                                  ('Forward time per image', 'inference_ms_per_image', 1, ' ms')]:
        cells = [f'{scale*mean(m,key):.2f}{unit} ± {scale*stats.stdev(values(m,key)):.2f}' for m in MODELS]
        table.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    paired = '\n'.join(f"| {s} | {records[MODELS[0],s]['dice_mean']:.2%} | {records[MODELS[1],s]['dice_mean']:.2%} | {100*(records[MODELS[1],s]['dice_mean']-records[MODELS[0],s]['dice_mean']):+.2f} |" for s in SEEDS)
    delta = 100*(mean('proposed','dice_mean')-mean('transformer_unet','dice_mean'))
    readme = root / 'README.md'
    old = readme.read_text(encoding='utf-8')
    backup = root / 'runs/portfolio-readme-backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup.mkdir(parents=True)
    (backup / 'README.md').write_text(old, encoding='utf-8')
    historical = root / 'README_previous_results.md'
    if not historical.exists():
        historical.write_text(old, encoding='utf-8')
    marker = '## Setup and dataset'
    assert marker in old, 'Expected setup section missing; README was not replaced.'
    tail = marker + old.split(marker, 1)[1]
    tail = tail.replace('repeat both models across seeds, and include an untouched external evaluation dataset.',
                        'include an untouched external evaluation dataset. The three-seed study above now measures variation across training seeds.')
    text = f'''# Polyp Segmentation with Attention and Transformers

A PyTorch project that segments polyp regions in colonoscopy images using U-Net variants,
attention gates, and a Transformer bottleneck. Includes reproducible training, mask prediction,
visual overlays, and a paired comparison across three training seeds.

**Attention + Transformer U-Net: {mean('proposed','dice_mean'):.2%} mean test Dice
and {mean('proposed','iou_mean'):.2%} mean test IoU across three seeds.**

**Stack:** Python | PyTorch | OpenCV | NumPy | SciPy | Matplotlib | pytest

## Recorded results: three-seed study

Both architectures used the same 700/150/150 train/validation/test split, seeds 42, 123,
and 3407, and the matched extended-training recipe. Each checkpoint was selected by
validation Dice, then all six checkpoints were evaluated on the same 150 test images.
These are three training repetitions on one split, not three independent test datasets.

| Test metric | Transformer U-Net | Attention + Transformer U-Net |
|---|---:|---:|
{chr(10).join(table)}

Values are mean ± sample standard deviation across training seeds, not confidence intervals.
HD95 is in pixels at 256×256 resolution. Forward timing is per image in batches of four on
an RTX 4090, excluding preprocessing; it is not end-to-end application latency.

The combined model improved mean Dice by **{delta:.2f} percentage points** and won on
two of three seeds. It showed lower observed seed variability, with greater inference cost.
Three seeds do not establish statistical significance or universal superiority.

| Seed | Transformer Dice | Attention + Transformer Dice | Difference (points) |
|---|---:|---:|---:|
{paired}

![Three-seed test comparison](docs/images/seed_study_test_comparison.png)

## Evidence and reproducibility

- [Study plan and source fingerprints](results/{STUDY}/plan.json)
- [Frozen configurations](results/{STUDY}/configs)
- [Shared split](results/{STUDY}/split.csv)
- [Validation summary](results/{STUDY}/summary.json)
- [Test summary and checkpoint hashes](results/{STUDY}/test/summary.json)
- [Per-run test metrics and per-image scores](results/{STUDY}/test)
- [Comparison CSV](results/{STUDY}/test/comparison.csv)

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

'''
    readme.write_text(text + tail, encoding='utf-8')
    print('Updated README.md, historical README, three-seed chart, and comparison.csv.')
    print('Original README backup:', backup)
    print('No training, checkpoint modification, commit, or push performed.')


if __name__ == '__main__':
    main()
