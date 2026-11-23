"""Audit the published study using only the standard library. No training or writes."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/seed-study-20261122-190743-088986'
MODELS = ['transformer_unet', 'proposed']
SEEDS = [42, 123, 3407]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-8),
            f'{label}: {actual} != {expected}')


def check():
    plan = read_json(STUDY / 'plan.json')
    expected = {(m, s) for m in MODELS for s in SEEDS}
    require(len(plan['jobs']) == 6 and {(j['model'], j['seed']) for j in plan['jobs']} == expected,
            'Plan must contain exactly the six paired runs')
    # Git on Windows may check out CRLF text; accept only exact bytes or LF-equivalent bytes.
    for name, digest in plan['input_hashes'].items():
        data = (ROOT / name).read_bytes()
        require(digest in [hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()],
                f'Frozen input changed: {name}')
    split = read_csv(STUDY / 'split.csv')
    require(len(split) == 1000, 'Expected 1,000 split rows')
    for field in ['image', 'mask']:
        require(len({r[field].replace('\\', '/') for r in split}) == 1000, f'Duplicate {field}')
    require({s: sum(r['split'] == s for r in split) for s in ['train', 'val', 'test']} ==
            {'train': 700, 'val': 150, 'test': 150}, 'Unexpected split sizes')
    test_names = {Path(r['image'].replace('\\', '/')).name for r in split if r['split'] == 'test'}
    records = {}
    reference_recipe = None
    for job in plan['jobs']:
        model, seed = job['model'], job['seed']
        tag = f'{model}-seed{seed}'
        config = read_json(ROOT / job['config'])
        require(config['model']['name'] == model and config['seed'] == seed, f'{tag}: wrong configuration')
        recipe = (config['train'], config['data'])
        if reference_recipe is None:
            reference_recipe = recipe
        require(recipe == reference_recipe, f'{tag}: unmatched training/data recipe')
        directory = STUDY / 'test' / tag
        metrics = read_json(directory / 'metrics.json')
        require(metrics['split'] == 'test' and metrics['seed'] == seed and metrics['test_samples'] == 150,
                f'{tag}: wrong evaluation metadata')
        rows = read_csv(directory / 'per_image_metrics.csv')
        require(len(rows) == 150 and {r['name'] for r in rows} == test_names, f'{tag}: test images differ')
        for key in ['dice', 'iou', 'precision', 'recall', 'hd95_pixels']:
            vals = [float(r[key]) for r in rows]
            require(all(math.isfinite(v) and v >= 0 and (key == 'hd95_pixels' or v <= 1.000001) for v in vals),
                    f'{tag}: invalid {key}')
            close(metrics[key + '_mean'], statistics.mean(vals), f'{tag} {key} mean')
            close(metrics[key + '_std'], statistics.stdev(vals), f'{tag} {key} SD')
        history = read_csv(STUDY / f'{tag}.history.csv')
        validation = read_json(STUDY / f'{tag}.json')
        require(validation['split'] == 'val' and validation['seed'] == seed, f'{tag}: validation metadata')
        # Training-time validation uses AMP; standalone evaluation uses full precision.
        # Their thresholded scores can differ, so equality would be an invalid check.
        require(history and all(math.isfinite(float(r['val_dice'])) and 0 <= float(r['val_dice']) <= 1
                                for r in history), f'{tag}: invalid validation history')
        records[model, seed] = metrics
    for stage, path in [('test', STUDY / 'test/summary.json'), ('validation', STUDY / 'summary.json')]:
        summary = read_json(path)
        require(summary['completed_runs'] == 6 and summary['split'] == stage, f'{stage}: incomplete summary')
        require(len(summary['runs']) == 6 and {(r['model'], r['seed']) for r in summary['runs']} == expected,
                f'{stage}: incorrect run roster')
        for r in summary['runs']:
            m = records[r['model'], r['seed']] if stage == 'test' else read_json(STUDY / f"{r['model']}-seed{r['seed']}.json")
            for key in ['dice', 'iou']:
                close(r[key], m[key + '_mean'], f'{stage} summary {key}')
        for model in MODELS:
            for key in (['dice', 'iou'] if stage == 'test' else ['dice']):
                vals = [r[key] for r in summary['runs'] if r['model'] == model]
                close(summary[model][key + '_mean'], statistics.mean(vals), f'{model} {stage} mean')
                close(summary[model][key + '_seed_std'], statistics.stdev(vals), f'{model} {stage} seed SD')
    comparison = read_csv(STUDY / 'test/comparison.csv')
    require(len(comparison) == 6 and {(r['model'], int(r['seed'])) for r in comparison} == expected,
            'Comparison CSV roster differs')
    keys = {'dice': 'dice_mean', 'iou': 'iou_mean', 'precision': 'precision_mean', 'recall': 'recall_mean',
            'hd95_pixels': 'hd95_pixels_mean', 'forward_ms_per_image': 'inference_ms_per_image'}
    for row in comparison:
        m = records[row['model'], int(row['seed'])]
        for column, metric in keys.items():
            close(row[column], m[metric], f'comparison {column}')
    print('PASS: six paired runs; frozen inputs; shared test images; per-image metrics; validation/test summaries; comparison CSV.')
    return records


if __name__ == '__main__':
    check()
