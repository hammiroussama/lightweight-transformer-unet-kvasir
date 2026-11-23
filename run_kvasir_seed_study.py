"""Run a frozen, paired three-seed validation study, sequentially on one GPU.

Prepare: python run_kvasir_seed_study.py --prepare
Run:     python run_kvasir_seed_study.py --study results/seed-study-TIMESTAMP --gpu 1
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime

SEEDS = [42, 123, 3407]
REFERENCES = [
    'transformer-extended-20261115-164954',
    'attention-transformer-extended-20261116-200334',
]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def fingerprint():
    files = [Path('train.py'), Path('evaluate.py'), Path(__file__).resolve()]
    for directory in ['models', 'datasets', 'utils']:
        files.extend(sorted(Path(directory).rglob('*.py')))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare(data_root):
    configs = [json.loads((Path('results') / r / 'config.json').read_text()) for r in REFERENCES]
    assert configs[0]['train'] == configs[1]['train'], 'Reference training recipes differ'
    assert configs[0]['model']['name'] == 'transformer_unet'
    assert configs[1]['model']['name'] == 'proposed'
    def split_rows(run):
        with (Path('results') / run / 'split.csv').open(newline='') as f:
            return sorted((r['image'].replace('\\', '/'), r['mask'].replace('\\', '/'), r['split'])
                          for r in csv.DictReader(f))
    rows = split_rows(REFERENCES[0])
    assert rows == split_rows(REFERENCES[1]), 'Reference splits differ'
    assert len(rows) == 1000 and len({r[0] for r in rows}) == 1000
    assert {s: sum(r[2] == s for r in rows) for s in ['train', 'val', 'test']} == {'train':700, 'val':150, 'test':150}
    for image, mask, _ in rows:
        for name in [image, mask]:
            if not (data_root / name).is_file():
                raise FileNotFoundError(data_root / name)
    study = Path('results') / ('seed-study-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    study.mkdir(parents=True, exist_ok=False)
    with (study / 'split.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['image', 'mask', 'split']); writer.writerows(rows)
    jobs = []
    for seed in SEEDS:
        for reference in configs:
            cfg = json.loads(json.dumps(reference))
            name = cfg['model']['name']
            cfg['seed'] = seed
            cfg['experiment_name'] = f'{name}: repeated-seed validation, seed {seed}'
            cfg['data']['root'] = str(data_root)
            cfg['data']['split_csv'] = str(study / 'split.csv')
            cfg['output_dir'] = f'runs/{study.name}/{name}-seed{seed}'
            config = study / 'configs' / f'{name}-seed{seed}.json'
            save(config, cfg)  # JSON is also valid YAML for the existing loader.
            jobs.append(dict(model=name, seed=seed, config=str(config), output=cfg['output_dir']))
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    immutable = [study / 'split.csv'] + [Path(j['config']) for j in jobs]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in immutable}
    save(study / 'plan.json', dict(seeds=SEEDS, jobs=jobs, evaluation_split='val',
         source_hashes=fingerprint(), input_hashes=hashes, git_revision=revision.stdout.strip(),
         policy='All six runs reported; no best-seed selection. Validation only; no test evaluation.',
         determinism='CUBLAS_WORKSPACE_CONFIG=:4096:8; strict deterministic algorithms; math attention backend'))
    print('Prepared six runs. Nothing trained yet.')
    print('STUDY:', study)
    print('After checking GPU availability, run:')
    print(f'python -u {Path(__file__).name} --study {study} --gpu N')


def worker(stage, config, output):
    # Parent sets CUDA/CUBLAS environment before this interpreter starts.
    import torch
    import utils
    import utils.seed
    torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    original = utils.seed.set_seed
    def strict_seed(seed, deterministic=True):
        original(seed, deterministic)
        torch.use_deterministic_algorithms(True, warn_only=False)
    utils.set_seed = strict_seed
    utils.seed.set_seed = strict_seed
    import runpy
    if stage == 'train':
        sys.argv = ['train.py', '--config', config, '--output', output, '--device', 'cuda']
        runpy.run_path('train.py', run_name='__main__')
    else:
        sys.argv = ['evaluate.py', '--checkpoint', str(Path(output)/'checkpoints/best.pt'), '--split', 'val', '--device', 'cuda']
        runpy.run_path('evaluate.py', run_name='__main__')


def summarize(study, plan):
    rows = []
    for job in plan['jobs']:
        result = study / f"{job['model']}-seed{job['seed']}.json"
        if result.exists():
            metrics = json.loads(result.read_text())
            rows.append(dict(model=job['model'], seed=job['seed'], dice=metrics['dice_mean'], iou=metrics['iou_mean']))
    report = {'split':'validation', 'completed_runs':len(rows), 'planned_runs':6, 'runs':rows}
    for name in ['transformer_unet', 'proposed']:
        values = [r['dice'] for r in rows if r['model'] == name]
        if values:
            report[name] = dict(n=len(values), dice_mean=statistics.mean(values),
                                dice_seed_std=statistics.stdev(values) if len(values)>1 else None)
    pairs=[]
    for seed in SEEDS:
        match={r['model']:r['dice'] for r in rows if r['seed']==seed}
        if len(match)==2:
            pairs.append({'seed':seed,'combined_minus_transformer_dice_points':100*(match['proposed']-match['transformer_unet'])})
    report['paired_differences']=pairs
    save(study / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


def run(study, gpu):
    plan=json.loads((study/'plan.json').read_text())
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    logdir=Path('logs')/study.name; logdir.mkdir(parents=True,exist_ok=True)
    for job in plan['jobs']:
        assert fingerprint()==plan['source_hashes'], 'Source code changed since preparation; do not mix implementations'
        for name,digest in plan['input_hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, f'Prepared input changed: {name}'
        tag=f"{job['model']}-seed{job['seed']}"
        result=study/f'{tag}.json'
        if result.exists():
            print('Already completed:',tag,flush=True);continue
        output=Path(job['output'])
        trained=study/f'{tag}.trained.json'
        env['PYTHONHASHSEED']=str(job['seed'])
        with (logdir/f'{tag}.log').open('a') as log:
            for stage in ['train','evaluate']:
                if stage=='train' and trained.exists():continue
                if stage=='train' and output.exists() and any(output.iterdir()):
                    raise RuntimeError(f'Incomplete run preserved at {output}. Do not overwrite it; inspect its log.')
                print(f'{stage.upper()}: {tag}; log: {logdir/tag}.log',flush=True)
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',stage,
                    '--config',job['config'],'--output',str(output)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if stage=='train':save(trained, {'training_process_exited_successfully':True})
        metrics=json.loads((output/'validation/metrics.json').read_text())
        assert metrics['split']=='val' and metrics['seed']==job['seed']
        save(result,metrics)
        # Preserve the learning history as text evidence, without checkpoints.
        (study/f'{tag}.history.csv').write_bytes((output/'history.csv').read_bytes().replace(b'\r\n',b'\n'))
        summarize(study,plan)
    print('Completed. All six runs summarized; test set was not evaluated.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--study',type=Path)
    p.add_argument('--gpu',type=int)
    p.add_argument('--data-root',type=Path,default=Path('data/Kvasir-SEG'))
    p.add_argument('--worker',choices=['train','evaluate'],help=argparse.SUPPRESS)
    p.add_argument('--config',help=argparse.SUPPRESS)
    p.add_argument('--output',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:worker(args.worker,args.config,args.output)
    elif args.prepare:prepare(args.data_root)
    elif args.study is not None and args.gpu is not None:run(args.study,args.gpu)
    else:p.error('Use --prepare, or --study PATH --gpu INDEX')
