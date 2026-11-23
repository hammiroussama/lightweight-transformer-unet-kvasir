# Experiment protocol

This document describes the published three-seed study. See the [README](../README.md) for results, setup, and historical comparisons.

## Comparison and data

The current comparison measures the effect of attention-gated skip connections within Transformer U-Net. Both architectures are trained from scratch. Their configuration names are `transformer_unet` and `proposed`.

- Dataset: 1,000 Kvasir-SEG image/mask pairs.
- Fixed assignments: 700 training, 150 validation, 150 test images.
- Training seeds: 42, 123, 3407; the split does not change with seed.
- Input: 256 × 256 RGB; bilinear image resizing and nearest-neighbor mask resizing.
- Normalization: mean [0.485, 0.456, 0.406], standard deviation [0.229, 0.224, 0.225].
- Training augmentation: horizontal/vertical flips, rotation, and brightness/contrast changes; validation and test have no random augmentation.

The [saved split](../results/seed-study-20261122-190743-088986/split.csv) defines the experiment. Duplicate image or mask paths are rejected by the loader. Patient-level separation has not been established.

## Matched training recipe

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate / minimum | 1e-4 / 1e-6 |
| Weight decay | 1e-4 |
| Loss | 0.5 binary cross-entropy + 0.5 Dice loss |
| Batch size | 4 |
| Maximum epochs | 100 |
| Warm-up | 5 epochs |
| Schedule | Linear warm-up followed by cosine decay |
| Gradient clipping | Norm 1.0 |
| Early-stopping patience | 25 epochs without validation Dice improvement |
| Checkpoint selection | Highest validation Dice |
| Mask threshold | 0.5 |

CUDA training and training-time validation use automatic mixed precision; Dice reductions use float32. Standalone evaluation uses full precision, so reevaluated validation scores can differ slightly from the checkpoint-selection scores in the history. Each run saves best/last checkpoints, a split copy, and an epoch history. Non-empty output directories are rejected.

## Execution policy and provenance

The runner sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enables strict deterministic algorithms, disables Flash and memory-efficient attention, and enables math attention. It sets Python hash seeds and limits CPU threads. This is not a guarantee of bitwise equality across platforms or PyTorch releases.

The [original plan](../results/seed-study-20261122-190743-088986/plan.json) freezes six jobs, configuration/split hashes, source hashes, and the preparation revision. Its validation-only policy describes the initial training stage. Later test evaluation is recorded separately under [test/](../results/seed-study-20261122-190743-088986/test), including checkpoint SHA-256 hashes. The plan is preserved rather than retroactively edited.

The absolute runner path in that plan is historical provenance, not a path required on another machine. Prepare a new study to create local fingerprints and output directories. Moving or editing source after preparation invalidates the runner's checks. Saved configurations are JSON, also accepted by the YAML loader.

Recorded environment: Python 3.11.17, PyTorch 2.5.1+cu121, CUDA runtime 12.1, RTX 4090. Dependency ranges are not a full environment lock.

## Evaluation and aggregation

1. Select one checkpoint per run using validation Dice only.
2. Freeze settings and evaluate all six checkpoints on the same 150 test images.
3. Compute Dice, IoU, precision, recall, and HD95 per image.
4. Average across images to obtain each run's score.
5. Report the mean and sample standard deviation (`ddof=1`) of the three run scores per model.

Per-image standard deviations in `metrics.json` describe image difficulty. Across-seed standard deviations in `test/summary.json` describe training variation. Neither is a confidence interval. The repetitions do not create 450 independent test images. No significance claim is made for this comparison.

HD95 uses resized-image pixels; when only one mask is empty, the implementation returns the image diagonal. Timing uses a batch of four, 10 warm-ups, 50 measured passes, and CUDA synchronization. It excludes loading, preprocessing, export, and UI overhead. Test evaluation uses full precision; training AMP does not imply AMP inference timing.

Validation outputs go to each run's `validation/` directory; test outputs go to its root. Repeating evaluation overwrites that run's metrics and per-image CSV. Published exports preserve the recorded evaluations. Checkpoint hashes cannot be verified without the private checkpoint files.

## Deterministic evaluation example

The ordinary evaluation CLI does not itself select the math attention backend. For the study settings on Linux, use the following from the repository root with your trained checkpoint, seed, and available GPU index:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=42 python - <<'PY'
import runpy
import sys
import torch
import utils.seed

torch.set_num_threads(2)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
original = utils.seed.set_seed
def strict_seed(seed, deterministic=True):
    original(seed, deterministic)
    torch.use_deterministic_algorithms(True, warn_only=False)
utils.seed.set_seed = strict_seed
sys.argv = ['evaluate.py', '--checkpoint', 'runs/combined_new/checkpoints/best.pt',
            '--split', 'test', '--device', 'cuda']
runpy.run_path('evaluate.py', run_name='__main__')
PY
```

During development use `--split val`; only use test after configuration selection is complete.

## Test-set history

This test split was already inspected in earlier extended experiments. The study measures training variation but does not restore a fully blinded test set. Future tuning must use validation data, with an untouched external dataset for broader evaluation.

The original four-architecture results used an earlier recipe and loss implementation; identical splits against the newer experiments have not been verified. The single-seed extended runs shared assignments but preceded strict deterministic execution. Keep these stages separate. Historical statistical tests do not establish significance for the three-seed results.

## Evidence checks and figures

```bash
python scripts/check_results.py
python scripts/plot_seed_study.py
```

Checks cover the six-run roster, split membership, per-image aggregates, summaries, frozen inputs, and comparison CSV without a GPU or dataset. Figure generation reads that evidence. Neither command modifies measurements or rewrites the README. Qualitative examples remain attributed to the earlier checkpoints in their [manifest](images/extended_examples_manifest.json).
