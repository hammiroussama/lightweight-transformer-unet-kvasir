# Apply the update and train

## What was checked

The update was prepared against local commit `725f07c`. Original measured results remain unchanged. The README now describes an engineering portfolio project without course-report or publication framing. Existing license notices and dataset attribution are preserved.

Changes include float32 Dice-loss reductions, optional gradient clipping and warmup, a 100-epoch candidate configuration, validation-only evaluation by default, saved split copies, duplicate split checks, and refusal to overwrite a training directory.

Local verification: 10 tests plus a three-epoch synthetic CPU training run, separate validation/test evaluation, prediction export, and an overwrite-protection check. Synthetic checks verify execution, not segmentation quality. No new Kvasir training has been performed and no improved benchmark score is claimed.

## Install on Windows and the server

1. Extract `kvasir-project-update.zip` into a temporary folder.
2. Copy its contents into your existing `lightweight-transformer-unet-kvasir` repository. Merge folders and replace the included files. The archive contains only changed/new files, not datasets, checkpoints, or Git history.
3. In GitHub Desktop, review the changes, commit them, and push.
4. On the server, open the Kvasir project folder and pull the changes. If GitHub access fails, transfer the same extracted files through VS Code's SSH Explorer into the Kvasir project root.
5. Activate the environment you previously used for this segmentation project. Do not assume the classification environment has every dependency.

In the server's project terminal:

```bash
pwd
ls train.py configs/transformer_unet_extended.yaml
python -m pytest -q tests
nvidia-smi
```

Use the original dataset and `splits/kvasir_seed42.csv`. If that CSV is missing, locate it from your previous server run before comparing scores.

## Train the candidate

After choosing an available GPU, set its index below (replace `N`):

```bash
GPU_INDEX=N
RUN_NAME="transformer-extended-$(date +%Y%m%d-%H%M%S)"
mkdir -p logs
CUDA_VISIBLE_DEVICES="$GPU_INDEX" nohup python -u train.py --config configs/transformer_unet_extended.yaml --output "runs/$RUN_NAME" --device cuda > "logs/$RUN_NAME.log" 2>&1 < /dev/null &
echo "PID: $!"
echo "Run: runs/$RUN_NAME"
tail -f "logs/$RUN_NAME.log"
```

Keep the log outside the output directory so the new run directory starts empty. Ctrl+C stops `tail`, while training launched with `nohup` continues. This command starts a fresh training run; it does not resume an interrupted checkpoint.

After training finishes, use the printed run path:

```bash
python evaluate.py --checkpoint "runs/$RUN_NAME/checkpoints/best.pt" --split val --device cpu
```

If you open a new terminal, set `RUN_NAME` again to the actual run-folder name. Compare validation Dice with the original Transformer U-Net checkpoint. For a stronger comparison, also retrain `configs/transformer_unet.yaml` in a new directory with the corrected loss. Choose using validation results before explicitly evaluating with `--split test`.

## CV wording based on existing evidence

Built a PyTorch polyp-segmentation pipeline comparing four U-Net variants; achieved 85.22% Dice and 77.66% IoU on a 150-image Kvasir-SEG test split, with reproducible training, automated evaluation, and prediction overlays.

Only replace those scores after a new evaluation has been completed and saved.
