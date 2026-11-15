# Experimental protocol

## Hypothesis

Adding attention-gated skip connections and a lightweight Transformer bottleneck to U-Net may improve polyp segmentation by combining boundary-sensitive convolutional features with longer-range context.

## Controlled variables

All four configurations use the same fixed split, image size, optimizer, learning rate, loss, augmentation, early-stopping rule, threshold, and random seed. Only the two architectural components change.

| Experiment | Attention gates | Transformer bottleneck |
|---|---:|---:|
| U-Net | No | No |
| Attention U-Net | Yes | No |
| Transformer U-Net | No | Yes |
| Proposed | Yes | Yes |

## Primary endpoint

Mean test Dice coefficient from the checkpoint with the best validation Dice.

## Secondary endpoints

IoU, precision, recall, HD95 in resized-image pixels, parameter count, and inference time per image.

## Reproducibility rules

- Seed: 42
- Fixed CSV split: `splits/kvasir_seed42.csv`
- Test set is used once after model selection on validation data.
- All reported numbers must come from saved evaluation outputs.
- Do not manually alter metrics.
- Record GPU, CUDA, PyTorch version, and evaluation date alongside the results.

## New candidate runs

The extended-training configuration is separate from the original controlled architecture comparison. It changes the training budget, scheduler, clipping, and stopping patience. Compare validation Dice against a fresh baseline using the same corrected loss and original split. Keep architecture, image size, batch size, and seed fixed for this first comparison. Do not interpret the outcome as an isolated effect of one hyperparameter.

The updated Dice loss uses float32 reductions; historical scores remain historical measurements and are not silently replaced. Run `evaluate.py --split val` during development. Use `--split test` only once the candidate is selected. Repeat promising configurations with additional seeds before claiming a reliable improvement.

Each new run stores its split CSV. For historical runs without this copy, evaluation requires the original configured CSV and will not generate a replacement. Keep the same split across training seeds.
