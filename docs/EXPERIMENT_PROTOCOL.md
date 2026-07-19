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
- All reported numbers must be copied from `results/summary.csv` or generated LaTeX macros.
- Do not manually alter metrics.
- Record GPU, CUDA, PyTorch version, and date of execution in the paper.
