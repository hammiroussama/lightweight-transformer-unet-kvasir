# Lightweight Transformer-Enhanced Attention U-Net for Kvasir-SEG

Official source code for the experimental course paper:

> **A Lightweight Transformer-Enhanced Attention U-Net for Polyp Segmentation: A Controlled Experimental Study on Kvasir-SEG**

Repository: https://github.com/hammiroussama/lightweight-transformer-unet-kvasir

## Research question

Does adding (1) attention-gated skip connections and (2) a lightweight Transformer bottleneck improve U-Net polyp segmentation under an otherwise identical training protocol?

## Controlled experiments

| Configuration | Attention gates | Transformer bottleneck |
|---|---:|---:|
| U-Net | No | No |
| Attention U-Net | Yes | No |
| Transformer U-Net | No | Yes |
| Proposed model | Yes | Yes |

The same dataset split, seed, image size, augmentation, loss, optimizer, learning rate, early-stopping rule, and evaluation threshold are used for all four configurations.

## Repository structure

```text
configs/                  Four controlled YAML configurations
datasets/                 Kvasir-SEG loader and fixed split creation
models/                   U-Net, attention gates, Transformer bottleneck
utils/                    Losses, metrics, training engine, reproducibility
paper/                    Experimental LaTeX paper and generated figures
scripts/                  Synthetic smoke-test dataset generator
tests/                    Model-shape and metric tests
train.py                  Train one configuration
evaluate.py               Test metrics, HD95, parameters, and latency
predict.py                Predict and visualize one image
run_all_experiments.py    Run the complete ablation study
generate_paper_assets.py  Generate plots and LaTeX result macros
```

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

Recommended hardware: an NVIDIA GPU with at least 6 GB VRAM. The default batch size is 4 at 256 x 256 pixels. Reduce `batch_size` to 2 when memory is insufficient.

## 2. Dataset

Download Kvasir-SEG from its official research dataset page:

```text
https://datasets.simula.no/kvasir-seg/
```

The repository does **not** redistribute the dataset. Its official terms restrict use to research and education and require citation of the dataset paper.

Arrange the extracted files as:

```text
data/Kvasir-SEG/
├── images/
│   ├── cju0qkwl35piu0993l0dewei2.jpg
│   └── ...
└── masks/
    ├── cju0qkwl35piu0993l0dewei2.jpg
    └── ...
```

Image and mask basenames must match. The first training command creates the deterministic split file `splits/kvasir_seed42.csv` using 70% training, 15% validation, and 15% testing.

## 3. Validate the implementation

```bash
pytest -q
```

A small synthetic end-to-end check can be run without the real dataset:

```bash
python scripts/create_smoke_dataset.py --samples 24 --size 96
python train.py --config configs/unet.yaml --epochs 1 --device cpu
python evaluate.py --checkpoint runs/unet/checkpoints/best.pt --device cpu --warmup 1 --timed-iterations 2
```

Delete the synthetic `data/Kvasir-SEG` folder before installing the real dataset.

## 4. Run one experiment

```bash
python train.py --config configs/unet.yaml
python evaluate.py --checkpoint runs/unet/checkpoints/best.pt
```

## 5. Run the complete ablation study

```bash
python run_all_experiments.py
```

This command trains and evaluates all four configurations, then creates:

```text
results/summary.csv
results/ablation.csv
paper/results_macros.tex
paper/figures/validation_dice_curves.png
paper/figures/validation_loss_curves.png
paper/figures/test_dice_comparison.png
paper/figures/qualitative_results.png
results/qualitative_samples.csv
```

The LaTeX paper reads the generated macros directly, which reduces transcription mistakes between experimental logs and reported results.

## 6. Reproduce with explicit commands

```bash
python train.py --config configs/unet.yaml
python evaluate.py --checkpoint runs/unet/checkpoints/best.pt

python train.py --config configs/attention_unet.yaml
python evaluate.py --checkpoint runs/attention_unet/checkpoints/best.pt

python train.py --config configs/transformer_unet.yaml
python evaluate.py --checkpoint runs/transformer_unet/checkpoints/best.pt

python train.py --config configs/proposed_model.yaml
python evaluate.py --checkpoint runs/proposed_model/checkpoints/best.pt

python generate_paper_assets.py
```

## 7. Compile the paper

Copy the `paper/` directory into Overleaf or compile locally:

```bash
cd paper
pdflatex templateArxiv.tex
bibtex templateArxiv
pdflatex templateArxiv.tex
pdflatex templateArxiv.tex
```

Do not submit the paper while `paper/results_macros.tex` still contains `TBD`.

## Reproducibility safeguards

- Fixed seed and deterministic settings
- One fixed split reused by every model
- Model selection based only on validation Dice
- Test evaluation after checkpoint selection
- Automatic CSV/JSON logging
- Automatically generated paper macros
- Parameter and inference-time reporting
- Unit tests and synthetic smoke-test support

## Important limitation

Kvasir-SEG contains 1,000 annotated polyp images. This experiment is suitable for a controlled course project, but it is not sufficient to establish clinical generalization. The paper therefore treats single-dataset evaluation as a limitation and recommends future cross-dataset testing.

## Citation

See `CITATION.cff` and `paper/references.bib`.
