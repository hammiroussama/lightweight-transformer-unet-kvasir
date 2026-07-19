# Structural and Visual Intelligence for Colonoscopy Polyp Segmentation

Official source code, experimental results, and paper materials for the Advanced Artificial Intelligence course project:

> **Structural and Visual Intelligence for Colonoscopy Polyp Segmentation:  
> A Controlled Ablation of Attention Gates and a Lightweight Transformer Bottleneck on Kvasir-SEG**

**Author:** Oussama Hammir  
**Student ID:** 25sf51073  
**Supervisor:** Luyao  
**Institution:** Harbin Institute of Technology, Shenzhen  
**Course:** Advanced Artificial Intelligence  
**Email:** hammiroussama@gmail.com  

**Repository:**  
https://github.com/hammiroussama/lightweight-transformer-unet-kvasir

## Quick links

- [Final experimental paper](paper/Oussama_Hammir_Final_Advanced_AI_Report.pdf)
- [LaTeX paper source](paper/templateArxiv.tex)
- [Main experimental results](results/summary.csv)
- [Ablation results](results/ablation.csv)
- [Statistical comparisons](results/statistical_tests.csv)
- [Qualitative predictions](paper/figures/qualitative_results.png)

## Project overview

A colorectal polyp is a tissue growth on the inner surface of the colon or rectum. During colonoscopy, accurate pixel-level segmentation can help computer-aided systems estimate a polyp's location, shape, and boundary.

This project studies two architectural modifications to a standard U-Net:

1. **Attention-gated skip connections**, which aim to suppress irrelevant encoder features before skip fusion.
2. **A lightweight Transformer bottleneck**, which aims to model long-range relationships after convolutional downsampling.

The experiment connects three topics from the Advanced Artificial Intelligence course:

- **Structural intelligence:** CNN and U-Net architecture design.
- **Visual intelligence:** medical-image segmentation and pixel-level prediction.
- **Transformer intelligence:** self-attention for global contextual modeling.

## Research question

Does adding attention-gated skip connections and a lightweight Transformer bottleneck improve U-Net polyp segmentation under an otherwise identical training and evaluation protocol?

## Controlled experiments

| Configuration | Attention gates | Transformer bottleneck |
|---|---:|---:|
| U-Net | No | No |
| Attention U-Net | Yes | No |
| Transformer U-Net | No | Yes |
| Proposed model | Yes | Yes |

All four configurations use the same dataset split, random seed, image size, augmentation, loss function, optimizer, learning rate, early-stopping rule, checkpoint-selection criterion, and evaluation threshold.

## Main experimental findings

All models were evaluated on the same held-out test set of 150 images.

| Model | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 ↓ | Parameters | Latency/image ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7635 | 0.6660 | 0.8476 | 0.7709 | 50.01 px | 7.76 M | 0.97 ms |
| Attention U-Net | 0.8231 | 0.7389 | 0.8802 | 0.8300 | 36.88 px | 7.85 M | 1.66 ms |
| **Transformer U-Net** | **0.8522** | **0.7766** | **0.8838** | 0.8647 | **29.59 px** | 11.97 M | 1.66 ms |
| Proposed model | 0.8484 | 0.7684 | 0.8695 | **0.8693** | 31.15 px | 12.06 M | 2.67 ms |

**Transformer U-Net achieved the strongest overall segmentation performance.** Compared with standard U-Net, it improved Dice by 0.0887 and reduced HD95 from 50.01 to 29.59 pixels. Attention U-Net also improved the baseline with little parameter overhead. The combined attention-plus-Transformer model achieved the highest recall, but it did not improve Dice or IoU over Transformer U-Net and required greater inference time.

The standard deviations stored in `results/summary.csv` describe variation across individual test images. They are not standard deviations across repeated training seeds because each configuration was trained once with seed 42.

## Repository structure

```text
configs/                  Four controlled YAML configurations
datasets/                 Kvasir-SEG loader and split-generation code
models/                   U-Net, attention gates, Transformer bottleneck
utils/                    Losses, metrics, training engine, reproducibility
splits/                   Fixed deterministic train/validation/test split
results/                  Metrics, ablation results, and statistical comparisons
paper/                    Final PDF, LaTeX source, and generated figures
scripts/                  Synthetic smoke-test dataset generator
tests/                    Model-shape and metric tests
train.py                  Train one configuration
evaluate.py               Evaluate Dice, IoU, HD95, parameters, and latency
predict.py                Predict and visualize one image
run_all_experiments.py    Execute the complete four-model ablation
generate_paper_assets.py  Generate plots and LaTeX result macros
```

## 1. Environment setup

Create a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recommended hardware is an NVIDIA GPU with at least 6 GB of VRAM. The default input size is 256 × 256 pixels and the default batch size is 4. Reduce the batch size to 2 when memory is insufficient.

## 2. Reported experimental environment

- GPU: NVIDIA GeForce RTX 4090
- Python: 3.10.12
- PyTorch: 2.5.1+cu121
- CUDA runtime: 12.1
- Input size: 256 × 256
- Batch size: 4
- Random seed: 42
- Split: 70% training, 15% validation, and 15% testing
- Test samples: 150

## 3. Dataset

Download Kvasir-SEG from its official research-dataset page:

```text
https://datasets.simula.no/kvasir-seg/
```

The repository does **not** redistribute the dataset. Users must follow the official research and educational terms and cite the Kvasir-SEG dataset paper.

Arrange the extracted data as follows:

```text
data/Kvasir-SEG/
├── images/
│   ├── cju0qkwl35piu0993l0dewei2.jpg
│   └── ...
└── masks/
    ├── cju0qkwl35piu0993l0dewei2.jpg
    └── ...
```

Image and mask basenames must match. The experiment uses the deterministic split stored in:

```text
splits/kvasir_seed42.csv
```

## 4. Validate the implementation

Run the unit tests:

```bash
pytest -q
```

A small synthetic end-to-end test can be run without the real dataset:

```bash
python scripts/create_smoke_dataset.py --samples 24 --size 96
python train.py --config configs/unet.yaml --epochs 1 --device cpu
python evaluate.py --checkpoint runs/unet/checkpoints/best.pt --device cpu --warmup 1 --timed-iterations 2
```

Delete the synthetic `data/Kvasir-SEG` directory before installing the real dataset.

## 5. Run one experiment

Train and evaluate standard U-Net:

```bash
python train.py --config configs/unet.yaml
python evaluate.py --checkpoint runs/unet/checkpoints/best.pt
```

Replace the configuration and checkpoint path to run another model.

## 6. Run the complete ablation study

```bash
python run_all_experiments.py
```

This command trains and evaluates all four configurations and then creates:

```text
results/summary.csv
results/ablation.csv
results/qualitative_samples.csv
paper/results_macros.tex
paper/figures/validation_dice_curves.png
paper/figures/validation_loss_curves.png
paper/figures/test_dice_comparison.png
paper/figures/qualitative_results.png
```

The LaTeX paper reads the generated result macros directly, reducing transcription errors between experimental logs and the reported paper values.

## 7. Reproduce with explicit commands

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

## 8. Experimental outputs

- [Main results](results/summary.csv)
- [Ablation analysis](results/ablation.csv)
- [Statistical tests](results/statistical_tests.csv)
- [Qualitative sample selection](results/qualitative_samples.csv)
- [Validation Dice curves](paper/figures/validation_dice_curves.png)
- [Validation loss curves](paper/figures/validation_loss_curves.png)
- [Test Dice comparison](paper/figures/test_dice_comparison.png)
- [Qualitative model comparison](paper/figures/qualitative_results.png)

## 9. Compile the paper

Copy the `paper/` directory into Overleaf, or compile locally:

```bash
cd paper
pdflatex templateArxiv.tex
bibtex templateArxiv
pdflatex templateArxiv.tex
pdflatex templateArxiv.tex
```

The final compiled report is available here:

- [Oussama Hammir — Final Advanced AI Report](paper/Oussama_Hammir_Final_Advanced_AI_Report.pdf)

## Reproducibility safeguards

- Fixed random seed and deterministic settings
- One fixed data split reused by every model
- Identical preprocessing and augmentation
- Identical optimizer, loss, learning rate, and batch size
- Checkpoint selection based only on validation Dice
- Test evaluation performed after checkpoint selection
- Automatic CSV and JSON logging
- Automatically generated LaTeX result macros
- Parameter-count and inference-time reporting
- Unit tests and synthetic smoke-test support
- Public code, results, figures, and paper source

## Limitations

Kvasir-SEG contains 1,000 annotated polyp images from one dataset. The experiment is appropriate for a controlled course project, but it is insufficient to establish clinical generalization. Each configuration was trained once with seed 42, so run-to-run uncertainty was not measured. Future work should include multiple seeds, confidence intervals, external datasets, cross-center evaluation, and clinician assessment.

## Citation

Please cite the papers listed in [`paper/references.bib`](paper/references.bib) when using this project. The dataset must also be cited according to the official Kvasir-SEG instructions.
