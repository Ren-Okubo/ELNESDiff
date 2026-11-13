# Generative inverse estimation of 3D atomic coordination from near-edge spectra via equivariant diffusion models

This diffusion model generates three-dimensional local atomic environments conditioned on electron energy-loss near-edge structure (ELNES) spectra. This repository contains the training and generation code for Si–O environments.

## Setup

The current training code requires an NVIDIA GPU. Install the pinned dependencies with Conda:

```bash
conda env create -f environment.yml
conda activate ELNESdiff
```

Training and generation use [Weights & Biases](https://wandb.ai/) to record runs and locate model checkpoints. Log in with `wandb login` before running the commands below.

## Data

The examples below use `./dataset/42`, which contains separate `train`, `valid`, and `test` `.pt` files. Each file contains PyTorch Geometric graphs with atomic types (`x`), positions (`pos`), and raw ELNES spectra (`spectrum_raw`). You can pass another dataset with `--dataset_path`.

## Usage

Run commands from the repository root. Model settings are in `parameters.yaml`.

### Train and generate

```bash
python main.py \
  --project_name Si_O_ELNES \
  --run_name my_run \
  --dataset_path ./dataset/42 \
  --mode train_and_generate \
  --scaling area
```

The model checkpoint and generated graphs are saved in the W&B run directory.

### Generate with a trained model

```bash
PYTHONPATH=module python generate_with_learned_model.py \
  --learned_model ENTITY/PROJECT/RUN_ID \
  --project_name OUTPUT_PROJECT \
  --run_name OUTPUT_RUN \
  --dataset_path ./dataset/42/Si_O_ELNES_test.pt \
  --scaling area
```

This command reads the training run's configuration from W&B and expects its checkpoint path to be accessible locally. No pretrained checkpoint is included in this repository.

## Citation

R. Okubo et al., “Generative inverse estimation of 3D atomic coordination from near-edge spectra via equivariant diffusion models,” *npj Computational Materials* (2026). https://doi.org/10.1038/s41524-026-02384-4