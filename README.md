# Audio-Visual Speech Separation with RTFSNet and ConvTasNet

<p align="center">
  <a href="#about">About</a> •
  <a href="#models">Models</a> •
  <a href="#installation">Installation</a> •
  <a href="#data-and-assets">Data & Assets</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#hydra-configs">Hydra Configs</a>
</p>

---

## About

This repository contains experiments on **Audio-Visual Speech Separation (AVSS)**.

We use:

- **ConvTasNet** as a *baseline* purely audio model.
- **RTFSNet** as the **main audio-visual separation model**.

RTFSNet consumes **video embeddings** of the speaker’s mouth region.
To obtain them, we use a pretrained lipreading video encoder from:

> https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

Before training RTFSNet, we **precompute all video embeddings and store them on disk**.
This avoids recomputing the encoder on every epoch and significantly speeds up training and inference.

The codebase is built on top of the PyTorch project template (Hydra configs, experiment tracking, etc.), but adapted specifically for AVSS experiments.

---

## Models

### ConvTasNet (baseline)

- Audio-only separation model.
- Used as a strong baseline for comparison with AVSS approaches.
- Pretrained weights can be downloaded via `scripts/download_convtasnet.sh`.
- Training / inference / evaluation are controlled via ConvTasNet-specific Hydra configs (see below).

### RTFSNet (main model)

- Audio-Visual speech separation network.
- Takes as input:
  - mixture waveform,
  - video embeddings of speaker 1 and speaker 2.
- Uses a **pretrained lipreading video encoder** to extract embeddings from mouth crops.
- For efficiency, all video embeddings are **precomputed and cached to disk** before RTFSNet training.

Pretrained weights of the best RTFSNet model can be downloaded via `scripts/download_rtfsnet.sh`.

---

## Installation

### 0. (Optional) Create a new environment

Using `conda`:

```bash
conda create -n avss_env python=PYTHON_VERSION
conda activate avss_env
```

Using `venv` (optionally with `pyenv`):

```bash
python3 -m venv avss_env
source avss_env/bin/activate
```

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Install pre-commit hooks

```bash
pre-commit install
```

This enables automatic formatting and basic checks before each commit.

---

## Data and Assets

This repository provides helper scripts in the `scripts/` folder to download all required assets.

### Microphone Impulse Responses

```bash
bash scripts/download_mirs.sh OUT_DIR
```

* Downloads **microphone impulse responses** (IRs).
* Used to simulate realistic room/microphone acoustics during training.

### Noise Datasets

```bash
bash scripts/download_noises.sh OUT_DIR
```

* Downloads **noise recordings** (e.g. DNS Challenge noises).
* Used to generate noisy mixtures for training and evaluation.

### Pretrained Models

```bash
bash scripts/download_convtasnet.sh OUT_DIR
```

* Downloads pretrained **ConvTasNet** baseline weights.

```bash
bash scripts/download_rtfsnet.sh OUT_DIR
```

* Downloads weights of the **best trained RTFSNet** model.

```bash
bash scripts/download_video_encoder.sh OUT_DIR
```

* Downloads the **pretrained lipreading video encoder**
  (originally from `Lipreading_using_Temporal_Convolutional_Networks`).
* These weights are used to compute **video embeddings** that are later passed to RTFSNet.
* In typical workflow we:

  1. Run a preprocessing script that computes embeddings for all videos in the dataset.
  2. Save them to disk.
  3. Use these cached embeddings during RTFSNet training and inference.

---

## Training

All training scripts are driven by **Hydra configs** under `src/configs/`.

General pattern:

```bash
python train.py -cn=CONFIG_NAME HYDRA_OVERRIDES
```

Where:

* `CONFIG_NAME` is one of the training configs listed in [Hydra Configs](#hydra-configs).
* `HYDRA_OVERRIDES` are optional command-line overrides, e.g. `trainer.max_epochs=50`.

### Training scripts

`train.py`
Main training entrypoint. It:

* Instantiates the dataset, model, loss functions, optimizers, and schedulers from Hydra config.
* Runs the full training loop.
* Logs metrics and saves checkpoints according to the configuration.

---

## Inference

Inference is also launched via Hydra:

```bash
python inference.py -cn=CONFIG_NAME HYDRA_OVERRIDES
```

`inference.py` supports:

* Running separation on a dataset split (`train` / `val` / `test` / custom partition).
* Saving separated sources (predictions) to disk (e.g. into `predictions/`).
* Optionally computing metrics during inference if ground truth is available.

Key script:

`inference.py`

* Loads a pretrained model (ConvTasNet or RTFSNet, depending on config).
* Iterates over the dataset and writes separated signals to disk via `torchaudio.save`.
* Uses `inferencer.save_path` (Hydra config) as the root folder for predictions.

Before running inference, it is often convenient to clear previous predictions, e.g.:

```python
import os, shutil
if os.path.exists("predictions"):
    shutil.rmtree("predictions")
```

---

## Metrics

If ground-truth sources are available and you want to evaluate saved predictions **after** running inference, use:

```bash
python calc_metrics.py -cn=CONFIG_NAME HYDRA_OVERRIDES
```

`calc_metrics.py`:

* Loads predictions and corresponding ground truth from disk.
* Computes separation metrics (e.g. SI-SNR, SI-SDR, PIT-based metrics) depending on the config.
* Is especially useful if you:

  * already have predictions saved from a previous run, and
  * later obtained ground truth sources for them.

---

## Hydra Configs

All configs are stored in `src/configs/`. Below is a brief description of each key config file.

### Inference configs

* **`inf_convtasnet.yaml`**
  Inference configuration for the **ConvTasNet** baseline.

  * Sets up ConvTasNet model, dataset, and inference parameters.
  * Use with `inference.py` to save ConvTasNet predictions.

* **`inf_rtfsnet.yaml`**
  Inference configuration for **RTFSNet**.

  * Uses the RTFSNet model with precomputed **video embeddings**.
  * Includes paths to the dataset and the video embeddings directory.
  * Example:

    ```bash
    python inference.py -cn=inf_rtfsnet.yaml inferencer.from_pretrained=weights/rtfsnet.pth
    ```

### Metrics configs

* **`calc_metrics_convtasnet.yaml`**
  Configuration for **metrics computation** for ConvTasNet predictions.

  * Points to folders with ground truth (`gt_dir`) and ConvTasNet predictions (`pred_dir`).
  * Defines which BSS metrics to compute (e.g. SI-SNR, SDR, PIT).

* **`calc_metrics_rtfsnet.yaml`**
  Configuration for **metrics computation** for RTFSNet predictions.

  * Similar to `calc_metrics_convtasnet.yaml`, but assumes RTFSNet output layout.

### Training configs

* **`train_convtasnet.yaml`**
  Training configuration for the **baseline ConvTasNet** model.

  * Defines dataset, optimizer, learning rate schedule, and loss (typically SI-SNR-based).
  * This is the config used to train our baseline.

* **`train_rtfsnet.yaml`**
  **Main training configuration for RTFSNet.**

  * Uses precomputed video embeddings as input.
  * Trains RTFSNet with the primary loss (e.g. SI-SNR).
  * This is the default config for training the main AVSS model.

* **`train_rtfsnet_multiloss.yaml`**
  RTFSNet training with **additional spectral loss**.

  * Combines SI-SNR loss in the time domain with a spectrogram-based loss (e.g. L1/L2 on magnitude or log-magnitude STFT).
  * Used to study whether spectral regularization improves perceptual quality or robustness.

* **`train_rtfsnet_pit.yaml`**
  RTFSNet training with **PIT (Permutation Invariant Training)** in a BSS setting.

  * The objective is invariant to the ordering of separated speakers.
  * Useful when the speaker identities are not fixed or when training on mixtures without explicit speaker labeling.

* **`train_rtfsnet_with_augs.yaml`**
  RTFSNet training with **data augmentations**.

  * Makes use of noise injection, impulse responses, and other augmentations (e.g. `RandomBackgroundNoise`, `RandomImpulseResponse`).
  * Aimed at improving robustness to real-world acoustic conditions.

---

## How to Start

1. **Install the environment** using the steps in [Installation](#installation).
2. **Download assets** (noises, IRs, pretrained models, video encoder) using scripts in [Data and Assets](#data-and-assets).
3. **Precompute video embeddings** using the downloaded video encoder (see repo’s notebooks/scripts for details).
4. **Train a model**:

   * Baseline: `python train.py -cn=train_convtasnet.yaml`
   * AVSS: `python train.py -cn=train_rtfsnet.yaml`
5. **Run inference**:

   * ConvTasNet: `python inference.py -cn=inf_convtasnet.yaml`
   * RTFSNet: `python inference.py -cn=inf_rtfsnet.yaml inferencer.from_pretrained=weights/rtfsnet.pth`
6. **Compute metrics** (if you have ground truth):

   * `python calc_metrics.py -cn=calc_metrics_rtfsnet.yaml`
