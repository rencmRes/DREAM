# DREAM

DREAM is a deep-learning framework for fluorescence microscopy image restoration and super-resolution. The project contains training and inference code for a GAN-based DREAM model, with paired wide-field (WF) and ground-truth (GT) images as input data.

> Paper title, authors, institution, and citation information can be added here after the manuscript is finalized.

## Overview

DREAM aims to reconstruct high-resolution microscopy images from low-quality WF inputs. The current implementation combines:

- A DREAM generator for image restoration and super-resolution.
- A degradation representation encoder trained with a MoCo-style contrastive queue.
- A U-Net discriminator with spectral normalization for adversarial training.
- Pixel, perceptual, contrastive, and GAN losses for training.

The repository provides a compact workflow for training, testing, and saving restored results.

## Network Architecture

Place the network architecture figure at `docs/figures/dream_architecture.png`, then keep or update the link below.

![DREAM network architecture](docs/figures/dream_architecture.png)

Suggested figure content:

- WF image input.
- Encoder branch for degradation representation learning.
- DREAM restoration / super-resolution generator.
- GAN discriminator used during training.
- Output DREAM image and supervision from GT image.

## Results

Place representative qualitative results in `docs/results/`. The table below is reserved for GitHub visualization.

| Dataset | WF input | DREAM output | GT |
| --- | --- | --- | --- |
| ER | `docs/results/er_wf.png` | `docs/results/er_dream.png` | `docs/results/er_gt.png` |
| MTs | `docs/results/mts_wf.png` | `docs/results/mts_dream.png` | `docs/results/mts_gt.png` |

You can also add quantitative metrics here after evaluation.

| Dataset | PSNR | SSIM | Notes |
| --- | ---: | ---: | --- |
| ER | TBD | TBD | Add evaluation setting |
| MTs | TBD | TBD | Add evaluation setting |

## Repository Structure

```text
.
+-- data/
|   +-- dataset_DREAM.py          # DREAM paired WF/GT dataset
|   +-- select_dataset.py         # Dataset selector
+-- models/
|   +-- network_DREAM.py          # DREAM generator and encoder
|   +-- model_gan_DREAM.py        # GAN training and inference wrapper
|   +-- network_discriminator.py  # Discriminator definitions
|   +-- select_network.py         # Network selector
+-- options/
|   +-- train_DREAM_sr_x2_gan.json
|   +-- test_DREAM_sr_x2_gan.json
+-- testdata/
|   +-- ER/
|   +-- MTs/
+-- weights/
|   +-- ER.pth
|   +-- MTs.pth
+-- result/                       # Inference outputs
+-- main_train_gan_DREAM.py
+-- main_test_gan_DREAM.py
```

## Installation

Create a Python environment and install the required packages.

```bash
conda create -n dream python=3.10
conda activate dream
pip install torch torchvision numpy scipy scikit-image imageio opencv-python tqdm
```

Install the PyTorch version that matches your CUDA environment. For GPU training, follow the official PyTorch installation command for your CUDA version.

## Data Preparation

The dataset loader expects paired `.tif` files. WF and GT images should have matching file names.

```text
dataset_root/
+-- WF/
|   +-- 1.tif
|   +-- 2.tif
|   +-- ...
+-- GT/
    +-- 1.tif
    +-- 2.tif
    +-- ...
```

For training, update `options/train_DREAM_sr_x2_gan.json`:

```json
"dataroot_H": "path/to/GT",
"dataroot_L": "path/to/WF"
```

For testing, update `options/test_DREAM_sr_x2_gan.json`:

```json
"path": {
  "pretrained_netG": "weights/MTs.pth"
},
"datasets": {
  "test": {
    "dataroot_H": "testdata/MTs/GT",
    "dataroot_L": "testdata/MTs/WF"
  }
}
```

## Training

Run:

```bash
python main_train_gan_DREAM.py --opt options/train_DREAM_sr_x2_gan.json
```

Training outputs are saved under the configured training task directory:

```text
DREAM_GAN_MTs/
+-- dream_sr_x2_gan/
    +-- images/
    +-- models/
    +-- options/
    +-- train.log
```

The script automatically resumes from the latest checkpoint in the training `models/` directory when available.

## Testing

Run:

```bash
python main_test_gan_DREAM.py --opt options/test_DREAM_sr_x2_gan.json
```

Testing results are saved directly to:

```text
result/
```

The testing script does not create a new task folder from the configuration file.

## Pretrained Weights

The repository currently uses the following weight files:

| Dataset | Weight file |
| --- | --- |
| ER | `weights/ER.pth` |
| MTs | `weights/MTs.pth` |

Large model files may be better hosted through GitHub Releases, Google Drive, Zenodo, or another external storage service if the repository size becomes too large.

## Configuration

Important options are stored in JSON files under `options/`.

| Option | Description |
| --- | --- |
| `scale` | Super-resolution scale factor |
| `n_channels` | Number of image channels |
| `dataroot_H` | GT image directory |
| `dataroot_L` | WF image directory |
| `pretrained_netG` | Generator checkpoint used for inference |
| `checkpoint_print` | Training log interval |
| `checkpoint_save` | Model checkpoint interval |
| `checkpoint_test` | Validation interval during training |

## Citation

If you use this code, please cite the paper:

```bibtex
@article{dream2026,
  title   = {DREAM: Title to Be Updated},
  author  = {Author List to Be Updated},
  journal = {Journal or Conference to Be Updated},
  year    = {2026}
}
```

## Acknowledgements

This project builds on common components used in image restoration and super-resolution research, including GAN training, perceptual loss, and MoCo-style contrastive representation learning.

## License

Add the project license here before publishing the repository.
