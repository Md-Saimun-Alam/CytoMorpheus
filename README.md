# CytoMorpheus

**Deep Learning Assisted Label-Free Microscopy for Cell Death Analysis in Real-Time**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)

---

## Overview

CytoMorpheus classifies individual BT-20 cells as **Control**, **Apoptosis (Raptinal)** or **Necrosis (H₂O₂)** from label-free phase-contrast and dark-field time-lapse videos. Propidium iodide (PI) is used only to label the training cells.

The pipeline includes:
- **Cellpose-SAM** for cell segmentation
- **Mutual-best-overlap tracking** across frames
- **15-frame window** around the death frame of each cell, cropped to 48 × 48 px
- **4-model voting ensemble** per modality (3D-CNN, AlexNet-BiLSTM, MobileNetV2, EfficientNet-B0+Transformer), 3 seeds each
- **Cross-modality fusion** by averaging the probabilities of the phase-contrast and dark-field models
- **CytoMorpheus Analyzer**, a desktop GUI for analyzing raw videos

---

## Results

### Dataset

| Class | All cells (11 recordings) | Held-out test (3 recordings) |
|---|---|---|
| Control | 2,254 | 1,098 |
| Apoptosis | 1,515 | 790 |
| Necrosis | 877 | 455 |
| **Total** | **4,646** | **2,343** |

The test recordings (one untreated, one 100 µM Raptinal, one 25 mM H₂O₂) are never seen during training.

### Held-out recordings, balanced accuracy (mean ± sd, 3 seeds)

| Model | Phase Contrast | Dark Field | Fusion |
|---|---|---|---|
| 3D-CNN | 0.924 ± 0.007 | 0.922 ± 0.011 | 0.953 ± 0.006 |
| AlexNet-BiLSTM | 0.924 ± 0.008 | 0.951 ± 0.003 | 0.960 ± 0.003 |
| MobileNetV2 | 0.920 ± 0.008 | 0.975 ± 0.004 | 0.968 ± 0.005 |
| EfficientNet-B0+Transformer | 0.894 ± 0.035 | 0.969 ± 0.002 | 0.969 ± 0.007 |
| **Voting Ensemble** | **0.953 ± 0.005** | **0.973 ± 0.001** | **0.981 ± 0.002** |

Full metrics (accuracy, macro-F1, macro-AUC, per-class recall, early fusion) are in `results/`.

---

## Repository Structure
```
CytoMorpheus/
├── preprocessing/
│   ├── config.py              acquisition and preprocessing parameters
│   ├── segment_track.py       Cellpose-SAM segmentation and tracking
│   ├── pi_ground_truth.py     per-cell death frame from the PI channel
│   └── build_dataset.py       15-frame windows -> cache.npy, meta.npz, index.csv
├── cytomorpheus/
│   ├── models.py              the four architectures
│   ├── data.py                input channels, augmentation, splits
│   ├── train.py               training
│   ├── evaluate.py            voting, cross-modality fusion, metrics
│   └── early_fusion.py        six-channel early-fusion comparison
├── gui/
│   ├── cytomorpheus_analyzer.py
│   ├── build_exe.bat
│   └── app_icon.ico
├── results/
│   ├── table_heldout_and_fusion.csv     Table 1
│   ├── metrics_per_seed_heldout.csv     per-seed values behind Table 1
│   └── detector_vs_PI_heldout.csv       label-free rupture detection vs PI (Figure S1)
├── CITATION.cff
├── requirements.txt
└── README.md
```

---

## Pipeline
```
Phase-Contrast + Dark-Field Videos
    │
    ▼
Cellpose-SAM Segmentation (phase contrast, same masks for dark field)
    │
    ▼
Cell Tracking (mutual-best overlap)
    │
    ▼
15-Frame Window per Cell (death frame -10 to +4, 48×48)
    │
    ▼
4-Model Voting Ensemble per Modality
    │
    ▼
Cross-Modality Fusion (average of 8 models)
    │
    ▼
Classification: Control | Apoptosis | Necrosis
```

During training the death frame comes from PI. In the Analyzer it is found from the label-free images.

---

## Setup
```bash
pip install -r requirements.txt
```
Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/

---

## Usage
```bash
# segmentation, tracking and PI ground truth (one recording)
python -m preprocessing.segment_track --phase rec__phase.avi --out work/rec_masks.npz
python -m preprocessing.pi_ground_truth --fluor rec__fluor.avi --labels work/rec_masks.npz --treatment RAPTINAL --out work/rec_truth.csv

# build the dataset from a manifest (recording,phase,dark,labels,truth,fold)
python -m preprocessing.build_dataset --manifest recordings.csv --out 00_data

# train and evaluate (held-out recordings = fold F1)
python -m cytomorpheus.train --data 00_data --out 01_models --protocol LORO --fold F1
python -m cytomorpheus.evaluate --models 01_models --out 02_results --protocol LORO --fold F1

# early fusion (six-channel input)
python -m cytomorpheus.early_fusion --data 00_data --out 01_models --fold F1

# CytoMorpheus Analyzer
python gui/cytomorpheus_analyzer.py
```

`--treatment` is `RAPTINAL`, `H2O2` or `NONE`. Recordings marked `F1` in the manifest are the held-out test set.

---

## Data and Trained Models

Raw videos and trained model weights are not in this repository because of their size. They are available from the corresponding author on request.

---

## Citation

The paper describing this work is under review at *Journal of Biophotonics*. Until it is published, please cite the software:

```
Alam, M. S., Khoubafarin Doust, S., & Ray, A. (2026). CytoMorpheus (Version 1.0.0) [Computer software]. https://github.com/Md-Saimun-Alam/CytoMorpheus
```

The citation will be updated with the article DOI on publication. See `CITATION.cff`.

---

## Author

**Md Saimun Alam**
PhD Student, Department of Physics and Astronomy
University of Toledo, Toledo, OH 43606, USA
📧 Mdsaimun.alam@rockets.utoledo.edu

**Biophotonics & AI Laboratory**
Principal Investigator: Dr. Aniruddha Ray

**Collaborators**
- Somaiyeh Khoubafarin Doust — University of Toledo
- Dr. Aniruddha Ray — University of Toledo (PI, corresponding author)

Supported by NIH/NIBIB grants 1R15EB034552-01 and 3R15EB034552-01S1.
