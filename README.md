# CytoMorpheus Analyzer

**Multimodal Spatiotemporal Deep Learning for Real-Time Classification of Apoptosis and Necrosis Using Label-Free Microscopy**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)

---

## Overview

CytoMorpheus is a deep learning pipeline for automated classification of cell death modalities — **Control**, **Necrosis (H₂O₂)**, and **Apoptosis (Rapamycin)** — from label-free microscopy videos of BT-20 breast cancer cells. The system supports both **Phase Contrast** and **Dark Field** modalities, automatically detected at runtime.

The pipeline integrates:
- **Cellpose** for label-free cell segmentation
- **Hungarian algorithm** for multi-cell tracking across frames
- **4-model voting ensemble** (3D-CNN, AlexNet-BiLSTM, MobileNetV2, EfficientNet-B0+Transformer) per modality
- **Automatic modality detection** (Phase Contrast vs Dark Field)
- **Meta-learner** for cross-modality fusion
- **Gradio-based GUI** for real-time analysis

---

## Results

### Dataset
| Modality | Total Videos | Train | Validation | Classes |
|---|---|---|---|---|
| Phase Contrast | 4,549 | 3,639 | 910 | Control=219, H₂O₂=275, RAP=416 |
| Dark Field | 3,802 | 3,041 | 761 | Control=178, H₂O₂=229, RAP=354 |

### Phase Contrast — Individual Models (910 validation videos)

| Model | Accuracy | F1-Macro | AUC-Macro |
|---|---|---|---|
| 3D-CNN | 89.34% | 88.85% | 98.02% |
| AlexNet-BiLSTM | 88.35% | 87.77% | 97.06% |
| MobileNetV2 | 92.09% | 91.63% | 98.38% |
| EfficientNet-B0 | 95.38% | 94.98% | 99.52% |
| **Voting Ensemble** | **96.26%** | **96.06%** | **99.79%** |

### Dark Field — Individual Models (761 validation videos)

| Model | Accuracy | F1-Macro | AUC-Macro |
|---|---|---|---|
| 3D-CNN | 94.22% | 93.93% | 99.34% |
| AlexNet-BiLSTM | 90.67% | 89.47% | 97.12% |
| MobileNetV2 | 88.17% | 87.31% | 96.88% |
| EfficientNet-B0 | 91.85% | 91.15% | 98.17% |
| **Voting Ensemble** | **94.09%** | **93.48%** | **99.54%** |

### Universal Predictor — Combined (1,671 videos)

| Component | Result |
|---|---|
| Modality Detection | 99.04% |
| Overall Accuracy | 95.51% |
| F1-Macro | 95.20% |
| AUC-Macro | 99.50% |
| Control (F1) | 92.56% |
| H₂O₂ / Necrosis (F1) | 96.89% |
| RAP / Apoptosis (F1) | 96.17% |

---

## Repository Structure
```
CytoMorpheus/
├── 01_Training/
│   ├── Phase_Contrast/
│   │   ├── 3DCNN.ipynb
│   │   ├── AlexNet.ipynb
│   │   ├── MobileNetV2.ipynb
│   │   └── EfficientNet.ipynb
│   └── Dark_Field/
│       ├── 3DCNN.ipynb
│       ├── AlexNet.ipynb
│       ├── MobileNetV2.ipynb
│       └── EfficientNet.ipynb
├── 02_Evaluation/
│   └── Full_Evaluation.ipynb
├── 03_GUI/
│   └── CytoMorpheus_Analyzer.ipynb
├── requirements.txt
└── README.md
```

---

## Pipeline
```
Input Video
    │
    ▼
Modality Detection (Phase Contrast / Dark Field)  →  99.04% accuracy
    │
    ▼
Cellpose Segmentation (cyto3 model)
    │
    ▼
Cell Tracking (Hungarian Algorithm)
    │
    ▼
30-Frame Sequence per Cell (224×224, step=2)
    │
    ▼
4-Model Voting Ensemble
(3D-CNN + AlexNet-BiLSTM + MobileNetV2 + EfficientNet-B0)
    │
    ▼
Meta-Learner Fusion (cross-modality)
    │
    ▼
Classification: Control | Necrosis (H₂O₂) | Apoptosis (RAP)
```

---

## Setup
```bash
pip install tensorflow==2.19.0
pip install keras==3.10.0
pip install gradio==6.2.0
pip install cellpose
pip install opencv-python numpy scipy scikit-learn
```

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
- Dr. Aniruddha Ray — University of Toledo (PI)
