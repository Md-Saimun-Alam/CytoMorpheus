# CytoMorpheus
AI-powered cell death classification using dual-modality microscopy and ensemble deep learning
# CytoMorpheus Analyzer

**AI-powered automated cell death classification using dual-modality microscopy and ensemble deep learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)

---

## Overview

CytoMorpheus is a deep learning pipeline for automated classification of cell death modalities (Control, Necrosis/H₂O₂, Apoptosis/Rapamycin) from label-free microscopy videos. The system supports both **Phase Contrast** and **Dark Field** microscopy modalities, automatically detected at runtime.

The pipeline integrates:
- **Cellpose** for cell segmentation
- **Hungarian algorithm** for multi-cell tracking across video frames
- **4-model voting ensemble** (3D-CNN, AlexNet, MobileNetV2, EfficientNet-B0) per modality
- **Automatic modality detection** (Phase Contrast vs Dark Field)
- **Gradio-based GUI** for real-time analysis

---

## Results Summary

### Phase Contrast Models (910 validation videos)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 3D-CNN | 94.18% | 94.3% | 94.2% | 94.2% |
| AlexNet | 86.70% | 87.1% | 86.7% | 86.7% |
| MobileNetV2 | 96.70% | 96.8% | 96.7% | 96.7% |
| EfficientNet-B0 | 95.38% | 95.5% | 95.4% | 95.4% |
| **Voting Ensemble** | **100%*** | - | - | - |

### Dark Field Models (910 validation videos)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 3D-CNN | 94.22% | 94.4% | 94.2% | 94.2% |
| AlexNet | 80.00% | 81.2% | 80.0% | 80.0% |
| MobileNetV2 | 83.30% | 83.7% | 83.3% | 83.2% |
| EfficientNet-B0 | 83.30% | 83.9% | 83.3% | 83.1% |
| **Voting Ensemble** | **96.70%** | - | - | - |

### Universal Predictor
| Component | Accuracy |
|---|---|
| Modality Detector | 100% |
| Cross-modal Meta-Learner | 95.51% |

*On 30-video test subset

---

## Repository Structure
```
CytoMorpheus/
├── 01_Training/
│   ├── Phase_Contrast/        # Phase contrast model training notebooks
│   │   ├── 3DCNN.ipynb
│   │   ├── AlexNet.ipynb
│   │   ├── MobileNetV2.ipynb
│   │   └── EfficientNet.ipynb
│   └── Dark_Field/            # Dark field model training notebooks
│       ├── 3DCNN.ipynb
│       ├── AlexNet.ipynb
│       ├── MobileNetV2.ipynb
│       └── EfficientNet.ipynb
├── 02_Evaluation/
│   └── Full_Evaluation.ipynb  # Complete evaluation, ensembles, figures
├── 03_GUI/
│   └── CytoMorpheus_Analyzer.ipynb  # Gradio-based analysis GUI
├── requirements.txt
└── README.md
```

---

## Pipeline
```
Input Video
    │
    ▼
Modality Detection (Phase / Dark Field)
    │
    ▼
Cellpose Segmentation (cyto3)
    │
    ▼
Cell Tracking (Hungarian Algorithm)
    │
    ▼
30-Frame Sequence Building per Cell
    │
    ▼
4-Model Ensemble Voting
(3D-CNN + AlexNet + MobileNetV2 + EfficientNet)
    │
    ▼
Classification: Control | Necrosis | Apoptosis
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

## Data

Videos are 30-frame sequences (224×224) from label-free microscopy:
- **Classes:** Control, H₂O₂ (Necrosis), Rapamycin (Apoptosis)
- **Modalities:** Phase Contrast, Dark Field
- **Split:** 80% train / 20% validation (stratified, seed=42)

---

---

## Journal Paper

This repository accompanies the following manuscript currently under review:

> **Md Saimun Alam**, Somaiyeh Khoubafarin Doust, Aniruddha Ray*  
> *"Multimodal Spatiotemporal Deep Learning for Real-Time Classification of Apoptosis and Necrosis Using Label-Free Microscopy"*  
> Machine Learning: Science and Technology (MLST), IOP Publishing *(under review)*

---

## Author

**Md Saimun Alam**  
PhD Student, Department of Physics and Astronomy  
University of Toledo, Toledo, OH 43606, USA  
📧 Mdsaimun.alam@rockets.utoledo.edu  

**Biophotonics & AI Laboratory**  
Principal Investigator: Dr. Aniruddha Ray

---

## Collaborators

- Somaiyeh Khoubafarin Doust — University of Toledo
- Dr. Aniruddha Ray — University of Toledo (PI)
