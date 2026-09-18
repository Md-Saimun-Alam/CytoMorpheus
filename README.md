# CytoMorpheus

**Deep learning assisted label-free microscopy for cell death analysis in real time**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)

Code for the paper by Md Saimun Alam, Somaiyeh Khoubafarin Doust and Aniruddha Ray
(University of Toledo), *Journal of Biophotonics*.

CytoMorpheus tells apoptosis from necrosis in individual cells, from phase-contrast
and dark-field time-lapse videos alone. No stain is needed at analysis time. A
fluorescent stain is used once, during training, to give every training cell a
verified outcome.

---

## What the pipeline does

1. **Segmentation and tracking.** Cellpose-SAM outlines every cell in every
   phase-contrast frame. The masks are linked across frames by mutual-best
   overlap, with gap closing of up to two frames and a 12 µm displacement cap, so
   each cell gets one continuous track. The same masks are applied to the
   dark-field frames, which share the same coordinates.
2. **Per-cell ground truth.** Cellpose runs on the propidium iodide channel, so
   stained nuclei are detected as objects rather than scored by intensity. A cell
   is called dead at the first frame where a nucleus covers more than half of its
   mask for two consecutive frames. The label follows the cell, not the dish.
3. **Anchored windows.** Each cell becomes a fixed 15-frame window, from 10 frames
   before its death frame to 4 after it, cropped to 48 × 48 px. The fixed length
   matters: raw track length alone would reveal the treatment, because H₂O₂ kills
   faster than Raptinal, with a median time to loss of membrane integrity of 63
   minutes against 144 minutes.
4. **Three input channels.** Intensity normalised within the window, the difference
   between consecutive frames, and the dilated mask of the tracked cell, so the
   network knows which cell in a crowded crop it is judging.
5. **Four architectures, two modalities.** Each architecture is trained separately
   on phase contrast and on dark-field, three seeds each, 24 runs.
6. **Voting and cross-modality fusion.** Averaging of class probabilities, first
   across the four architectures within a modality, then across the two modalities
   for the same cell.
7. **CytoMorpheus Analyzer.** A desktop application that runs the whole pipeline on
   a raw video, without PI, finding the death frame from morphology instead.

---

## Pipeline

```
Phase-contrast video            Dark-field video           PI video (training only)
        │                              │                              │
        └──────────────┬───────────────┘                              │
                       ▼                                              ▼
        Cellpose-SAM segmentation  ──────── same masks ───────►  Cellpose-SAM on PI
                       │                                              │
                       ▼                                              ▼
        mutual-best-overlap tracking                    death frame = first frame with
        gap ≤ 2 frames, step ≤ 12 µm                    > 50 % nucleus for 2 frames
                       │                                              │
                       └──────────────┬───────────────────────────────┘
                                      ▼
                   anchored 15-frame window, t−10 … t+4, 48 × 48 px
                   3 channels: normalised intensity, frame difference, mask
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
        4 models on phase contrast                4 models on dark-field
        3D-CNN · AlexNet–BiLSTM ·                 3D-CNN · AlexNet–BiLSTM ·
        MobileNetV2 · EffNet-B0+Transformer       MobileNetV2 · EffNet-B0+Transformer
                 │                                         │
                 ▼  average 4 probability vectors           ▼
        phase voting ensemble                     dark-field voting ensemble
                 └────────────────────┬────────────────────┘
                                      ▼  average all 8 vectors for the same cell
                        cross-modality fusion  →  Apoptosis | Control | Necrosis
```

---

## How the models are combined

There is no single fused network and it has no weights of its own. The eight
networks are separate: a phase model never sees a dark-field image, nothing is
trained jointly, and no weights are shared. Fusion is a decision rule applied at
prediction time.

- **Voting, within a modality.** Each of the four models outputs a probability for
  apoptosis, control and necrosis. The four vectors are averaged and the highest
  mean wins. Averaging is used rather than a majority vote because it keeps each
  model's confidence and cannot tie.
- **Cross-modality fusion, across modalities.** Every cell has a phase-contrast and
  a dark-field sequence built from the same frames and the same masks, which is
  possible because both channels were recorded from the same field without moving
  the stage. The four phase probabilities and the four dark-field probabilities of
  that cell are averaged, so the final call rests on eight models.

Fusing at the probability level, rather than stacking both modalities into one
six-channel input and training jointly, has two advantages. Each modality is scored
only by models trained on it, so a weak signal in one cannot corrupt what the other
learned, and the same models still run when only one modality was recorded. The
alternative is included as a control in `cytomorpheus/early_fusion.py` and is worse
on unseen recordings, 0.959 against 0.981.

---

## How it is evaluated

Three whole recordings are held out, one untreated, one 100 µM Raptinal and one
25 mM H₂O₂, 2,343 cells in total. No cell from those recordings is seen during
training. This is the strict test, because cells from one recording share a dish,
illumination and focus, so a model trained on them can learn to recognise those
conditions rather than cell death itself.

Of the training cells, 15 % are held back for validation and used only to select
the checkpoint. The test cells are scored once. Every experiment is repeated with
three random seeds and reported as mean ± standard deviation.

---

## Results

Balanced accuracy on the three held-out recordings, mean ± sd over three seeds.
The full tables, including accuracy, macro F1, macro AUC and the per-class values,
are in [`results/`](results).

| Model | Phase contrast | Dark-field | Cross-modality fusion |
|---|---|---|---|
| 3D-CNN | 0.924 ± 0.007 | 0.922 ± 0.011 | 0.953 ± 0.006 |
| AlexNet–BiLSTM | 0.924 ± 0.008 | 0.951 ± 0.003 | 0.960 ± 0.003 |
| MobileNetV2 | 0.920 ± 0.008 | 0.975 ± 0.004 | 0.968 ± 0.005 |
| EfficientNet-B0 + Transformer | 0.894 ± 0.035 | 0.969 ± 0.002 | 0.969 ± 0.007 |
| **Voting ensemble** | **0.953 ± 0.005** | **0.973 ± 0.001** | **0.981 ± 0.002** |

Early fusion, the six-channel control, reaches 0.959 ± 0.008 as an ensemble, below
both the dark-field ensemble and probability-level fusion.

Cross-modality fusion reaches an accuracy of 0.984, a macro F1 of 0.981 and a macro
AUC of 0.999, with per-class recalls of 0.977 for apoptosis, 0.992 for control and
0.974 for necrosis.

---

## Dataset

4,646 cells with a verified outcome from 11 recordings of BT-20 cells, 1,515
apoptotic, 2,254 control and 877 necrotic. A further 3,111 treated cells never
became PI-positive within their recording, so they carry no label; they are held
aside and used only to ask how the trained models respond to a treated cell with an
intact membrane (`results/exposed_alive_arm.csv`).

In one H₂O₂ recording the fluorescence channel was unusable, because a large
fraction of every cell already read as PI-positive in the first frames. The 313
cells of that recording were anchored on the phase-contrast collapse instead of on
PI, using the same criterion the Analyzer applies. The other 564 necrotic cells,
and every apoptotic and control cell, are anchored on PI.

The image data is large and is not in this repository. The scripts in
`preprocessing/` rebuild the dataset from the raw videos, and the parameters used
are fixed in `config.json` and `preprocessing/config.py`.

---

## Repository layout

```
preprocessing/      raw videos -> training windows
  config.py           locked parameters, exactly as used
  segment_track.py    Cellpose-SAM segmentation, mutual-best-overlap tracking
  pi_ground_truth.py  per-cell death frame from the PI channel
  build_dataset.py    anchored 15-frame windows -> cache.npy, meta.npz, index.csv
cytomorpheus/       models, training, evaluation
  models.py           the four architectures
  data.py             input channels, augmentation, the data splits
  train.py            training, one run per architecture x modality x seed
  evaluate.py         voting, cross-modality fusion, metric tables
  early_fusion.py     the six-channel control experiment
gui/                CytoMorpheus Analyzer (desktop application)
results/            the metric tables reported in the paper
docs/REPRODUCE.md   step-by-step commands
```

---

## Installation

```bash
git clone https://github.com/Md-Saimun-Alam/CytoMorpheus.git
cd CytoMorpheus
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch for your own CUDA version from <https://pytorch.org/get-started/locally/>.
The models in the paper were trained on a single NVIDIA GeForce RTX 5070 Ti.

---

## Quick start

Rebuild the dataset from raw videos, then train and evaluate:

```bash
# one recording at a time
python -m preprocessing.segment_track --phase rec__phase.avi --out work/rec_masks.npz
python -m preprocessing.pi_ground_truth --fluor rec__fluor.avi --labels work/rec_masks.npz \
       --treatment RAPTINAL --out work/rec_truth.csv

# all recordings, listed in a manifest
python -m preprocessing.build_dataset --manifest recordings.csv --out 00_data

# 4 architectures x 2 modalities x 3 seeds, three recordings held out
python -m cytomorpheus.train    --data 00_data  --out 01_models   --protocol LORO --fold F1
python -m cytomorpheus.evaluate --models 01_models --out 02_results --protocol LORO --fold F1
```

`docs/REPRODUCE.md` has the full sequence, including the early-fusion control.

---

## CytoMorpheus Analyzer

```bash
python gui/cytomorpheus_analyzer.py
```

Load one or two videos of the same field. The modality is detected from the images,
so it does not have to be specified. Every cell is segmented, tracked and
classified, the death frame is found from the label-free images rather than from
PI, and the same 15-frame window and averaging are used as in training. A field of
about 1,000 cells takes under 5 minutes on the GPU above. The output is a table
with the class and class probabilities of every cell, a summary of the field, and
an annotated video in which each cell is outlined in the colour of its class.

The control class is labelled **Alive** in the interface, because the software
reports what a cell is rather than how it was treated. A cell is drawn in the Alive
colour before its 15-frame window, blends into its class colour across the window,
and keeps the class colour from the detected death frame on, so the field turns over
time instead of showing the final call from the first frame. Cells tracked for
fewer than 15 frames are reported as unclassified rather than forced into a class.

`gui/build_exe.bat` builds a standalone Windows application with PyInstaller. The
trained weights are expected in a `models/` folder beside the executable, laid out
as `models/<architecture>/model_{phase,dark}_LORO_F1_s{0,1,2}.pt`.

Trained weights are not in this repository because of their size. They are
available from the corresponding author on reasonable request.

---

## Citing

If you use this code, please cite the paper. See `CITATION.cff`.

---

## Authors

**Md Saimun Alam**
PhD Student, Department of Physics and Astronomy
University of Toledo, Toledo, OH 43606, USA
· mdsaimun.alam@rockets.utoledo.edu

**Biophotonics & AI Laboratory**
Principal Investigator: Dr. Aniruddha Ray

**Collaborators**
- Somaiyeh Khoubafarin Doust — Department of Physics and Astronomy, University of Toledo
- Aniruddha Ray — Department of Physics and Astronomy and Northwest Ohio Cancer Research Institute, University of Toledo (corresponding author, aniruddha.ray@utoledo.edu)

Supported by the National Institute of Biomedical Imaging and Bioengineering of the
National Institutes of Health, grants 1R15EB034552-01 and 3R15EB034552-01S1.
