# =============================================================================
# CytoMorpheus — Evaluation Pipeline
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Pipeline  : (1) Per-model evaluation   →  8 models (Phase + Dark)
#             (2) Voting Ensemble         →  Phase & Dark separately
#             (3) Universal Predictor    →  Modality detector + Meta-learner
# Input     : 30 frames × 224×224 × 3 per video  (identical to training)
# Classes   : Control | H2O2 (Necrosis) | RAP (Apoptosis)
# Splits    : SEED=42, test_size=0.2, stratified  (same as all training files)
# Outputs   : prediction CSVs | confusion matrices | ROC & PR curves
#             per-model metrics summary | ensemble & universal metrics
# =============================================================================
#
# IMPORTANT — Checkpoint format notes:
#   Full models  (.h5 / .keras) : Phase_3DCNN, Phase_MobileNetV2,
#                                  Phase_EfficientNet, Dark_MobileNetV2,
#                                  Dark_EfficientNet
#   Weights-only (.weights.h5)  : Phase_AlexNet, Dark_AlexNet, Dark_3DCNN
#     → These three require architecture rebuild before loading weights.
#       See Section 3 (Architecture Rebuilders) below.
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import glob
import random
import pickle
import collections

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')   # headless rendering for Colab
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models, applications, regularizers, backend as K
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Rescaling, TimeDistributed, GlobalAveragePooling2D,
    Dropout, Bidirectional, LSTM, Dense, Input
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)


# ── Data paths  (edit to match your Google Drive layout) ─────────────────────

PHASE_DATA_DIR  = "/content/drive/MyDrive/Cell Death/bright field"
DARK_DATA_DIR   = "/content/drive/MyDrive/Cell Death/dark field"
MODELS_ROOT     = "/content/drive/MyDrive/Journal_cell death/03_Trained_Models"
META_MODELS_DIR = "/content/drive/MyDrive/Journal_cell death/meta_models"
MODALITY_DET    = "/content/drive/MyDrive/Journal_cell death/04_Universal_Predictor/modality_detector.keras"
OUTPUT_ROOT     = "/content/drive/MyDrive/Journal on cell death"


# ── Model weight paths  (align with your saved checkpoint filenames) ──────────

PHASE_MODEL_PATHS = {
    "3dcnn"    : f"{MODELS_ROOT}/Phase_Contrast/3dcnn_advanced/best_model.h5",
    "alexnet"  : f"{MODELS_ROOT}/Phase_Contrast/alexnet_pipeline1/best_alexnet.h5",
    "mobilenet": f"{MODELS_ROOT}/Phase_Contrast/mobilenet_pipeline1/best_mobilenet.h5",
    "effnet"   : f"{MODELS_ROOT}/Phase_Contrast/effnetb0_partialft_transformer/model.keras",
}

DARK_MODEL_PATHS = {
    "3dcnn"    : f"{MODELS_ROOT}/Dark_Field/3dcnn_pipeline1/best_model.h5",
    "alexnet"  : f"{MODELS_ROOT}/Dark_Field/alexnet_pipeline1/best_alexnet.h5",
    "mobilenet": f"{MODELS_ROOT}/Dark_Field/mobilenet_pipeline1/best_mobilenet.h5",
    "effnet"   : f"{MODELS_ROOT}/Dark_Field/effnetb0_partialft_transformer/model.keras",
}

# Models saved as weights-only (need architecture rebuild before loading)
WEIGHTS_ONLY_MODELS = {"phase_alexnet", "dark_alexnet", "dark_3dcnn"}


# ── Shared configuration  (must match all training files exactly) ─────────────

class CFG:
    classes     = ["Control", "H2O2", "RAP"]
    n_frames    = 30
    frame_step  = 2
    output_size = (224, 224)   # (height, width)
    seed        = 42


# ── Visual style ─────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Control" : "#2E7D32",
    "H2O2"    : "#C62828",
    "RAP"     : "#1565C0",
}


# ── Create output directories ─────────────────────────────────────────────────

for sub in ["evaluations", "predictions_csv", "figures", "manuscript_figures"]:
    os.makedirs(f"{OUTPUT_ROOT}/{sub}", exist_ok=True)

print("=" * 70)
print("CytoMorpheus  —  Evaluation Pipeline")
print("=" * 70)
print(f"  Phase data    : {PHASE_DATA_DIR}")
print(f"  Dark data     : {DARK_DATA_DIR}")
print(f"  Models root   : {MODELS_ROOT}")
print(f"  Output root   : {OUTPUT_ROOT}")
print(f"  Classes       : {CFG.classes}")
print(f"  n_frames      : {CFG.n_frames}  |  frame_step : {CFG.frame_step}")
print(f"  output_size   : {CFG.output_size}")
print(f"  random seed   : {CFG.seed}")
print("=" * 70)


# ── Section 2: Frame Sampler (deterministic for evaluation) ──────────────────
#
# Training used random start for augmentation.
# Evaluation uses a fixed centre start to guarantee reproducibility.

def frames_from_video_file(video_path, n_frames=CFG.n_frames,
                            output_size=CFG.output_size,
                            frame_step=CFG.frame_step,
                            deterministic=True):
    """
    Extract n_frames evenly-stepped frames from a video file.

    Args:
        video_path    : path to the .avi video
        n_frames      : number of frames to sample
        output_size   : (H, W) resize target
        frame_step    : number of frames to advance between samples
        deterministic : if True uses centre-of-video start (evaluation mode);
                        if False uses random start (training augmentation mode)

    Returns:
        float32 numpy array of shape (n_frames, H, W, 3) in RGB order [0, 1].
        Returns zero array if the video cannot be opened.
    """
    h, w  = output_size
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need  = 1 + (n_frames - 1) * frame_step

    if total == 0:
        cap.release()
        return np.zeros((n_frames, h, w, 3), dtype=np.float32)

    if need > total:
        start = 0
    elif deterministic:
        start = max(0, (total - need) // 2)   # centre of video
    else:
        start = random.randint(0, total - need)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    result = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            result.append(np.zeros((h, w, 3), dtype=np.float32))
        else:
            frame = cv2.resize(frame, (w, h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            result.append(frame)
        for _ in range(frame_step - 1):
            cap.grab()

    cap.release()
    return np.stack(result)   # (n_frames, H, W, 3)


# ── Section 3: Architecture Rebuilders (for weights-only checkpoints) ─────────
#
# Phase_AlexNet, Dark_AlexNet, Dark_3DCNN were saved with save_weights_only=True.
# load_model() will fail on these — we must rebuild the graph then call
# model.load_weights(path).

def _build_alexnet_conv(input_shape):
    """AlexNet convolutional trunk (identical to training files)."""
    inp = Input(shape=input_shape)
    x   = Conv2D(96,  kernel_size=11, strides=4, padding='valid')(inp)
    x   = BatchNormalization()(x)
    x   = Activation('relu')(x)
    x   = MaxPooling2D(pool_size=3, strides=2)(x)
    x   = Conv2D(256, kernel_size=5, padding='same')(x)
    x   = BatchNormalization()(x)
    x   = Activation('relu')(x)
    x   = MaxPooling2D(pool_size=3, strides=2)(x)
    x   = Conv2D(384, kernel_size=3, padding='same')(x); x = Activation('relu')(x)
    x   = Conv2D(384, kernel_size=3, padding='same')(x); x = Activation('relu')(x)
    x   = Conv2D(256, kernel_size=3, padding='same')(x); x = Activation('relu')(x)
    x   = MaxPooling2D(pool_size=3, strides=2)(x)
    return models.Model(inp, x, name='alexnet_conv')


def build_alexnet_bilstm(input_shape, n_frames, num_classes,
                          fine_tune_at=50, dropout=0.4, lstm_units=128,
                          model_name='AlexNetVideo_BiLSTM'):
    """Rebuild AlexNet-BiLSTM graph (Phase and Dark share the same topology)."""
    conv_base = _build_alexnet_conv(input_shape)
    conv_base.trainable = True
    if fine_tune_at:
        for layer in conv_base.layers[:-fine_tune_at]:
            layer.trainable = False

    video_in = Input(shape=(n_frames, *input_shape), name='video_input')
    x = Rescaling(scale=255.0)(video_in)
    x = TimeDistributed(conv_base, name='td_alexnet')(x)
    x = TimeDistributed(GlobalAveragePooling2D(), name='td_gap2d')(x)
    x = Dropout(dropout, name='drop_pre_lstm')(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=False), name='bi_lstm')(x)
    x = Dropout(dropout, name='drop_post_lstm')(x)
    out = Dense(num_classes, activation='softmax', name='predictions')(x)
    return models.Model(video_in, out, name=model_name)


def build_dark_3dcnn(input_shape, num_classes):
    """
    Rebuild Dark_3DCNN graph (3-block variant trained with focal loss).
    Loss is not required here — inference only.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(32,  (3,3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D((1, 2, 2))(x)
    x = layers.Conv3D(64,  (3,3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Conv3D(128, (3,3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, out, name='Dark_3DCNN')


def load_model_safe(model_key, path):
    """
    Load a model from disk, handling both full-model and weights-only formats.

    Full-model  : Phase_3DCNN, Phase_MobileNetV2, Phase_EfficientNet,
                  Dark_MobileNetV2, Dark_EfficientNet
    Weights-only: Phase_AlexNet, Dark_AlexNet, Dark_3DCNN
                  (architecture is rebuilt, then weights are loaded)

    Args:
        model_key : string key like 'phase_alexnet', 'dark_3dcnn', etc.
        path      : filesystem path to the checkpoint

    Returns:
        Keras model ready for inference (compile=False).
    """
    frame_shape  = (*CFG.output_size, 3)
    input_shape  = (CFG.n_frames, *CFG.output_size, 3)
    num_classes  = len(CFG.classes)

    if model_key in WEIGHTS_ONLY_MODELS:
        print(f"  [weights-only] Rebuilding architecture for {model_key} ...")

        if model_key in ("phase_alexnet", "dark_alexnet"):
            m = build_alexnet_bilstm(frame_shape, CFG.n_frames, num_classes)
        elif model_key == "dark_3dcnn":
            m = build_dark_3dcnn(input_shape, num_classes)

        m.load_weights(path)
        print(f"  ✓ Weights loaded : {os.path.basename(path)}")
        return m
    else:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"  ✓ Full model loaded : {os.path.basename(path)}")
        return m


# ── Section 4: Data Split (must reproduce training splits exactly) ────────────

def build_val_split(data_dir, label=None):
    """
    Reproduce the exact 80/20 validation split used during training.
    Uses SEED=42 and stratification — identical to all 8 training scripts.

    Args:
        data_dir : root folder containing  <class>/*.avi  sub-directories
        label    : optional label string for logging (e.g. 'Phase', 'Dark')

    Returns:
        val_paths  : list of validation video paths
        val_labels : list of integer class indices
    """
    file_paths, targets = [], []
    for idx, cls in enumerate(CFG.classes):
        vids = glob.glob(os.path.join(data_dir, cls, "*.avi"))
        file_paths.extend(vids)
        targets.extend([idx] * len(vids))

    targets = np.array(targets)

    _, val_paths, _, val_labels = train_test_split(
        file_paths, targets,
        test_size=0.2,
        stratify=targets,
        random_state=CFG.seed
    )

    tag = f" [{label}]" if label else ""
    print(f"{tag} Total: {len(file_paths)}  |  Val: {len(val_paths)}")
    print(f"     Distribution: {dict(collections.Counter(val_labels))}")
    return val_paths, list(val_labels)


print("\n── Building validation splits ──────────────────────────────────────")
phase_val_paths,  phase_val_labels  = build_val_split(PHASE_DATA_DIR, "Phase")
dark_val_paths,   dark_val_labels   = build_val_split(DARK_DATA_DIR,  "Dark")


# ── Section 5: Single-Model Evaluator ────────────────────────────────────────

def evaluate_model(model_key, model_path, modality_tag,
                   val_paths, val_labels):
    """
    Run full evaluation for one model:
      - Inference on all validation videos
      - Saves raw prediction CSV
      - Computes: Accuracy, Precision, Recall, F1, AUC (macro + per-class)
      - Saves: Confusion Matrix, ROC curves, Precision-Recall curves
      - Returns a one-row metrics DataFrame

    Args:
        model_key    : key for load_model_safe  (e.g. 'phase_alexnet')
        model_path   : filesystem path to checkpoint
        modality_tag : string prefix for output files  (e.g. 'PhaseContrast')
        val_paths    : list of validation video paths
        val_labels   : list of integer ground-truth labels
    """
    print(f"\n{'='*70}")
    print(f"  EVALUATING : {modality_tag}_{model_key}")
    print(f"{'='*70}")

    model = load_model_safe(model_key, model_path)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_probs = [], []

    for i, path in enumerate(val_paths):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(val_paths)} ...")

        frames = frames_from_video_file(path, deterministic=True)
        batch  = np.expand_dims(frames, axis=0)
        probs  = model.predict(batch, verbose=0)[0]

        all_preds.append(int(np.argmax(probs)))
        all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "video_path"      : [os.path.basename(p) for p in val_paths],
        "true_label"      : [CFG.classes[l] for l in val_labels],
        "predicted_label" : [CFG.classes[p] for p in all_preds],
        "true_idx"        : val_labels,
        "pred_idx"        : all_preds,
        "prob_Control"    : all_probs[:, 0],
        "prob_H2O2"       : all_probs[:, 1],
        "prob_RAP"        : all_probs[:, 2],
    })
    csv_path = f"{OUTPUT_ROOT}/predictions_csv/{modality_tag}_{model_key}_predictions.csv"
    pred_df.to_csv(csv_path, index=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true = np.array(val_labels)
    y_pred = all_preds
    y_prob = all_probs

    acc                      = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _  = precision_recall_fscore_support(y_true, y_pred, average='macro',    zero_division=0)
    p_wei, r_wei, f1_wei, _  = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    p_cls, r_cls, f1_cls, _  = precision_recall_fscore_support(y_true, y_pred, average=None,       zero_division=0, labels=[0,1,2])
    auc_cls                  = [roc_auc_score((y_true == i).astype(int), y_prob[:, i]) for i in range(3)]
    auc_mac                  = float(np.mean(auc_cls))
    cm                       = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"  Accuracy : {acc:.4f}  |  F1-Macro : {f1_mac:.4f}  |  AUC-Macro : {auc_mac:.4f}")

    metrics_df = pd.DataFrame({
        "Model"            : [f"{modality_tag}_{model_key}"],
        "Modality"         : [modality_tag],
        "Accuracy"         : [acc],
        "Precision_Macro"  : [p_mac],
        "Recall_Macro"     : [r_mac],
        "F1_Macro"         : [f1_mac],
        "F1_Weighted"      : [f1_wei],
        "AUC_Macro"        : [auc_mac],
        "Control_Precision": [p_cls[0]], "Control_Recall": [r_cls[0]],
        "Control_F1"       : [f1_cls[0]],"Control_AUC"   : [auc_cls[0]],
        "H2O2_Precision"   : [p_cls[1]], "H2O2_Recall"   : [r_cls[1]],
        "H2O2_F1"          : [f1_cls[1]],"H2O2_AUC"      : [auc_cls[1]],
        "RAP_Precision"    : [p_cls[2]], "RAP_Recall"    : [r_cls[2]],
        "RAP_F1"           : [f1_cls[2]],"RAP_AUC"       : [auc_cls[2]],
    })
    metrics_df.to_csv(
        f"{OUTPUT_ROOT}/evaluations/{modality_tag}_{model_key}_metrics.csv",
        index=False
    )

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CFG.classes, yticklabels=CFG.classes,
                ax=ax, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12, weight='bold')
    ax.set_ylabel('True',      fontsize=12, weight='bold')
    ax.set_title(f'Confusion Matrix — {modality_tag} {model_key}',
                 fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{modality_tag}_{model_key}_cm.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── ROC Curves ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for i, cls in enumerate(CFG.classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], linewidth=2.5,
                label=f'{cls} (AUC={auc_cls[i]:.3f})', alpha=0.85)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=12, weight='bold')
    ax.set_title(f'ROC Curves — {modality_tag} {model_key}',
                 fontsize=13, weight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{modality_tag}_{model_key}_roc.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── Precision-Recall Curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for i, cls in enumerate(CFG.classes):
        prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
        ap = average_precision_score((y_true == i).astype(int), y_prob[:, i])
        ax.plot(rec, prec, color=CLASS_COLORS[cls], linewidth=2.5,
                label=f'{cls} (AP={ap:.3f})', alpha=0.85)
    ax.set_xlabel('Recall',    fontsize=12, weight='bold')
    ax.set_ylabel('Precision', fontsize=12, weight='bold')
    ax.set_title(f'PR Curves — {modality_tag} {model_key}',
                 fontsize=13, weight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{modality_tag}_{model_key}_pr.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Figures saved  |  ✓ Metrics saved")
    tf.keras.backend.clear_session()
    return metrics_df


# ── Run all 8 models ──────────────────────────────────────────────────────────

print("\n\n── Per-Model Evaluation ────────────────────────────────────────────")

all_metrics = []

phase_registry = [
    ("3dcnn",     PHASE_MODEL_PATHS["3dcnn"],     "PhaseContrast", phase_val_paths, phase_val_labels),
    ("alexnet",   PHASE_MODEL_PATHS["alexnet"],   "PhaseContrast", phase_val_paths, phase_val_labels),
    ("mobilenet", PHASE_MODEL_PATHS["mobilenet"], "PhaseContrast", phase_val_paths, phase_val_labels),
    ("effnet",    PHASE_MODEL_PATHS["effnet"],    "PhaseContrast", phase_val_paths, phase_val_labels),
]

dark_registry = [
    ("3dcnn",     DARK_MODEL_PATHS["3dcnn"],     "DarkField", dark_val_paths, dark_val_labels),
    ("alexnet",   DARK_MODEL_PATHS["alexnet"],   "DarkField", dark_val_paths, dark_val_labels),
    ("mobilenet", DARK_MODEL_PATHS["mobilenet"], "DarkField", dark_val_paths, dark_val_labels),
    ("effnet",    DARK_MODEL_PATHS["effnet"],    "DarkField", dark_val_paths, dark_val_labels),
]

for model_key, path, modality_tag, val_paths, val_labels in (phase_registry + dark_registry):
    metrics = evaluate_model(model_key, path, modality_tag, val_paths, val_labels)
    all_metrics.append(metrics)

summary_df = pd.concat(all_metrics, ignore_index=True)
summary_df.to_csv(f"{OUTPUT_ROOT}/evaluations/all_models_metrics_summary.csv", index=False)

print("\n✓ All 8 models evaluated.")
print(summary_df[["Model", "Accuracy", "F1_Macro", "AUC_Macro"]].to_string(index=False))


# ── Section 6: Voting Ensemble ────────────────────────────────────────────────
#
# Majority hard vote across the 4 per-modality models.
# Soft average of class probabilities used for ROC/PR curves.

def voting_ensemble(modality_tag, model_keys, val_paths, val_labels):
    """
    Majority voting ensemble for one modality.

    Args:
        modality_tag : 'PhaseContrast' or 'DarkField'
        model_keys   : list of model key strings (must match saved CSV names)
        val_paths    : validation video paths
        val_labels   : integer ground-truth labels

    Returns:
        One-row metrics DataFrame for the ensemble.
    """
    print(f"\n{'='*70}")
    print(f"  VOTING ENSEMBLE : {modality_tag}")
    print(f"{'='*70}")

    # Load saved prediction CSVs (avoids reloading all 4 models into memory)
    preds_list, probs_list = [], []
    for key in model_keys:
        csv = f"{OUTPUT_ROOT}/predictions_csv/{modality_tag}_{key}_predictions.csv"
        df  = pd.read_csv(csv)
        preds_list.append(df["pred_idx"].values)
        probs_list.append(df[["prob_Control", "prob_H2O2", "prob_RAP"]].values)

    # Hard majority vote
    preds_array    = np.array(preds_list)          # (4, N)
    ensemble_preds = []
    for col in range(preds_array.shape[1]):
        votes = preds_array[:, col]
        ensemble_preds.append(int(np.argmax(np.bincount(votes, minlength=3))))
    ensemble_preds = np.array(ensemble_preds)

    # Soft average probabilities (used for AUC / ROC / PR)
    ensemble_probs = np.mean(probs_list, axis=0)  # (N, 3)

    y_true = np.array(val_labels)
    y_pred = ensemble_preds
    y_prob = ensemble_probs

    acc                      = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _  = precision_recall_fscore_support(y_true, y_pred, average='macro',    zero_division=0)
    p_cls, r_cls, f1_cls, _  = precision_recall_fscore_support(y_true, y_pred, average=None,       zero_division=0, labels=[0,1,2])
    auc_cls                  = [roc_auc_score((y_true == i).astype(int), y_prob[:, i]) for i in range(3)]
    auc_mac                  = float(np.mean(auc_cls))
    cm                       = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"  Accuracy : {acc:.4f}  |  F1-Macro : {f1_mac:.4f}  |  AUC-Macro : {auc_mac:.4f}")

    tag = f"{modality_tag}_VotingEnsemble"

    # Save prediction CSV
    pd.DataFrame({
        "video_path"      : [os.path.basename(p) for p in val_paths],
        "true_label"      : [CFG.classes[l] for l in val_labels],
        "predicted_label" : [CFG.classes[p] for p in y_pred],
        "true_idx"        : val_labels,
        "pred_idx"        : y_pred,
        "prob_Control"    : y_prob[:, 0],
        "prob_H2O2"       : y_prob[:, 1],
        "prob_RAP"        : y_prob[:, 2],
    }).to_csv(f"{OUTPUT_ROOT}/predictions_csv/{tag}_predictions.csv", index=False)

    # Metrics CSV
    metrics_df = pd.DataFrame({
        "Model"       : [tag],
        "Modality"    : ["Ensemble"],
        "Accuracy"    : [acc],
        "F1_Macro"    : [f1_mac],
        "AUC_Macro"   : [auc_mac],
        "Control_F1"  : [f1_cls[0]], "Control_AUC" : [auc_cls[0]],
        "H2O2_F1"     : [f1_cls[1]], "H2O2_AUC"   : [auc_cls[1]],
        "RAP_F1"      : [f1_cls[2]], "RAP_AUC"    : [auc_cls[2]],
    })
    metrics_df.to_csv(f"{OUTPUT_ROOT}/evaluations/{tag}_metrics.csv", index=False)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CFG.classes, yticklabels=CFG.classes,
                ax=ax, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12, weight='bold')
    ax.set_ylabel('True',      fontsize=12, weight='bold')
    ax.set_title(f'Confusion Matrix — {tag}', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{tag}_cm.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── ROC Curves ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for i, cls in enumerate(CFG.classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], linewidth=2.5,
                label=f'{cls} (AUC={auc_cls[i]:.3f})', alpha=0.85)
    ax.plot([0,1], [0,1], 'k--', linewidth=1.5, alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=12, weight='bold')
    ax.set_title(f'ROC Curves — {tag}', fontsize=13, weight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{tag}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── PR Curves ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for i, cls in enumerate(CFG.classes):
        prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
        ap = average_precision_score((y_true == i).astype(int), y_prob[:, i])
        ax.plot(rec, prec, color=CLASS_COLORS[cls], linewidth=2.5,
                label=f'{cls} (AP={ap:.3f})', alpha=0.85)
    ax.set_xlabel('Recall',    fontsize=12, weight='bold')
    ax.set_ylabel('Precision', fontsize=12, weight='bold')
    ax.set_title(f'PR Curves — {tag}', fontsize=13, weight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{tag}_pr.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Figures saved  |  ✓ Metrics saved")
    return metrics_df


print("\n\n── Voting Ensembles ─────────────────────────────────────────────────")

phase_ensemble_metrics = voting_ensemble(
    modality_tag = "PhaseContrast",
    model_keys   = ["3dcnn", "alexnet", "mobilenet", "effnet"],
    val_paths    = phase_val_paths,
    val_labels   = phase_val_labels
)

dark_ensemble_metrics = voting_ensemble(
    modality_tag = "DarkField",
    model_keys   = ["3dcnn", "alexnet", "mobilenet", "effnet"],
    val_paths    = dark_val_paths,
    val_labels   = dark_val_labels
)


# ── Section 7: Universal Predictor (Modality Detector + Meta-Learner) ─────────
#
# Pipeline:
#   1. Modality detector CNN  →  classifies a single frame as Phase or Dark
#   2. Route to the matching modality's 4 base models
#   3. Concatenate their 3-class probability outputs  →  12-feature vector
#   4. Sklearn meta-learner  →  final class prediction
#
# Evaluated on the combined Phase + Dark validation set.

def detect_modality(video_path, modality_detector):
    """
    Extract the middle frame and classify it as 'phase' or 'dark'.

    Args:
        video_path        : path to video file
        modality_detector : loaded Keras binary classifier (sigmoid output)

    Returns:
        modality   : 'dark' if sigmoid output > 0.5, else 'phase'
        confidence : scalar confidence in [0.5, 1.0]
    """
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid   = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return 'phase', 0.5   # fallback default

    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)

    score    = float(modality_detector.predict(frame, verbose=0)[0][0])
    modality = 'dark' if score > 0.5 else 'phase'
    conf     = score if score > 0.5 else (1.0 - score)
    return modality, conf


def run_universal_predictor():
    """
    Full evaluation of the Universal Predictor system.

    Loads:
      - modality_detector.keras
      - phase_meta_learner_v2.pkl  (sklearn; input: 4 models × 3 probs = 12 features)
      - dark_meta_learner_v2.pkl
      - all 8 base models

    Evaluates on combined Phase + Dark validation set.
    Reports modality detection accuracy alongside classification accuracy.
    """
    print(f"\n{'='*70}")
    print(f"  UNIVERSAL PREDICTOR  (Modality Detector + Meta-Learner)")
    print(f"{'='*70}")

    # ── Load all components ───────────────────────────────────────────────────
    print("\n  Loading modality detector ...")
    modality_detector = tf.keras.models.load_model(MODALITY_DET, compile=False)
    print("  ✓ Modality detector loaded")

    print("  Loading meta-learners ...")
    with open(f"{META_MODELS_DIR}/phase_meta_learner_v2.pkl", 'rb') as f:
        phase_meta = pickle.load(f)
    with open(f"{META_MODELS_DIR}/dark_meta_learner_v2.pkl", 'rb') as f:
        dark_meta  = pickle.load(f)
    print("  ✓ Meta-learners loaded")

    print("  Loading all 8 base models ...")
    phase_base, dark_base = {}, {}

    for key, path in PHASE_MODEL_PATHS.items():
        phase_base[key] = load_model_safe(f"phase_{key}", path)

    for key, path in DARK_MODEL_PATHS.items():
        dark_base[key] = load_model_safe(f"dark_{key}", path)

    print("  ✓ All 8 base models loaded")

    # ── Combined validation set ───────────────────────────────────────────────
    combined_paths     = phase_val_paths  + dark_val_paths
    combined_labels    = phase_val_labels + dark_val_labels
    combined_true_mods = ['phase'] * len(phase_val_paths) + ['dark'] * len(dark_val_paths)

    print(f"\n  Combined val set : {len(combined_paths)} videos")
    print(f"    Phase : {len(phase_val_paths)}  |  Dark : {len(dark_val_paths)}")

    # ── Inference loop ────────────────────────────────────────────────────────
    all_preds, all_probs, detected_mods = [], [], []

    for i, path in enumerate(combined_paths):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(combined_paths)} ...")

        # Step 1 — detect modality from middle frame
        det_mod, _ = detect_modality(path, modality_detector)
        detected_mods.append(det_mod)

        # Step 2 — load video frames
        frames = frames_from_video_file(path, deterministic=True)
        batch  = np.expand_dims(frames, axis=0)

        # Step 3 — route through 4 base models, build 12-d feature vector
        base = phase_base if det_mod == 'phase' else dark_base
        meta = phase_meta if det_mod == 'phase' else dark_meta

        feat_vec = []
        for m in base.values():
            feat_vec.extend(m.predict(batch, verbose=0)[0])

        meta_input = np.array(feat_vec).reshape(1, -1)   # (1, 12)

        # Step 4 — meta-learner final prediction
        final_pred  = int(meta.predict(meta_input)[0])
        final_probs = meta.predict_proba(meta_input)[0]

        all_preds.append(final_pred)
        all_probs.append(final_probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ── Modality detection accuracy ───────────────────────────────────────────
    mod_correct = sum(d == t for d, t in zip(detected_mods, combined_true_mods))
    mod_acc     = mod_correct / len(combined_true_mods)
    print(f"\n  Modality detection : {mod_correct}/{len(combined_paths)} = {mod_acc*100:.2f}%")

    # ── Classification metrics ────────────────────────────────────────────────
    y_true = np.array(combined_labels)
    y_pred = all_preds
    y_prob = all_probs

    acc                      = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _  = precision_recall_fscore_support(y_true, y_pred, average='macro',    zero_division=0)
    p_cls, r_cls, f1_cls, _  = precision_recall_fscore_support(y_true, y_pred, average=None,       zero_division=0, labels=[0,1,2])
    auc_cls                  = [roc_auc_score((y_true == i).astype(int), y_prob[:, i]) for i in range(3)]
    auc_mac                  = float(np.mean(auc_cls))
    cm                       = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"  Accuracy : {acc:.4f}  |  F1-Macro : {f1_mac:.4f}  |  AUC-Macro : {auc_mac:.4f}")
    print(f"  Per-class:")
    for i, cls in enumerate(CFG.classes):
        print(f"    {cls:10s}  P={p_cls[i]:.4f}  R={r_cls[i]:.4f}"
              f"  F1={f1_cls[i]:.4f}  AUC={auc_cls[i]:.4f}")

    tag = "UniversalPredictor_MetaLearner"

    # Save prediction CSV
    pd.DataFrame({
        "video_path"       : [os.path.basename(p) for p in combined_paths],
        "true_label"       : [CFG.classes[l] for l in combined_labels],
        "predicted_label"  : [CFG.classes[p] for p in y_pred],
        "true_idx"         : combined_labels,
        "pred_idx"         : y_pred,
        "detected_modality": detected_mods,
        "true_modality"    : combined_true_mods,
        "prob_Control"     : y_prob[:, 0],
        "prob_H2O2"        : y_prob[:, 1],
        "prob_RAP"         : y_prob[:, 2],
    }).to_csv(f"{OUTPUT_ROOT}/predictions_csv/{tag}_predictions.csv", index=False)

    metrics_df = pd.DataFrame({
        "Model"                  : [tag],
        "Modality"               : ["Universal"],
        "Accuracy"               : [acc],
        "F1_Macro"               : [f1_mac],
        "AUC_Macro"              : [auc_mac],
        "Modality_Detection_Acc" : [mod_acc],
        "Control_F1"  : [f1_cls[0]], "Control_AUC" : [auc_cls[0]],
        "H2O2_F1"     : [f1_cls[1]], "H2O2_AUC"   : [auc_cls[1]],
        "RAP_F1"      : [f1_cls[2]], "RAP_AUC"    : [auc_cls[2]],
    })
    metrics_df.to_csv(f"{OUTPUT_ROOT}/evaluations/{tag}_metrics.csv", index=False)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CFG.classes, yticklabels=CFG.classes,
                ax=ax, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12, weight='bold')
    ax.set_ylabel('True',      fontsize=12, weight='bold')
    ax.set_title('Confusion Matrix — Universal Predictor + Meta-Learner',
                 fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{tag}_cm.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── ROC Curves ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for i, cls in enumerate(CFG.classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls], linewidth=2.5,
                label=f'{cls} (AUC={auc_cls[i]:.3f})', alpha=0.85)
    ax.plot([0,1], [0,1], 'k--', linewidth=1.5, alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=12, weight='bold')
    ax.set_title('ROC Curves — Universal Predictor + Meta-Learner',
                 fontsize=13, weight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{tag}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── PR Curves ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for i, cls in enumerate(CFG.classes):
        prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
        ap = average_precision_score((y_true == i).astype(int), y_prob[:, i])
        ax.plot(rec, prec, color=CLASS_COLORS[cls], linewidth=2.5,
                label=f'{cls} (AP={ap:.3f})', alpha=0.85)
    ax.set_xlabel('Recall',    fontsize=12, weight='bold')
    ax.set_ylabel('Precision', fontsize=12, weight='bold')
    ax.set_title('PR Curves — Universal Predictor + Meta-Learner',
                 fontsize=13, weight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_ROOT}/figures/{tag}_pr.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ Figures saved  |  ✓ Metrics saved")
    tf.keras.backend.clear_session()
    return metrics_df


print("\n\n── Universal Predictor ──────────────────────────────────────────────")
universal_metrics = run_universal_predictor()


# ── Section 8: Final Summary Table ───────────────────────────────────────────

print("\n\n── Final Summary ────────────────────────────────────────────────────")

final_df = pd.concat([
    summary_df[["Model", "Modality", "Accuracy", "F1_Macro", "AUC_Macro"]],
    phase_ensemble_metrics[["Model", "Modality", "Accuracy", "F1_Macro", "AUC_Macro"]],
    dark_ensemble_metrics[["Model", "Modality", "Accuracy", "F1_Macro", "AUC_Macro"]],
    universal_metrics[["Model", "Modality", "Accuracy", "F1_Macro", "AUC_Macro"]],
], ignore_index=True)

final_df["Accuracy_%"] = (final_df["Accuracy"] * 100).round(2)
final_df["F1_Macro_%"] = (final_df["F1_Macro"] * 100).round(2)
final_df["AUC_Macro_%"]= (final_df["AUC_Macro"] * 100).round(2)

final_df.to_csv(f"{OUTPUT_ROOT}/evaluations/FINAL_summary_all_models.csv", index=False)

print(final_df[["Model", "Accuracy_%", "F1_Macro_%", "AUC_Macro_%"]].to_string(index=False))
print(f"\n✓ FINAL_summary_all_models.csv  →  {OUTPUT_ROOT}/evaluations/")
print("\n✓ CytoMorpheus evaluation pipeline complete.")
