# =============================================================================
# CytoMorpheus — Phase Contrast EfficientNetB0 + Transformer Training
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Architecture : EfficientNetB0 (ImageNet pretrained) + Transformer
#                Per-frame features extracted by EfficientNetB0 via
#                TimeDistributed, positional embedding added, then
#                Multi-Head Self-Attention for temporal modelling
# Modality     : Phase Contrast microscopy
# Input        : 30 frames × 224×224 × 3 channels per video
# Classes      : Control | H2O2 (Necrosis) | RAP (Apoptosis)
# Dataset      : 4,549 videos  →  Train: 3,639 | Val: 910
# Training     : Two-phase — 5-epoch warmup then main training with callbacks
# Result       : 95.38% accuracy | F1-Macro: 94.98% | AUC-Macro: 99.52%
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import glob
import random
import collections

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.model_selection import train_test_split


class CFG:
    data_dir    = "/content/drive/MyDrive/Cell Death/bright field"
    classes     = ["Control", "H2O2", "RAP"]
    n_frames    = 30
    frame_step  = 2          # frames to skip between samples
    output_size = (224, 224) # (height, width)
    batch_size  = 8
    epochs      = 50


print("CFG:")
print(f"  data_dir    = {CFG.data_dir}")
print(f"  classes     = {CFG.classes}")
print(f"  n_frames    = {CFG.n_frames}")
print(f"  frame_step  = {CFG.frame_step}")
print(f"  output_size = {CFG.output_size}")
print(f"  batch_size  = {CFG.batch_size}")
print(f"  epochs      = {CFG.epochs}")


# ── Section 2: Load File Paths ────────────────────────────────────────────────

file_paths = []
labels     = []

for idx, cls in enumerate(CFG.classes):
    cls_pattern = os.path.join(CFG.data_dir, cls, "*.avi")
    vids = glob.glob(cls_pattern)
    print(f"Found {len(vids)} videos for class '{cls}'")
    file_paths.extend(vids)
    labels.extend([idx] * len(vids))

print(f"Total videos found: {len(file_paths)}")
for p in file_paths[:3]:
    print("  Sample path:", p)


# ── Section 3: Stratified Split ───────────────────────────────────────────────

from sklearn.model_selection import train_test_split
import collections

train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")
print("Train distribution:", dict(collections.Counter(train_labels)))
print("Val   distribution:", dict(collections.Counter(val_labels)))


# ── Section 4: Frame Sampler (Pure NumPy) ─────────────────────────────────────

def frames_from_video_file(video_path, n_frames,
                           output_size=(224, 224),
                           frame_step=CFG.frame_step):
    """
    Sample n_frames evenly-stepped frames from a video file.
    Randomly selects a start position each call (training augmentation).
    Returns float32 numpy array of shape (n_frames, H, W, 3) in RGB order.
    Pads with zeros if the video is shorter than required.
    """
    h, w  = output_size
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need  = 1 + (n_frames - 1) * frame_step
    start = 0 if need > total else random.randint(0, total - need)
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


def sample_video_frames(video_path, n_frames, output_size):
    return frames_from_video_file(video_path, n_frames, output_size)

# Sanity check on one video
test = frames_from_video_file(
    train_paths[0], CFG.n_frames, CFG.output_size, CFG.frame_step
)
print("Sampled shape:", test.shape,
      "| dtype:", test.dtype,
      "| min/max:", test.min(), "/", test.max())


# ── Section 5: TensorFlow Dataset Pipeline ───────────────────────────────────

def load_video_frames_tf(path, label):
    frames, ohe = tf.py_function(
        func=lambda p, l: (
            sample_video_frames(
                p.numpy().decode('utf-8'), CFG.n_frames, CFG.output_size
            ),
            tf.one_hot(l, depth=len(CFG.classes))
        ),
        inp=[path, label],
        Tout=[tf.float32, tf.float32]
    )
    frames.set_shape((CFG.n_frames, CFG.output_size[1], CFG.output_size[0], 3))
    ohe.set_shape((len(CFG.classes),))
    return frames, ohe


train_ds = (
    tf.data.Dataset
      .from_tensor_slices((train_paths, train_labels))
      .shuffle(len(train_paths), seed=42)
      .map(load_video_frames_tf, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(CFG.batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset
      .from_tensor_slices((val_paths, val_labels))
      .map(load_video_frames_tf, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(CFG.batch_size)
      .prefetch(tf.data.AUTOTUNE)
)

# Verify one batch
for batch_frames, batch_labels in train_ds.take(1):
    print("Batch frames shape:", batch_frames.shape)
    print("Batch labels shape:", batch_labels.shape)


# ── Section 6: Model Architecture — EfficientNetB0 + Transformer ─────────────

def build_efficientnet_transformer_model(input_shape, num_classes):
    """
    Video classifier combining EfficientNetB0 and a Transformer encoder.

    Architecture:
      1. EfficientNetB0 (ImageNet pretrained, pooling='avg') applied per
         frame via TimeDistributed → produces a feature vector per frame
      2. Positional embedding added to frame features to encode temporal order
      3. Multi-Head Self-Attention (4 heads) for temporal modelling
      4. GlobalAveragePooling1D + Dropout
      5. Dense(softmax) classification head

    Two-phase training is used:
      - Warmup (5 epochs): all layers train without EarlyStopping to let
        the Transformer weights stabilise
      - Main phase: full callbacks including EarlyStopping and ReduceLR
    """
    inputs = layers.Input(shape=input_shape)

    # 1. EfficientNetB0 per-frame feature extraction
    effnet = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape[1:],   # (H, W, C)
        pooling='avg'
    )
    effnet.trainable = True

    frame_features = layers.TimeDistributed(effnet)(inputs)
    # frame_features shape: (batch, n_frames, feature_dim)

    # 2. Positional embedding
    n_frames       = input_shape[0]
    feature_dim    = frame_features.shape[-1]
    positions      = tf.range(start=0, limit=n_frames, delta=1)
    pos_embedding  = layers.Embedding(
        input_dim=n_frames,
        output_dim=feature_dim
    )(positions)

    x = frame_features + pos_embedding

    # 3. Multi-Head Self-Attention
    x = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=feature_dim // 2
    )(x, x)

    # 4. Temporal pooling + regularisation
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    # 5. Classification head
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs,
                        name='EfficientNetB0_Transformer')


model = build_efficientnet_transformer_model(
    input_shape=(CFG.n_frames, *CFG.output_size, 3),
    num_classes=len(CFG.classes)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ── Section 7: Callbacks ─────────────────────────────────────────────────────

base_out    = "/content/drive/MyDrive/Cell Death/bright_field"
model_name  = "effnetb0_partialft_transformer"
results_dir = os.path.join(base_out, model_name)
os.makedirs(results_dir, exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(results_dir, "best_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)
csv_logger_cb = CSVLogger(
    filename=os.path.join(results_dir, "training_log.csv"),
    separator=',',
    append=True      # append=True supports resumed training
)
earlystop_cb = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,     # longer patience for Transformer
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
tensorboard_cb = TensorBoard(
    log_dir=os.path.join(results_dir, 'logs'),
    histogram_freq=1,
    profile_batch='500,520'
)

print(f"Callbacks configured → {results_dir}")


# ── Section 8: Two-Phase Training ────────────────────────────────────────────
#
# Phase 1 — Warmup (5 epochs, no EarlyStopping):
#   Allows Transformer attention weights to stabilise before the main
#   training loop begins. Only CSVLogger and TensorBoard are active.
#
# Phase 2 — Main training (remaining epochs, full callbacks):
#   Picks up from where warmup ended using initial_epoch.

WARMUP_EPOCHS = 5
TOTAL_EPOCHS  = CFG.epochs

print(f"\nPhase 1 — Warmup ({WARMUP_EPOCHS} epochs, no early stopping)...")
history_warmup = model.fit(
    train_ds,
    epochs=WARMUP_EPOCHS,
    validation_data=val_ds,
    callbacks=[csv_logger_cb, tensorboard_cb]
)

print(f"\nPhase 2 — Main training (epochs {WARMUP_EPOCHS}–{TOTAL_EPOCHS})...")
history = model.fit(
    train_ds,
    initial_epoch=history_warmup.epoch[-1] + 1,
    epochs=TOTAL_EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, csv_logger_cb,
               earlystop_cb, reduce_lr_cb, tensorboard_cb]
)

best_val_acc = max(
    history_warmup.history['val_accuracy'] +
    history.history['val_accuracy']
)

print(f"\nTraining complete.")
print(f"Best val_accuracy : {best_val_acc:.4f}")
print(f"Model saved to    : {results_dir}")
