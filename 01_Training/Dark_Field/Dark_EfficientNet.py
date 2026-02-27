# =============================================================================
# CytoMorpheus — Dark Field EfficientNetB0 + Transformer Training
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Architecture : EfficientNetB0 (ImageNet pretrained) + Transformer
#                Per-frame features by EfficientNetB0 via TimeDistributed,
#                positional embedding added, Multi-Head Self-Attention
#                for temporal modelling
# Modality     : Dark Field microscopy
# Input        : 30 frames × 224×224 × 3 channels per video
# Classes      : Control | H2O2 (Necrosis) | RAP (Apoptosis)
# Dataset      : 3,802 videos  →  Train: 3,041 | Val: 761
# Training     : Two-phase — 20-epoch warmup then main training with callbacks
# Extra        : Label smoothing (0.05), class weights, Adam clipnorm=1.0,
#                AUC tracked during training, .keras checkpoint format
# Result       : 91.85% accuracy | F1-Macro: 91.15% | AUC-Macro: 98.17%
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import glob
import random
import collections
import json

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class CFG:
    data_dir    = "/content/drive/MyDrive/Cell Death/dark field"
    classes     = ["Control", "H2O2", "RAP"]
    n_frames    = 30
    frame_step  = 2          # frames to skip between samples
    output_size = (224, 224) # (height, width)
    batch_size  = 16
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
    """
    tf.py_function wrapper around the NumPy sampler.
    Labels are one-hot encoded float32 (required by CategoricalCrossentropy).
    """
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
         frame via TimeDistributed → produces a (1280,) feature vector per frame
      2. Learnable positional embedding added to frame features
      3. Multi-Head Self-Attention (4 heads, key_dim = D//2)
      4. GlobalAveragePooling1D + Dropout(0.5)
      5. Dense(softmax) classification head

    Training strategy:
      - Label smoothing (0.05) to reduce overconfidence
      - Class weights to address class imbalance in dark field data
      - Adam with clipnorm=1.0 for stable Transformer training
      - AUC metric tracked alongside accuracy
      - Two-phase training: 20-epoch warmup without EarlyStopping,
        then main phase with full callbacks
    """
    inputs = layers.Input(shape=input_shape, name="video_input")

    # 1. EfficientNetB0 per-frame feature extraction
    effnet = applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape[1:],   # (H, W, C)
        pooling="avg"
    )
    effnet.trainable = True

    frame_feats = layers.TimeDistributed(effnet, name="td_effnet")(inputs)
    # frame_feats shape: (batch, n_frames, 1280)

    # 2. Positional embedding over the time dimension
    T   = input_shape[0]
    D   = frame_feats.shape[-1]
    pos = layers.Embedding(input_dim=T, output_dim=D, name="pos_emb")(tf.range(T))
    pos = tf.expand_dims(pos, 0)   # (1, T, D) — broadcasts over batch
    x   = layers.Add(name="add_pos")([frame_feats, pos])

    # 3. Temporal Transformer encoder (single Multi-Head Attention block)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=D // 2, name="mha")(x, x)

    # 4. Temporal pooling + regularisation
    x = layers.GlobalAveragePooling1D(name="gap1d")(x)
    x = layers.Dropout(0.5, name="drop")(x)

    # 5. Classification head
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="cls_head")(x)

    return models.Model(inputs, outputs, name="EffNetB0_Transformer_Video_Dark")


dark_effnet = build_efficientnet_transformer_model(
    input_shape=(CFG.n_frames, CFG.output_size[1], CFG.output_size[0], 3),
    num_classes=len(CFG.classes)
)

dark_effnet.summary()


# ── Section 7: Compile, Class Weights & Two-Phase Training ───────────────────

# Paths
base_out    = "/content/drive/MyDrive/Cell_apoptosis_necrosis/dark_efficient"
model_name  = "effnetb0_partialft_transformer"
results_dir = os.path.join(base_out, model_name)
os.makedirs(results_dir, exist_ok=True)

# Optimizer — Adam with gradient clipping for stable Transformer training
opt  = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

# Loss — label smoothing (0.05) reduces overconfidence on dark field data
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

dark_effnet.compile(
    optimizer=opt,
    loss=loss,
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", multi_label=False)
    ]
)

# Callbacks
ckpt_path = os.path.join(results_dir, "best_model.keras")  # .keras for Keras 3

checkpoint_cb  = ModelCheckpoint(
    filepath=ckpt_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)
csv_logger_cb  = CSVLogger(
    os.path.join(results_dir, "training_log.csv"),
    append=True      # append=True supports resumed training
)
earlystop_cb   = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb   = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
tensorboard_cb = TensorBoard(
    log_dir=os.path.join(results_dir, "logs"),
    histogram_freq=0
)

print(f"Saving to       : {results_dir}")
print(f"Checkpoint path : {ckpt_path}")

# Class weights — compensate for class imbalance in dark field dataset
try:
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CFG.classes)),
        y=np.array(train_labels)
    )
    CLASS_WEIGHTS = {i: float(w) for i, w in enumerate(class_weights_arr)}
    print("Class weights:", CLASS_WEIGHTS)
except Exception as e:
    print(f"[WARN] Class weights not computed, proceeding without: {e}")
    CLASS_WEIGHTS = None

# Save class weights for reference
with open(os.path.join(results_dir, "class_weights.json"), "w") as f:
    json.dump(
        {"class_weights": CLASS_WEIGHTS, "classes": CFG.classes},
        f, indent=2
    )

# ── Two-Phase Training ────────────────────────────────────────────────────────
#
# Phase 1 — Warmup (20 epochs, no EarlyStopping):
#   Allows Transformer attention weights and the EfficientNet backbone to
#   stabilise before the main training loop begins.
#   Only CSVLogger and TensorBoard are active.
#
# Phase 2 — Main training (remaining epochs, full callbacks):
#   Picks up from where warmup ended using initial_epoch.
#   EarlyStopping, ReduceLR, and checkpoint active.

WARMUP_EPOCHS = 20
TOTAL_EPOCHS  = CFG.epochs

print(f"\nPhase 1 — Warmup ({WARMUP_EPOCHS} epochs, no early stopping)...")
history_warm = dark_effnet.fit(
    train_ds,
    epochs=WARMUP_EPOCHS,
    validation_data=val_ds,
    class_weight=CLASS_WEIGHTS,
    callbacks=[csv_logger_cb, tensorboard_cb],
    verbose=1
)

print(f"\nPhase 2 — Main training (epochs {WARMUP_EPOCHS}–{TOTAL_EPOCHS})...")
history_main = dark_effnet.fit(
    train_ds,
    initial_epoch=history_warm.epoch[-1] + 1,
    epochs=TOTAL_EPOCHS,
    validation_data=val_ds,
    class_weight=CLASS_WEIGHTS,
    callbacks=[checkpoint_cb, csv_logger_cb,
               earlystop_cb, reduce_lr_cb, tensorboard_cb],
    verbose=1
)

best_val_acc = max(
    (history_warm.history.get("val_accuracy") or [0]) +
    (history_main.history.get("val_accuracy") or [0])
)

print(f"\nTraining complete.")
print(f"Best val_accuracy : {best_val_acc:.4f}")
print(f"Best model saved to : {ckpt_path}")
