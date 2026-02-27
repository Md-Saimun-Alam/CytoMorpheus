# =============================================================================
# CytoMorpheus — Phase Contrast 3D-CNN Training
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Architecture : 3D-CNN (4 convolutional blocks + GlobalAveragePooling3D)
# Modality     : Phase Contrast microscopy
# Input        : 30 frames × 224×224 × 3 channels per video
# Classes      : Control | H2O2 (Necrosis) | RAP (Apoptosis)
# Dataset      : 4,549 videos  →  Train: 3,639 | Val: 910
# Result       : 89.34% accuracy | F1-Macro: 88.85% | AUC-Macro: 98.02%
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import glob
import random
import collections

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split


class CFG:
    data_dir    = "/content/drive/MyDrive/Cell Death/bright field"
    classes     = ["Control", "H2O2", "RAP"]
    n_frames    = 30
    frame_step  = 2          # frames to skip between samples
    output_size = (224, 224) # (height, width)
    batch_size  = 4
    epochs      = 50


print("CFG:")
print(f"  data_dir    = {CFG.data_dir}")
print(f"  classes     = {CFG.classes}")
print(f"  n_frames    = {CFG.n_frames}")
print(f"  frame_step  = {CFG.frame_step}")
print(f"  output_size = {CFG.output_size}")
print(f"  batch_size  = {CFG.batch_size}")
print(f"  epochs      = {CFG.epochs}")


# ── Section 2: Load File Paths & Stratified Split ────────────────────────────

file_paths = []
labels     = []

for idx, cls in enumerate(CFG.classes):
    cls_pattern = os.path.join(CFG.data_dir, cls, "*.avi")
    vids = glob.glob(cls_pattern)
    print(f"Found {len(vids)} videos for class '{cls}'")
    file_paths.extend(vids)
    labels.extend([idx] * len(vids))

print(f"Total videos found: {len(file_paths)}")

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


# ── Section 3: Frame Sampler ──────────────────────────────────────────────────

def frames_from_video_file(video_path, n_frames,
                           output_size=(224, 224),
                           frame_step=CFG.frame_step):
    """
    Extract n_frames evenly-stepped frames from a video file.
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


# ── Section 4: TensorFlow Dataset Pipeline ───────────────────────────────────

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


# ── Section 5: Model Architecture ────────────────────────────────────────────

def build_3dcnn_advanced(input_shape, num_classes,
                          l2_reg=1e-4, dropout_rate=0.3):
    """
    4-block 3D-CNN with BatchNorm, L2 regularisation, and dropout.
    Head: GlobalAveragePooling3D → Dense(512) → softmax(num_classes).

    Block filters : 32 → 64 → 128 → 256
    Pooling       : (1,2,2) → (2,2,2) → (2,2,2) → (2,2,2)
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv3D(32, (3, 3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((1, 2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2
    x = layers.Conv3D(64, (3, 3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 3
    x = layers.Conv3D(128, (3, 3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 4
    x = layers.Conv3D(256, (3, 3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2, 2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Classification head
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           dtype='float32')(x)

    return models.Model(inputs, outputs, name='3D_CNN_Advanced')


input_shape = (CFG.n_frames, CFG.output_size[1], CFG.output_size[0], 3)
model = build_3dcnn_advanced(input_shape, num_classes=len(CFG.classes))

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ── Section 6: Callbacks ─────────────────────────────────────────────────────

base_out    = "/content/drive/MyDrive/Cell Death/bright_field"
model_name  = "3dcnn_advanced"
results_dir = os.path.join(base_out, model_name)
os.makedirs(results_dir, exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(results_dir, "best_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
csv_logger_cb = CSVLogger(
    filename=os.path.join(results_dir, "training_log.csv"),
    append=False
)
earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

print("Callbacks configured:")
print(f"  Checkpoint   : {checkpoint_cb.filepath}")
print(f"  CSV Logger   : {csv_logger_cb.filename}")
print(f"  EarlyStopping: monitor={earlystop_cb.monitor}, patience={earlystop_cb.patience}")
print(f"  ReduceLR     : monitor={reduce_lr_cb.monitor}, factor={reduce_lr_cb.factor}")


# ── Section 7: Training ───────────────────────────────────────────────────────

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG.epochs,
    callbacks=[checkpoint_cb, csv_logger_cb, earlystop_cb, reduce_lr_cb]
)

print("\nTraining completed.")
print(f"Final train acc : {history.history['accuracy'][-1]:.4f}")
print(f"Final val   acc : {history.history['val_accuracy'][-1]:.4f}")
print(f"Best model saved to: {results_dir}")
