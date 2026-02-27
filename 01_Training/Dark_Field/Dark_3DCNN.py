# =============================================================================
# CytoMorpheus — Dark Field 3D-CNN Training
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Architecture : 3D-CNN (3 convolutional blocks + GlobalAveragePooling3D)
# Loss         : Focal loss (gamma=2.0, alpha=0.25)
# Modality     : Dark Field microscopy
# Input        : 30 frames × 224×224 × 3 channels per video
# Classes      : Control | H2O2 (Necrosis) | RAP (Apoptosis)
# Dataset      : 3,802 videos  →  Train: 3,041 | Val: 761
# Result       : 94.22% accuracy | F1-Macro: 93.93% | AUC-Macro: 99.34%
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import glob
import random
import collections

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


class CFG:
    epochs      = 50
    batch_size  = 32
    classes     = ["Control", "H2O2", "RAP"]
    n_frames    = 30
    frame_step  = 2
    output_size = (224, 224)

# Dataset root on Google Drive
base_path = "/content/drive/MyDrive/Cell Death/dark field"

print("CFG:")
print(f"  base_path   = {base_path}")
print(f"  classes     = {CFG.classes}")
print(f"  n_frames    = {CFG.n_frames}")
print(f"  frame_step  = {CFG.frame_step}")
print(f"  output_size = {CFG.output_size}")
print(f"  batch_size  = {CFG.batch_size}")
print(f"  epochs      = {CFG.epochs}")


# ── Section 2: Frame Sampler ──────────────────────────────────────────────────

def format_frames(frame, output_size):
    """Convert a BGR uint8 frame to a float32 tensor, resized with padding."""
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames,
                           output_size=(224, 224),
                           frame_step=CFG.frame_step):
    """
    Sample n_frames evenly-stepped frames from a video file.
    Randomly selects a start position each call (training augmentation).
    Returns float32 array of shape (n_frames, H, W, 3) in RGB order.
    Pads with zeros if the video is shorter than required.
    """
    h, w         = output_size
    src          = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length  = 1 + (n_frames - 1) * frame_step

    start = 0 if need_length > video_length else random.randint(
        0, int(video_length - need_length)
    )
    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    result = []
    for _ in range(n_frames):
        ret, frame = src.read()
        if not ret:
            result.append(np.zeros((h, w, 3), dtype=np.float32))
        else:
            frame = cv2.resize(frame, (w, h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            result.append(frame)
        for _ in range(frame_step - 1):
            src.grab()

    src.release()
    return np.stack(result)   # (n_frames, H, W, 3)


# ── Section 3: Load File Paths & Stratified Split ────────────────────────────

file_paths = []
targets    = []

for i, cls in enumerate(CFG.classes):
    cls_files = glob.glob(f"{base_path}/{cls}/*.avi")
    file_paths.extend(cls_files)
    targets.extend([i] * len(cls_files))
    print(f"Found {len(cls_files)} videos for class '{cls}'")

targets = np.array(targets)
print(f"\nTotal videos found: {len(file_paths)}")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, targets,
    test_size=0.2,
    stratify=targets,
    random_state=42
)

print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")
print("Train distribution:", dict(collections.Counter(train_labels)))
print("Val   distribution:", dict(collections.Counter(val_labels)))


# ── Section 4: Generator & TensorFlow Dataset Pipeline ───────────────────────

n_classes = len(CFG.classes)


def video_generator(file_paths, targets, n_frames, output_size, frame_step):
    """
    Python generator — yields (frames, one_hot_label) pairs.
    Labels are one-hot encoded float32 vectors (required by focal loss).
    """
    for path, label in zip(file_paths, targets):
        frames         = frames_from_video_file(path, n_frames, output_size, frame_step)
        label_one_hot  = tf.one_hot(label, depth=n_classes)
        yield frames, label_one_hot


output_types  = (tf.float32, tf.float32)
output_shapes = (
    (CFG.n_frames, CFG.output_size[0], CFG.output_size[1], 3),
    (n_classes,)
)

train_ds = tf.data.Dataset.from_generator(
    lambda: video_generator(
        train_paths, train_labels,
        CFG.n_frames, CFG.output_size, CFG.frame_step
    ),
    output_types=output_types,
    output_shapes=output_shapes
).shuffle(100).batch(CFG.batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: video_generator(
        val_paths, val_labels,
        CFG.n_frames, CFG.output_size, CFG.frame_step
    ),
    output_types=output_types,
    output_shapes=output_shapes
).batch(CFG.batch_size).prefetch(tf.data.AUTOTUNE)


# ── Section 5: Focal Loss ─────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for multi-class classification.
    Addresses class imbalance by down-weighting easy examples.

    Args:
        gamma : focusing parameter — higher values penalise easy examples more
        alpha : class weight balancing factor
    Returns:
        loss function compatible with model.compile()
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon       = K.epsilon()
        y_true        = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred        = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_pred        = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight        = alpha * tf.math.pow(1.0 - y_pred, gamma)
        loss          = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return focal_loss_fixed


# ── Section 6: Model Architecture ────────────────────────────────────────────

def build_3dcnn_model(input_shape, n_classes):
    """
    3-block 3D-CNN with GlobalAveragePooling3D classification head.

    Block filters : 32 → 64 → 128
    Pooling       : (1,2,2) → (2,2,2) → (2,2,2)
    Head          : GlobalAveragePooling3D → Dense(256, relu) → Dropout(0.5)
                    → Dense(n_classes, softmax)
    Loss          : Focal loss (handles class imbalance in dark field data)
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    # Block 2
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 3
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Classification head
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


input_shape = (CFG.n_frames, CFG.output_size[0], CFG.output_size[1], 3)
model = build_3dcnn_model(input_shape, n_classes)

model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

model.summary()


# ── Section 7: Callback — Model Checkpoint ───────────────────────────────────

save_dir = '/content/drive/MyDrive/Cell Death/Latest Models/Dark/3DCNN'
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(
    save_dir,
    '3dcnn_epoch{epoch:02d}_valAcc{val_accuracy:.2f}.weights.h5'
)

checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

print(f"Checkpoints will be saved to: {save_dir}")


# ── Section 8: Training ───────────────────────────────────────────────────────

history_3dcnn = model.fit(
    train_ds,
    epochs=CFG.epochs,
    validation_data=val_ds,
    callbacks=[checkpoint_cb]
)

print("\nTraining completed.")
print(f"Final train acc : {history_3dcnn.history['accuracy'][-1]:.4f}")
print(f"Final val   acc : {history_3dcnn.history['val_accuracy'][-1]:.4f}")
print(f"Best model saved to: {save_dir}")
