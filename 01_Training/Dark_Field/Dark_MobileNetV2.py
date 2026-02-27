# =============================================================================
# CytoMorpheus — Dark Field MobileNetV2 Training
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Architecture : MobileNetV2 (ImageNet pretrained, partial fine-tuning)
#                Per-frame features via TimeDistributed MobileNetV2,
#                temporal pooling via GlobalAveragePooling1D
# Modality     : Dark Field microscopy
# Input        : 30 frames × 224×224 × 3 channels per video
# Classes      : Control | H2O2 (Necrosis) | RAP (Apoptosis)
# Dataset      : 3,802 videos  →  Train: 3,041 | Val: 761
# Result       : 88.17% accuracy | F1-Macro: 87.31% | AUC-Macro: 96.88%
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import glob
import random
import collections

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
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
    result       = []
    src          = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length  = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        start = random.randint(0, int(video_length - need_length))

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    if not ret:
        src.release()
        return np.zeros((n_frames, *output_size, 3), dtype=np.float32)

    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            result.append(format_frames(frame, output_size))
        else:
            result.append(np.zeros_like(result[0]))

    src.release()
    result = np.array(result)[..., [2, 1, 0]]  # BGR → RGB
    return result


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

def video_generator(file_paths, targets, n_frames, output_size, frame_step):
    """
    Python generator — yields (frames, label) pairs one video at a time.
    Labels are integer indices (used with sparse_categorical_crossentropy).
    """
    for path, label in zip(file_paths, targets):
        frames = frames_from_video_file(path, n_frames, output_size, frame_step)
        yield frames, label


output_types  = (tf.float32, tf.int32)
output_shapes = (
    (CFG.n_frames, CFG.output_size[0], CFG.output_size[1], 3),
    ()
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


# ── Section 5: Model Architecture — MobileNetV2 ──────────────────────────────

# Hyperparameters
FINE_TUNE_AT  = 100   # unfreeze last 100 layers of MobileNetV2
DROPOUT_RATE  = 0.5   # dropout after temporal pooling
LEARNING_RATE = 5e-5  # low lr for fine-tuning pretrained weights


def build_video_model_mobilenet(input_shape, n_frames, num_classes,
                                 fine_tune_at=None, dropout_rate=0.5):
    """
    Video classifier using MobileNetV2 as a per-frame feature extractor.

    Architecture:
      1. MobileNetV2 (ImageNet pretrained, pooling='avg') applied per frame
         via TimeDistributed
      2. GlobalAveragePooling1D pools the frame-level feature sequence
         into a single video-level descriptor
      3. Dropout + Dense(softmax) classification head

    Partial fine-tuning: all layers trainable; layers before the last
    `fine_tune_at` layers are frozen to preserve low-level ImageNet features.
    """
    # Load MobileNetV2 backbone (no top, global avg pooling built-in)
    base_cnn = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    # Partial fine-tuning
    base_cnn.trainable = True
    if fine_tune_at is not None:
        for layer in base_cnn.layers[:-fine_tune_at]:
            layer.trainable = False

    # Video-level input: (n_frames, H, W, 3)
    video_input = layers.Input(shape=(n_frames, *input_shape))

    # Apply CNN to each frame independently
    x = layers.TimeDistributed(base_cnn)(video_input)

    # Temporal pooling across all frames
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Classification head
    output = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=video_input, outputs=output,
                        name="Video_MobileNetV2_Dark")


input_shape = (*CFG.output_size, 3)
model = build_video_model_mobilenet(
    input_shape=input_shape,
    n_frames=CFG.n_frames,
    num_classes=len(CFG.classes),
    fine_tune_at=FINE_TUNE_AT,
    dropout_rate=DROPOUT_RATE
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"]
)

model.summary()


# ── Section 6: Callback — Model Checkpoint ───────────────────────────────────

save_dir = '/content/drive/MyDrive/Cell Death/30 frames models/dark'
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, 'MobileNetV2_best.h5')

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

print(f"Best model will be saved to: {checkpoint_path}")


# ── Section 7: Training ───────────────────────────────────────────────────────

history = model.fit(
    train_ds,
    epochs=CFG.epochs,
    validation_data=val_ds,
    callbacks=[checkpoint]
)

print("\nTraining completed.")
print(f"Final train acc : {history.history['accuracy'][-1]:.4f}")
print(f"Final val   acc : {history.history['val_accuracy'][-1]:.4f}")
print(f"Best model saved to: {checkpoint_path}")
