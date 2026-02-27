# =============================================================================
# CytoMorpheus — Dark Field AlexNet-BiLSTM Training
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Architecture : AlexNet (custom conv trunk) + Bidirectional LSTM
#                Per-frame features extracted by AlexNet via TimeDistributed,
#                temporal dynamics modelled by Bi-LSTM
# Modality     : Dark Field microscopy
# Input        : 30 frames × 224×224 × 3 channels per video
# Classes      : Control | H2O2 (Necrosis) | RAP (Apoptosis)
# Dataset      : 3,802 videos  →  Train: 3,041 | Val: 761
# Result       : 90.67% accuracy | F1-Macro: 89.47% | AUC-Macro: 97.12%
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import glob
import random
import collections

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Rescaling, TimeDistributed, GlobalAveragePooling2D,
    Dropout, Bidirectional, LSTM, Dense
)
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


# ── Section 5: Model Architecture — AlexNet-BiLSTM ───────────────────────────

# Hyperparameters
FINE_TUNE_AT  = 50    # unfreeze last N layers of AlexNet conv trunk
DROPOUT_RATE  = 0.4   # dropout before and after Bi-LSTM
LSTM_UNITS    = 128   # units in the Bidirectional LSTM
LEARNING_RATE = 1e-4


def build_alexnet_conv(input_shape):
    """
    Custom AlexNet convolutional trunk.
    5 conv layers with BatchNorm and MaxPooling.
    Outputs spatial feature maps per frame.
    """
    inp = Input(shape=input_shape)

    x = Conv2D(96, kernel_size=11, strides=4, padding='valid')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    x = Conv2D(256, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    x = Conv2D(384, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(384, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    return models.Model(inp, x, name='alexnet_conv')


def build_alexnet_video(input_shape, n_frames, num_classes):
    """
    Full video classifier:
      1. AlexNet conv trunk applied per-frame via TimeDistributed
      2. GlobalAveragePooling2D per frame → feature vector sequence
      3. Bidirectional LSTM for temporal modelling
      4. Dropout + Dense(softmax) classification head

    Rescaling converts [0,1] inputs to [0,255] to match
    AlexNet's expected input range.
    """
    conv_base = build_alexnet_conv(input_shape)
    conv_base.trainable = True
    if FINE_TUNE_AT is not None:
        for layer in conv_base.layers[:-FINE_TUNE_AT]:
            layer.trainable = False

    video_in = Input(shape=(n_frames, *input_shape), name='video_input')

    x = Rescaling(scale=255.0)(video_in)
    x = TimeDistributed(conv_base,                name='td_alexnet')(x)
    x = TimeDistributed(GlobalAveragePooling2D(), name='td_gap2d')(x)
    x = Dropout(DROPOUT_RATE,                     name='drop_pre_lstm')(x)
    x = Bidirectional(
            LSTM(LSTM_UNITS, return_sequences=False),
            name='bi_lstm'
        )(x)
    x = Dropout(DROPOUT_RATE, name='drop_post_lstm')(x)
    output = Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(video_in, output, name='AlexNetVideo_BiLSTM_Dark')


model = build_alexnet_video(
    input_shape=(*CFG.output_size, 3),
    n_frames=CFG.n_frames,
    num_classes=len(CFG.classes)
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ── Section 6: Callback — Model Checkpoint ───────────────────────────────────

save_dir = '/content/drive/MyDrive/Cell Death/Latest Models/Dark/AlexNetVideo'
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(
    save_dir,
    'alexnet_epoch{epoch:02d}_valAcc{val_accuracy:.2f}.weights.h5'
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


# ── Section 7: Training ───────────────────────────────────────────────────────

history_alex = model.fit(
    train_ds,
    epochs=CFG.epochs,
    validation_data=val_ds,
    callbacks=[checkpoint_cb]
)

print("\nTraining completed.")
print(f"Final train acc : {history_alex.history['accuracy'][-1]:.4f}")
print(f"Final val   acc : {history_alex.history['val_accuracy'][-1]:.4f}")
print(f"Best model saved to: {save_dir}")
