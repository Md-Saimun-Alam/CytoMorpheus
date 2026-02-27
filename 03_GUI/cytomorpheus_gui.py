# =============================================================================
# CytoMorpheus — Gradio GUI Application
# =============================================================================
# Paper : Multimodal Spatiotemporal Deep Learning for Real-Time Classification
#         of Apoptosis and Necrosis Using Label-Free Microscopy
# Author: Md Saimun Alam, Somaiyeh Khoubafarin Doust, Aniruddha Ray
# Lab   : Biophotonics & AI Laboratory, University of Toledo
# =============================================================================
# Interface : Gradio web app (4 tabs: Analysis | Results | Settings | Batch)
# Pipeline  : (1) Automatic Phase / Dark Field modality detection
#             (2) Cellpose cyto3 cell segmentation per frame
#             (3) Hungarian-algorithm cell tracking across frames
#             (4) 4-model voting ensemble classification per track
#             (5) Annotated output image + video + CSV / JSON export
# Input     : .avi microscopy video (Phase Contrast or Dark Field)
# Classes   : Control | Necrosis (H2O2) | Apoptosis (RAP)
# Launch    : python cytomorpheus_gui.py
# =============================================================================
#
# Dependencies:
#   pip install gradio cellpose tensorflow opencv-python scipy numpy
#
# Model paths (edit GUI_MODELS below to match your local layout):
#   GUI_MODELS/
#   ├── Phase_Contrast/   3dcnn.keras  alexnet.keras  mobilenet.keras  effnet.keras
#   ├── Dark_Field/       3dcnn.keras  alexnet.keras  mobilenet.keras  effnet.keras
#   └── Modality_Detector/modality_detector.keras
# =============================================================================


# ── Section 1: Imports & Configuration ───────────────────────────────────────

import os
import tempfile
import json
import csv
from collections import Counter

import numpy as np
import cv2
import tensorflow as tf
import gradio as gr
from cellpose import models as cp_models
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# ── Model root path  (edit this to match your local directory) ────────────────

GUI_MODELS = r"C:\Saimun\cell death GUI_12-22\cell death classification\GUI"

PHASE_MODEL_PATHS = {
    "3dcnn"    : f"{GUI_MODELS}/Phase_Contrast/3dcnn.keras",
    "alexnet"  : f"{GUI_MODELS}/Phase_Contrast/alexnet.keras",
    "mobilenet": f"{GUI_MODELS}/Phase_Contrast/mobilenet.keras",
    "effnet"   : f"{GUI_MODELS}/Phase_Contrast/effnet.keras",
}
DARK_MODEL_PATHS = {
    "3dcnn"    : f"{GUI_MODELS}/Dark_Field/3dcnn.keras",
    "alexnet"  : f"{GUI_MODELS}/Dark_Field/alexnet.keras",
    "mobilenet": f"{GUI_MODELS}/Dark_Field/mobilenet.keras",
    "effnet"   : f"{GUI_MODELS}/Dark_Field/effnet.keras",
}
MODALITY_DETECTOR_PATH = f"{GUI_MODELS}/Modality_Detector/modality_detector.keras"


# ── Cellpose & tracking parameters ───────────────────────────────────────────

DIAMETER              = 50
MIN_CELL_AREA         = 200
MAX_CELL_AREA         = 700
FLOW_THRESHOLD        = 0.4
CELLPROB_THRESHOLD    = 0.0
MAX_DISTANCE          = 50      # max pixel distance for cell linking
TARGET_SEQ_LEN        = 30      # frames fed to classifier per track
RADIUS_MULTIPLIER     = 1.3     # crop radius around cell boundary
MAX_FRAMES_TO_USE     = 60      # default max frames loaded from video
MIN_TRACK_LEN_USE     = 15      # minimum track length to classify
N_FRAMES_FOR_MODALITY = 10      # frames sampled for modality detection

CLASSES      = ["Control", "Necrosis", "Apoptosis"]
CLASS_COLORS = {
    "Control"  : (0, 255, 0),    # green
    "Necrosis" : (255, 0, 0),    # red
    "Apoptosis": (255, 255, 0),  # yellow
}


# ── Lazy-loaded global model holders (avoid reloading on every analysis) ─────

cellpose_model    = None
modality_detector = None
phase_models      = None
dark_models       = None


# ── Startup model check ───────────────────────────────────────────────────────

print("=" * 60)
print("CytoMorpheus Analyzer — Ensemble Configuration")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
print(f"  Keras      : {tf.keras.__version__}")
print(f"  Gradio     : {gr.__version__}")
print()

all_ok = True
for name, path in {
    **PHASE_MODEL_PATHS,
    **DARK_MODEL_PATHS,
    "Modality Detector": MODALITY_DETECTOR_PATH
}.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {name:<20} ({size:.1f} MB)")
    else:
        print(f"  ❌ {name:<20} NOT FOUND — {path}")
        all_ok = False

print()
print("✅ All models found!" if all_ok else "❌ Some models missing — check paths above!")
print("=" * 60)


# ── Section 2: Helper Functions ───────────────────────────────────────────────

def filter_masks_by_size(masks, min_area, max_area):
    """
    Keep only cell masks whose pixel area falls within [min_area, max_area].
    Re-labels surviving masks with consecutive IDs starting from 1.
    """
    filtered = np.zeros_like(masks)
    new_id   = 1
    for cid in np.unique(masks):
        if cid == 0:
            continue
        area = int(np.sum(masks == cid))
        if min_area <= area <= max_area:
            filtered[masks == cid] = new_id
            new_id += 1
    return filtered


def get_cell_centroids(mask):
    """Return {cell_id: (cx, cy)} centroid dictionary for one frame mask."""
    centroids = {}
    for cid in np.unique(mask):
        if cid == 0:
            continue
        ys, xs         = np.where(mask == cid)
        centroids[cid] = (float(xs.mean()), float(ys.mean()))
    return centroids


def track_cells(all_masks, max_distance=50):
    """
    Link cell detections across frames using the Hungarian algorithm.

    Each track is a list of (frame_idx, cell_id, centroid) tuples.
    A new track is started for any detection not matched within max_distance.

    Args:
        all_masks    : list of per-frame segmentation masks
        max_distance : maximum centroid displacement (pixels) to accept a link

    Returns:
        dict  {track_id: [(frame_idx, cell_id, centroid), ...]}
    """
    tracks         = {}
    next_tid       = 1
    centroids_prev = get_cell_centroids(all_masks[0])

    for cid, c in centroids_prev.items():
        tracks[next_tid] = [(0, cid, c)]
        next_tid        += 1

    for f_idx in range(1, len(all_masks)):
        centroids_curr = get_cell_centroids(all_masks[f_idx])
        active         = {tid: pts for tid, pts in tracks.items()
                          if pts[-1][0] == f_idx - 1}

        if not active:
            for cid, c in centroids_curr.items():
                tracks[next_tid] = [(f_idx, cid, c)]
                next_tid        += 1
            continue

        track_ids    = list(active.keys())
        prev_centers = [active[tid][-1][2] for tid in track_ids]
        curr_ids     = list(centroids_curr.keys())
        curr_centers = [centroids_curr[cid] for cid in curr_ids]

        if not curr_ids:
            continue

        cost               = cdist(prev_centers, curr_centers)
        row_ind, col_ind   = linear_sum_assignment(cost)
        matched_curr       = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_distance:
                tid = track_ids[r]
                cid = curr_ids[c]
                tracks[tid].append((f_idx, cid, curr_centers[c]))
                matched_curr.add(cid)

        for cid, c in centroids_curr.items():
            if cid not in matched_curr:
                tracks[next_tid] = [(f_idx, cid, c)]
                next_tid        += 1

    return tracks


def crop_cell_from_frame(frame, mask, cell_id,
                          radius_multiplier=1.3, target_size=(64, 64)):
    """
    Crop a square patch centred on cell_id from frame, scaled by the cell radius.

    Returns:
        Resized (target_size) uint8 crop, or None if the cell is not in mask.
    """
    ys, xs = np.where(mask == cell_id)
    if len(xs) == 0:
        return None

    cx, cy = xs.mean(), ys.mean()
    dists  = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    radius = dists.max()
    crop_r = radius * radius_multiplier

    h, w   = frame.shape[:2]
    x_min  = int(max(0, cx - crop_r))
    x_max  = int(min(w, cx + crop_r))
    y_min  = int(max(0, cy - crop_r))
    y_max  = int(min(h, cy + crop_r))

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    return cv2.resize(crop, target_size)


def build_30_frame_sequence_from_track(track, frames, all_masks,
                                        target_len=30, radius_multiplier=1.3):
    """
    Build a fixed-length sequence of cell crop images from a track.

    If the track is longer than target_len, frames are uniformly sub-sampled.
    Missing crops are filled by repeating the last valid crop.

    Returns:
        List of target_len uint8 images of shape (64, 64, 3).
    """
    track = sorted(track, key=lambda t: t[0])
    L     = len(track)
    idxs  = (np.linspace(0, L - 1, target_len).astype(int)
             if L >= target_len else np.arange(L))

    sequence   = []
    last_valid = None

    for idx in idxs:
        frame_idx, cell_id, _ = track[idx]
        crop = crop_cell_from_frame(
            frames[frame_idx], all_masks[frame_idx],
            cell_id, radius_multiplier=radius_multiplier
        )
        if crop is None:
            if last_valid is not None:
                sequence.append(last_valid)
            continue
        last_valid = crop
        sequence.append(crop)

    # Pad to target_len if needed
    pad = last_valid if last_valid is not None \
          else np.zeros((64, 64, 3), dtype=np.uint8)
    while len(sequence) < target_len:
        sequence.append(pad)

    return sequence[:target_len]


def graph_color_labels(mask):
    """
    Assign distinct RGB colours to cell regions using greedy graph colouring
    (4-connected adjacency).  Returns an RGB image of the same spatial size.
    """
    from collections import defaultdict

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    adjacency = defaultdict(set)
    for label in labels:
        ys, xs = np.where(mask == label)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx    = ys + dy, xs + dx
            valid     = ((ny >= 0) & (ny < mask.shape[0]) &
                         (nx >= 0) & (nx < mask.shape[1]))
            neighbors = mask[ny[valid], nx[valid]]
            adjacency[label].update(
                neighbors[(neighbors != label) & (neighbors != 0)]
            )

    palette = [
        [255, 0, 0],  [0, 255, 0],    [0, 0, 255],    [255, 255, 0],
        [255, 0, 255],[0, 255, 255],   [128, 0, 0],    [0, 128, 0],
        [0, 0, 128],  [128, 128, 0],  [128, 0, 128],  [0, 128, 128],
    ]

    label_colors = {}
    for label in labels:
        used = {label_colors[n] for n in adjacency[label] if n in label_colors}
        for i, _ in enumerate(palette):
            if i not in used:
                label_colors[label] = i
                break

    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color_idx in label_colors.items():
        colored[mask == label] = palette[color_idx]
    return colored


def apply_circle_nms(circles, radius=15, max_overlap=0.55):
    """
    Non-maximum suppression for overlapping cell annotation circles.
    Removes circles whose centres are closer than the overlap threshold.

    Args:
        circles     : list of (cx, cy, label)
        radius      : display circle radius in pixels
        max_overlap : maximum allowed fractional overlap between two circles

    Returns:
        Filtered list of (cx, cy, label).
    """
    min_dist = 2 * radius * (1 - max_overlap)
    accepted = []
    for cx, cy, label in circles:
        too_close = any(
            np.sqrt((cx - ax)**2 + (cy - ay)**2) < min_dist
            for ax, ay, _ in accepted
        )
        if not too_close:
            accepted.append((cx, cy, label))
    return accepted


# ── Section 3: Model Loading & Ensemble Classification ───────────────────────

def load_cellpose():
    """Lazy-load Cellpose cyto3 model (GPU with CPU fallback)."""
    global cellpose_model
    if cellpose_model is not None:
        return cellpose_model
    print("Loading Cellpose model...")
    try:
        cellpose_model = cp_models.CellposeModel(gpu=True, model_type='cyto3')
        print("✓ Cellpose GPU loaded")
    except Exception:
        cellpose_model = cp_models.CellposeModel(gpu=False, model_type='cyto3')
        print("✓ Cellpose CPU loaded")
    return cellpose_model


def load_modality_detector():
    """Lazy-load the binary modality detector (Phase vs Dark)."""
    global modality_detector
    if modality_detector is not None:
        return modality_detector
    print("Loading modality detector...")
    modality_detector = tf.keras.models.load_model(
        MODALITY_DETECTOR_PATH, compile=False
    )
    print("✓ Modality detector loaded")
    return modality_detector


def load_phase_models():
    """Lazy-load all 4 Phase Contrast ensemble models."""
    global phase_models
    if phase_models is not None:
        return phase_models
    print("Loading Phase Contrast ensemble...")
    phase_models = {}
    for name, path in PHASE_MODEL_PATHS.items():
        phase_models[name] = tf.keras.models.load_model(path, compile=False)
        print(f"  ✓ {name}")
    return phase_models


def load_dark_models():
    """Lazy-load all 4 Dark Field ensemble models."""
    global dark_models
    if dark_models is not None:
        return dark_models
    print("Loading Dark Field ensemble...")
    dark_models = {}
    for name, path in DARK_MODEL_PATHS.items():
        dark_models[name] = tf.keras.models.load_model(path, compile=False)
        print(f"  ✓ {name}")
    return dark_models


def classify_sequence(seq_imgs, models_dict):
    """
    Classify a single-cell crop sequence by majority vote across 4 models.

    Args:
        seq_imgs    : list of TARGET_SEQ_LEN (64×64) uint8 images
        models_dict : {name: Keras model} for the relevant modality

    Returns:
        Predicted class string from CLASSES.
    """
    resized = [cv2.resize(img, (224, 224)) for img in seq_imgs]
    x       = np.array(resized, dtype='float32') / 255.0
    x       = np.expand_dims(x, axis=0)   # (1, 30, 224, 224, 3)

    votes = [int(np.argmax(m.predict(x, verbose=0)[0]))
             for m in models_dict.values()]
    return CLASSES[Counter(votes).most_common(1)[0][0]]


def detect_modality_from_frames(frames):
    """
    Automatic Phase vs Dark Field detection using N_FRAMES_FOR_MODALITY frames.
    The modality detector outputs a sigmoid score: >0.5 → dark, ≤0.5 → phase.

    Returns:
        (modality_string, confidence_float)
    """
    md   = load_modality_detector()
    n    = min(N_FRAMES_FOR_MODALITY, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)

    imgs      = [cv2.resize(frames[i], (224, 224)) for i in idxs]
    x         = np.array(imgs, dtype='float32') / 255.0
    mean_pred = float(np.mean(md.predict(x, verbose=0)))

    if mean_pred < 0.5:
        return "phase", 1.0 - mean_pred
    else:
        return "dark", mean_pred


# ── Section 4: Video Analysis Pipeline ───────────────────────────────────────

# Global store — populated by analyze_video for CSV/JSON export
last_results = {}


def generate_annotated_video(frames, per_cell_results, output_path, radius=15):
    """
    Write an annotated .mp4 video with NMS-filtered classification circles
    drawn on each frame.

    Args:
        frames           : list of RGB uint8 frames
        per_cell_results : list of {label, track_id, track, sequence} dicts
        output_path      : filesystem path for the output .mp4
        radius           : display circle radius in pixels
    """
    h, w    = frames[0].shape[:2]
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(output_path, fourcc, 10, (w, h))

    # Pre-build frame → circles map
    frame_circles = {}
    for result in per_cell_results:
        label = result['label']
        color = CLASS_COLORS[label]
        for frame_idx, cell_id, centroid in result['track']:
            cx, cy = int(centroid[0]), int(centroid[1])
            frame_circles.setdefault(frame_idx, []).append(
                (cx, cy, color, label)
            )

    for i, frame in enumerate(frames):
        bgr        = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raw        = frame_circles.get(i, [])
        nms_in     = [(cx, cy, lbl) for cx, cy, _, lbl in raw]
        nms_out    = apply_circle_nms(nms_in, radius=radius, max_overlap=0.55)
        nms_lookup = {(cx, cy): lbl for cx, cy, lbl in nms_out}

        for cx, cy, color, label in raw:
            if (cx, cy) in nms_lookup:
                cv2.circle(bgr, (cx, cy), radius,
                           (color[2], color[1], color[0]), 2)   # BGR

        writer.write(bgr)

    writer.release()
    return output_path


def analyze_video(video_path, max_frames=MAX_FRAMES_TO_USE,
                  progress=gr.Progress()):
    """
    Full single-video analysis pipeline.

    Steps:
        1. Load frames from video
        2. Detect modality (phase / dark)
        3. Cellpose segmentation on every frame
        4. Hungarian algorithm cell tracking
        5. Build 30-frame crop sequence per track
        6. Ensemble voting classification per track
        7. NMS-filtered annotation on frame 0
        8. Generate annotated output video
        9. Build cell gallery (up to 5 per class)
       10. Populate last_results for CSV/JSON export

    Returns:
        Tuple consumed by Gradio outputs:
        (detection_overlay, annotated_image, summary_text,
         total_cells, control_count, necrosis_count, apoptosis_count,
         gallery_images, annotated_video_path)
    """
    global last_results

    progress(0,    desc="Initializing...")
    cp = load_cellpose()

    # ── 1. Load frames ────────────────────────────────────────────────────────
    progress(0.05, desc="Loading video frames...")
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError("Could not read any frames from the video.")

    num_frames = len(frames)
    progress(0.10, desc=f"Loaded {num_frames} frames")

    # ── 2. Detect modality ────────────────────────────────────────────────────
    progress(0.15, desc="Detecting modality...")
    modality_choice, modality_conf = detect_modality_from_frames(frames)

    if modality_choice == "phase":
        models_dict  = load_phase_models()
        modality_str = f"Phase Contrast ({modality_conf * 100:.1f}%)"
    else:
        models_dict  = load_dark_models()
        modality_str = f"Dark Field ({modality_conf * 100:.1f}%)"

    # ── 3. Cellpose segmentation ──────────────────────────────────────────────
    progress(0.25, desc="Detecting cells...")
    all_masks = []
    mask0     = None

    for i, fr in enumerate(frames):
        masks, _, _ = cp.eval(
            fr,
            diameter=DIAMETER,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD
        )
        filt = filter_masks_by_size(masks, MIN_CELL_AREA, MAX_CELL_AREA)
        all_masks.append(filt)
        if i == 0:
            mask0 = filt.copy()

        if i % 10 == 0 or i == num_frames - 1:
            n_cells = len(np.unique(filt)) - 1
            progress(0.25 + 0.15 * (i / max(1, num_frames - 1)),
                     desc=f"Detecting cells... {i+1}/{num_frames} ({n_cells} cells)")

    # ── 4. Detection overlay ──────────────────────────────────────────────────
    frame0  = frames[0]
    colored = graph_color_labels(mask0)
    overlay = (frame0 * 0.65 + colored * 0.35).astype(np.uint8)

    # ── 5. Track cells ────────────────────────────────────────────────────────
    progress(0.45, desc="Tracking cells...")
    tracks       = track_cells(all_masks, max_distance=MAX_DISTANCE)
    total_tracks = len(tracks)

    if total_tracks == 0:
        raise RuntimeError("No cell tracks found — try adjusting detection parameters.")

    # ── 6. Classify each track ────────────────────────────────────────────────
    progress(0.50, desc="Classifying cells...")
    per_cell_results = []

    for i, (tid, track) in enumerate(tracks.items(), 1):
        if len(track) < MIN_TRACK_LEN_USE:
            continue

        seq_imgs = build_30_frame_sequence_from_track(
            track, frames, all_masks,
            target_len=TARGET_SEQ_LEN,
            radius_multiplier=RADIUS_MULTIPLIER
        )
        if not seq_imgs:
            continue

        final_label = classify_sequence(seq_imgs, models_dict)
        per_cell_results.append({
            'label'   : final_label,
            'track_id': tid,
            'track'   : track,
            'sequence': seq_imgs,
        })

        if i % 50 == 0 or i == total_tracks:
            partial = Counter([r['label'] for r in per_cell_results])
            progress(0.50 + 0.35 * (i / total_tracks),
                     desc=f"Classifying {i}/{total_tracks} — {dict(partial)}")

    if not per_cell_results:
        raise RuntimeError("No cells classified — try increasing Max Frames "
                           "or lowering Min Track Length.")

    # ── 7. Count results ──────────────────────────────────────────────────────
    vote_counts     = Counter([r['label'] for r in per_cell_results])
    total_cells     = sum(vote_counts.values())
    control_count   = vote_counts.get("Control",   0)
    necrosis_count  = vote_counts.get("Necrosis",  0)
    apoptosis_count = vote_counts.get("Apoptosis", 0)

    # ── 8. Annotated static image (NMS circles) ───────────────────────────────
    progress(0.88, desc="Creating annotated image...")
    all_circles = []
    for result in per_cell_results:
        frame_idx, cell_id, centroid = result['track'][0]
        cx, cy = int(centroid[0]), int(centroid[1])
        all_circles.append((cx, cy, result['label']))

    accepted_circles = apply_circle_nms(all_circles, radius=15, max_overlap=0.55)
    annotated        = frame0.copy()
    for cx, cy, label in accepted_circles:
        cv2.circle(annotated, (cx, cy), 15, CLASS_COLORS[label], 2)

    # ── 9. Annotated output video ─────────────────────────────────────────────
    progress(0.92, desc="Generating annotated video...")
    video_out = os.path.join(tempfile.gettempdir(), "cytomorpheus_annotated.mp4")
    generate_annotated_video(frames, per_cell_results, video_out)

    # ── 10. Cell gallery (up to 5 per class) ──────────────────────────────────
    gallery_images = []
    class_counts   = {cls: 0 for cls in CLASSES}
    for result in per_cell_results:
        label = result['label']
        if class_counts[label] < 5:
            gallery_images.append((result['sequence'][0], label))
            class_counts[label] += 1
        if all(v >= 5 for v in class_counts.values()):
            break

    # ── 11. Summary text ──────────────────────────────────────────────────────
    def pct(n):
        return (n / total_cells * 100.0) if total_cells > 0 else 0.0

    summary = (
        f"Video    : {os.path.basename(video_path)}\n"
        f"Modality : {modality_str}\n"
        f"Frames   : {num_frames}\n"
        f"Cells classified : {total_cells}\n\n"
        f"🟢 Control   : {control_count:>4}  ({pct(control_count):.1f}%)\n"
        f"🔴 Necrosis  : {necrosis_count:>4}  ({pct(necrosis_count):.1f}%)\n"
        f"🟡 Apoptosis : {apoptosis_count:>4}  ({pct(apoptosis_count):.1f}%)\n"
    )

    # ── 12. Store for export ──────────────────────────────────────────────────
    last_results = {
        'video'    : os.path.basename(video_path),
        'modality' : modality_str,
        'frames'   : num_frames,
        'total'    : total_cells,
        'control'  : control_count,
        'necrosis' : necrosis_count,
        'apoptosis': apoptosis_count,
        'per_cell' : [{'track_id': r['track_id'], 'label': r['label']}
                      for r in per_cell_results],
    }

    progress(1.0, desc="Complete!")

    return (overlay, annotated, summary,
            int(total_cells), int(control_count),
            int(necrosis_count), int(apoptosis_count),
            gallery_images, video_out)


# ── Section 5: Gradio GUI ─────────────────────────────────────────────────────

def create_gui():
    """
    Build and return the CytoMorpheus Analyzer Gradio interface.

    Tabs:
      Analysis         — upload video, run pipeline, view results live
      Results          — summary text, export CSV / JSON, annotated video
      Settings         — detection & tracking parameter sliders (UI only)
      Batch Processing — placeholder for future multi-video support
    """

    # ── CSS: forced dark theme ────────────────────────────────────────────────
    css = """
    :root {
        --body-background-fill:            #1a3a5c !important;
        --body-background-fill-secondary:  #1e4570 !important;
        --background-fill-primary:         #1e3f6a !important;
        --background-fill-secondary:       #163354 !important;
        --border-color-primary:            rgba(255,255,255,0.25) !important;
        --border-color-accent:             rgba(255,255,255,0.40) !important;
        --block-background-fill:           rgba(255,255,255,0.07) !important;
        --block-border-color:              rgba(255,255,255,0.25) !important;
        --block-label-background-fill:     rgba(30,60,100,0.90)  !important;
        --block-label-text-color:          white !important;
        --block-title-text-color:          white !important;
        --body-text-color:                 white !important;
        --body-text-color-subdued:         rgba(255,255,255,0.70) !important;
        --input-background-fill:           rgba(255,255,255,0.10) !important;
        --input-background-fill-focus:     rgba(255,255,255,0.15) !important;
        --input-border-color:              rgba(255,255,255,0.30) !important;
        --input-border-color-focus:        rgba(255,255,255,0.60) !important;
        --input-placeholder-color:         rgba(255,255,255,0.50) !important;
        --loader-color:                    #f97316 !important;
        --shadow-drop:                     none !important;
        --shadow-drop-lg:                  none !important;
        --slider-color:                    #f97316 !important;
        --stat-background-fill:            rgba(255,255,255,0.10) !important;
        --color-accent:                    #f97316 !important;
        --panel-background-fill:           rgba(20,50,90,0.90)   !important;
        --panel-border-color:              rgba(255,255,255,0.20) !important;
    }

    .gradio-container, .gradio-container * { color: white !important; }
    body, .gradio-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8d 100%) !important;
    }

    /* ── Tabs ── */
    .tab-nav {
        background: rgba(255,255,255,0.05) !important;
        border-bottom: 1px solid rgba(255,255,255,0.20) !important;
    }
    .tab-nav button {
        color: rgba(255,255,255,0.75) !important;
        background: transparent !important;
        border: none !important;
        padding: 10px 20px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    .tab-nav button.selected {
        color: white !important;
        border-bottom: 2px solid #f97316 !important;
        font-weight: 700 !important;
    }
    .tab-nav button:hover {
        color: white !important;
        background: rgba(255,255,255,0.08) !important;
    }

    /* ── Section headers ── */
    .section-header {
        background: rgba(15,50,100,0.90) !important;
        color: white !important;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
        margin-top: 4px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.4px !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
    }

    /* ── Blocks ── */
    .block, .gr-block, .gr-box, .gr-form {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: 10px !important;
    }

    /* ── All labels ── */
    label, label span, label p, .block > label, .block > label > span,
    span[data-testid], .label-wrap span, [class*="label"] {
        color: white !important;
        font-weight: 500 !important;
    }

    /* ── Number inputs (metric boxes) ── */
    input[type="number"] {
        background: rgba(255,255,255,0.12) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.30) !important;
        border-radius: 8px !important;
        text-align: center !important;
        font-size: 22px !important;
        font-weight: bold !important;
        padding: 12px !important;
        min-height: 58px !important;
    }

    /* ── Slider containers ── */
    .gr-block .wrap, .gradio-slider {
        padding: 8px 4px 16px 4px !important;
        min-height: 85px !important;
    }
    input[type="range"] {
        width: 100% !important;
        accent-color: #f97316 !important;
        height: 6px !important;
        cursor: pointer !important;
    }

    /* ── Slider value box ── */
    .gradio-slider input[type="number"],
    .wrap input[type="number"],
    .wrap > div > input[type="number"] {
        background: #ffffff !important;
        color: #111111 !important;
        border: 2px solid #f97316 !important;
        border-radius: 6px !important;
        font-size: 15px !important;
        font-weight: bold !important;
        padding: 5px 8px !important;
        min-height: 34px !important;
        min-width: 60px !important;
        text-align: center !important;
    }

    /* ── Slider min/max text ── */
    .range-slider span, .wrap span, .wrap > span {
        color: rgba(255,255,255,0.80) !important;
        font-size: 12px !important;
    }

    /* ── Text inputs & textareas ── */
    input[type="text"], textarea {
        background: rgba(255,255,255,0.90) !important;
        color: #111 !important;
        border-radius: 6px !important;
    }

    /* ── File upload ── */
    .upload-container, .file-preview, [data-testid="file"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px dashed rgba(255,255,255,0.35) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    .upload-container span, .upload-container p, .file-preview span {
        color: white !important;
    }

    /* ── Gallery ── */
    .gallery-container, [data-testid="gallery"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }
    .gallery-item figcaption, .thumbnail-item .caption {
        color: white !important;
        background: rgba(0,0,0,0.70) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    /* ── Buttons ── */
    .primary-btn, button.primary {
        background: #f97316 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button.secondary, .secondary-btn {
        background: rgba(255,255,255,0.12) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.30) !important;
        border-radius: 8px !important;
    }

    h1, h2, h3, h4, h5, h6, p, span { color: white !important; }

    /* ── Hide "built with Gradio" text ── */
    .built-with span { display: none !important; }
    """

    with gr.Blocks(title="CytoMorpheus Analyzer", css=css) as demo:

        gr.Markdown("# 🧬 CytoMorpheus Analyzer")

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — ANALYSIS
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Analysis"):
                with gr.Row():

                    # ── Left panel ────────────────────────────────────────────
                    with gr.Column(scale=1):

                        gr.HTML("<div class='section-header'>Raw Microscopy</div>")
                        video_input = gr.File(
                            label="Upload Video",
                            file_types=[".avi"],
                            file_count="single"
                        )

                        gr.HTML("<div class='section-header'>Max Frames</div>")
                        max_frames = gr.Slider(
                            minimum=30, maximum=150,
                            value=60, step=10,
                            label="Number of frames to process",
                            info="Drag slider or type value (30–150)"
                        )

                        analyze_btn = gr.Button(
                            "🔬 Run Analysis",
                            variant="primary", size="lg"
                        )

                        gr.HTML("<div class='section-header'>Classification Summary</div>")
                        total_cells = gr.Number(
                            label="Total Cells", value=0,
                            interactive=False, precision=0
                        )
                        with gr.Row(equal_height=True):
                            control_count = gr.Number(
                                label="🟢 Control", value=0,
                                interactive=False, precision=0, scale=1
                            )
                            necrosis_count = gr.Number(
                                label="🔴 Necrosis", value=0,
                                interactive=False, precision=0, scale=1
                            )
                            apoptosis_count = gr.Number(
                                label="🟡 Apoptosis", value=0,
                                interactive=False, precision=0, scale=1
                            )

                    # ── Right panel ───────────────────────────────────────────
                    with gr.Column(scale=2):

                        gr.HTML("<div class='section-header'>Detection & Classification</div>")
                        with gr.Row():
                            original_overlay = gr.Image(
                                label="Detection Overlay",
                                interactive=False, height=380
                            )
                            annotated_result = gr.Image(
                                label="Classified Cells",
                                interactive=False, height=380
                            )

                        progress_text = gr.Textbox(
                            label="Progress",
                            value="Ready to analyze...",
                            interactive=False, lines=2,
                            elem_classes=["progress-text"]
                        )

                        gr.HTML("<div class='section-header'>Individual Cell Preview</div>")
                        cell_gallery = gr.Gallery(
                            label="Scroll to browse cells — labeled by class",
                            show_label=True,
                            columns=5, rows=2,
                            height=240,
                            object_fit="contain",
                            allow_preview=True
                        )

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — RESULTS
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Results"):
                gr.HTML("<div class='section-header'>Analysis Results</div>")
                results_summary = gr.Textbox(
                    label="Summary Report",
                    lines=12, interactive=False
                )
                with gr.Row():
                    export_csv_btn  = gr.Button("💾 Export CSV",  size="lg", scale=1)
                    export_json_btn = gr.Button("💾 Export JSON", size="lg", scale=1)

                download_file = gr.File(label="Download File")

                gr.HTML("<div class='section-header'>Annotated Video</div>")
                result_video = gr.Video(
                    label="Annotated Classification Video",
                    interactive=False, height=450
                )

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 — SETTINGS
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Settings"):
                gr.HTML("<div class='section-header'>Detection Parameters</div>")
                with gr.Row():
                    with gr.Column():
                        gr.Slider(30,  100,  value=50,  label="Cell Diameter")
                        gr.Slider(100, 500,  value=200, label="Min Cell Area")
                        gr.Slider(500, 1000, value=700, label="Max Cell Area")
                    with gr.Column():
                        gr.Slider(0.1, 1.0, value=0.4, step=0.1,
                                  label="Flow Threshold")
                        gr.Slider(-6,  6,   value=0.0, step=0.5,
                                  label="Cell Prob Threshold")
                        gr.Slider(20,  100, value=50,
                                  label="Tracking Distance")
                gr.Button("💾 Save Settings",    variant="primary")
                gr.Button("🔄 Reset to Default")

            # ══════════════════════════════════════════════════════════════════
            # TAB 4 — BATCH PROCESSING
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Batch Processing"):
                gr.HTML("<div class='section-header'>Batch Analysis</div>")
                gr.Markdown("*Coming soon — analyze multiple videos at once*")
                gr.Textbox(
                    label="Video Folder Path",
                    placeholder="Select folder...",
                    interactive=False
                )
                gr.Button("🚀 Process Batch", interactive=False)

        # ── Export callbacks ──────────────────────────────────────────────────
        def export_csv():
            if not last_results:
                raise gr.Error("Run analysis first.")
            path = os.path.join(tempfile.gettempdir(), "cytomorpheus_results.csv")
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Video", "Modality", "Frames",
                                  "Total", "Control", "Necrosis", "Apoptosis"])
                writer.writerow([
                    last_results['video'],  last_results['modality'],
                    last_results['frames'], last_results['total'],
                    last_results['control'], last_results['necrosis'],
                    last_results['apoptosis']
                ])
                writer.writerow([])
                writer.writerow(["Track ID", "Label"])
                for cell in last_results['per_cell']:
                    writer.writerow([cell['track_id'], cell['label']])
            return path

        def export_json():
            if not last_results:
                raise gr.Error("Run analysis first.")
            path = os.path.join(tempfile.gettempdir(), "cytomorpheus_results.json")
            with open(path, 'w') as f:
                json.dump(last_results, f, indent=2)
            return path

        # ── Main analysis wiring ──────────────────────────────────────────────
        def gradio_analyze(video_file, max_frames_value,
                           progress=gr.Progress()):
            if video_file is None:
                raise gr.Error("Please upload a video first.")
            return analyze_video(
                video_file.name, int(max_frames_value), progress
            )

        analyze_btn.click(
            gradio_analyze,
            inputs=[video_input, max_frames],
            outputs=[original_overlay, annotated_result, results_summary,
                     total_cells, control_count, necrosis_count,
                     apoptosis_count, cell_gallery, result_video]
        )

        export_csv_btn.click( export_csv,  inputs=[], outputs=[download_file])
        export_json_btn.click(export_json, inputs=[], outputs=[download_file])

    return demo


# ── Section 6: Launch ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_gui()
    demo.launch(share=False, inbrowser=True)
