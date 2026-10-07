"""Segmentation and tracking.

Cellpose-SAM runs on every phase-contrast frame at full resolution.  The masks
are then linked from frame to frame by mutual-best overlap, with gap closing of
up to two frames and a displacement cap, so that each cell receives one
continuous track.  The same masks are applied to the dark-field frames, which
share the same coordinates, so a cell's two sequences describe the same object.

    python -m preprocessing.segment_track --phase rec__phase.avi --out rec_masks.npz
"""
import argparse
import math
import pathlib

import numpy as np

from .config import CELLPROB, FLOW, IOU_MIN, MAX_GAP, MAX_STEP_PX, read_video


def segment(stack, log=print):
    """Cellpose-SAM on every frame -> (T, H, W) int32 label image."""
    from cellpose import models
    import torch
    model = models.CellposeModel(gpu=torch.cuda.is_available())
    masks = np.zeros(stack.shape, np.int32)
    for t in range(len(stack)):
        masks[t] = model.eval(stack[t], flow_threshold=FLOW,
                              cellprob_threshold=CELLPROB)[0]
        if t % 10 == 0:
            log(f"  frame {t + 1}/{len(stack)}: {int(masks[t].max())} cells")
    return masks


def _props(lab):
    """Area and centroid of every label in one frame."""
    n = int(lab.max())
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    ys, xs = np.nonzero(lab)
    v = lab[ys, xs]
    cy = np.bincount(v, weights=ys, minlength=n + 1) / np.maximum(area, 1)
    cx = np.bincount(v, weights=xs, minlength=n + 1) / np.maximum(area, 1)
    return area, cy, cx


def _iou(a, b):
    """Intersection over union between every label of two frames."""
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    inter = np.bincount((a.astype(np.int64) * nb + b).ravel(),
                        minlength=na * nb).reshape(na, nb).astype(float)
    aa, bb = inter.sum(1, keepdims=True), inter.sum(0, keepdims=True)
    return inter / np.maximum(aa + bb - inter, 1)


def link_tracks(masks, log=print):
    """Per-frame masks -> (T, H, W) int32 where a label is one cell over time.

    A link is made only when the overlap is mutually best, is at least IOU_MIN,
    and the centroid has moved less than the displacement cap.  A track may skip
    up to MAX_GAP frames, which covers a cell that Cellpose misses briefly.
    """
    T = len(masks)
    labels = np.zeros(masks.shape, np.int32)
    next_id, last = 1, {}
    for t in range(T):
        lab = masks[t]
        area, cy, cx = _props(lab)
        n = int(lab.max())
        assigned = np.zeros(n + 1, bool)
        for gap in range(1, MAX_GAP + 2):
            if t - gap < 0:
                break
            cand = [tid for tid, (f, l, y, x) in last.items() if f == t - gap]
            if not cand:
                continue
            iou = _iou(masks[t - gap], lab)
            cand_lab = np.array([last[tid][1] for tid in cand])
            sub = iou[cand_lab]
            sub[:, 0] = 0
            sub[:, assigned] = 0
            best_b, best_a = sub.argmax(1), sub.argmax(0)
            for i, tid in enumerate(cand):
                j = best_b[i]
                if j == 0 or sub[i, j] < IOU_MIN or best_a[j] != i:
                    continue
                if math.hypot(cy[j] - last[tid][2], cx[j] - last[tid][3]) > MAX_STEP_PX * gap:
                    continue
                labels[t][lab == j] = tid
                assigned[j] = True
                last[tid] = (t, j, cy[j], cx[j])
        for j in range(1, n + 1):
            if not assigned[j]:
                labels[t][lab == j] = next_id
                last[next_id] = (t, j, cy[j], cx[j])
                next_id += 1
    log(f"  {next_id - 1} cell tracks")
    return labels


def main():
    ap = argparse.ArgumentParser(description="Segment and track one recording.")
    ap.add_argument("--phase", required=True, help="phase-contrast video")
    ap.add_argument("--out", required=True, help="output .npz with the label stack")
    a = ap.parse_args()

    stack = read_video(a.phase)
    print(f"{a.phase}: {stack.shape[0]} frames")
    labels = link_tracks(segment(stack))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, labels=labels)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
