"""Per-cell ground truth from the propidium iodide channel.

A recording carries the treatment of its dish, but the models are trained on
single cells, and within one recording some treated cells die early, others
late, and some do not die at all before the recording ends.  A treatment label
therefore carries an unknown error at the single-cell level.

Here every cell is labelled by its own PI signal instead.  Cellpose runs on the
fluorescence channel so that stained nuclei are detected as objects rather than
scored by intensity, which avoids the error of a bright neighbouring nucleus
raising the signal inside a mask.  A cell is called dead at the first frame
where a detected nucleus covers more than half of its mask for two consecutive
frames.  Cells that are already PI-positive in their first three frames were
damaged before the treatment and are dropped.

The label then follows the cell.  A Raptinal cell that becomes PI-positive is
apoptosis, an H2O2 cell that becomes PI-positive is necrosis, and an untreated
cell that stays PI-negative is control.  A treated cell that never becomes
PI-positive gets no label and is held aside.

    python -m preprocessing.pi_ground_truth --fluor rec__fluor.avi \
           --labels rec_masks.npz --treatment RAPTINAL --out rec_truth.csv
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

from .config import (PI_CELLPROB, PI_COVERAGE, PI_DEBOUNCE, PI_FLOW,
                     PI_PRE_CLEAN, read_video)

TREATMENT_TO_CLASS = {"RAPTINAL": "APOPTOSIS", "H2O2": "NECROSIS", "NONE": "CONTROL"}


def segment_pi(stack, log=print):
    """Cellpose on the fluorescence channel -> (T, H, W) int32 nucleus labels."""
    from cellpose import models
    import torch
    model = models.CellposeModel(gpu=torch.cuda.is_available())
    out = np.zeros(stack.shape, np.int32)
    for t in range(len(stack)):
        out[t] = model.eval(stack[t], flow_threshold=PI_FLOW,
                            cellprob_threshold=PI_CELLPROB)[0]
        if t % 10 == 0:
            log(f"  PI frame {t + 1}/{len(stack)}: {int(out[t].max())} nuclei")
    return out


def coverage(labels, nuclei):
    """Per frame and per track, the largest fraction of the mask covered by one nucleus."""
    T = len(labels)
    n_tracks = int(labels.max())
    cov = np.zeros((T, n_tracks + 1), np.float32)
    for t in range(T):
        lab, nuc = labels[t], nuclei[t]
        if lab.max() == 0 or nuc.max() == 0:
            continue
        area = np.bincount(lab.ravel(), minlength=n_tracks + 1).astype(float)
        nb = int(nuc.max()) + 1
        pair = np.bincount((lab.astype(np.int64) * nb + nuc).ravel(),
                           minlength=(n_tracks + 1) * nb).reshape(n_tracks + 1, nb)
        pair[:, 0] = 0                      # background of the PI channel
        cov[t] = pair.max(1) / np.maximum(area, 1)
    return cov


def death_frames(cov):
    """First frame of a sustained PI-positive run, per track.

    Returns a dict track -> death frame, and the set of tracks that were already
    positive in their first PI_PRE_CLEAN frames and must be discarded.
    """
    T, n = cov.shape
    deaths, dirty = {}, set()
    positive = cov > PI_COVERAGE
    for tid in range(1, n):
        col = positive[:, tid]
        if col[:PI_PRE_CLEAN].any():
            dirty.add(tid)
            continue
        run = 0
        for t in range(T):
            run = run + 1 if col[t] else 0
            if run >= PI_DEBOUNCE:
                deaths[tid] = t - PI_DEBOUNCE + 1
                break
    return deaths, dirty


def label_cells(labels, nuclei, treatment):
    """One row per track: class, death frame, and why a track was excluded."""
    cov = coverage(labels, nuclei)
    deaths, dirty = death_frames(cov)
    rows = []
    for tid in range(1, int(labels.max()) + 1):
        present = np.nonzero((labels == tid).any(axis=(1, 2)))[0]
        if len(present) == 0:
            continue
        if tid in dirty:
            rows.append(dict(track=tid, cls=None, death=None,
                             note="PI-positive in the first frames"))
            continue
        if tid in deaths:
            if treatment == "NONE":
                rows.append(dict(track=tid, cls=None, death=deaths[tid],
                                 note="untreated cell that became PI-positive"))
            else:
                rows.append(dict(track=tid, cls=TREATMENT_TO_CLASS[treatment],
                                 death=deaths[tid], note=""))
        else:
            if treatment == "NONE":
                rows.append(dict(track=tid, cls="CONTROL", death=None, note=""))
            else:
                rows.append(dict(track=tid, cls=None, death=None,
                                 note="treated, never PI-positive, held aside"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Per-cell PI ground truth for one recording.")
    ap.add_argument("--fluor", required=True, help="PI fluorescence video")
    ap.add_argument("--labels", required=True, help="tracked label stack from segment_track")
    ap.add_argument("--treatment", required=True, choices=list(TREATMENT_TO_CLASS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    labels = np.load(a.labels)["labels"]
    nuclei = segment_pi(read_video(a.fluor))
    df = label_cells(labels, nuclei, a.treatment)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    n_lab = df.cls.notna().sum()
    print(f"{len(df)} tracks, {n_lab} labelled, {len(df) - n_lab} held aside or dropped")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
