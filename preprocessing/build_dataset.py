"""Anchored windows, the array the models are trained on.

Recordings differ in length between conditions, because H2O2-treated cells die
much faster than Raptinal-treated cells, so a sequence covering a whole track
would reveal the treatment by its duration alone.  Each cell is therefore cut
to the same fixed window, 15 frames from 10 before its death frame to 4 after
it, 45 minutes in total, and the rupture always sits at the same position.

Control cells have no death frame, so their window is placed at a comparable
position within the recording.  Cells that die too close to the start or the end
of a recording cannot be given a full window and are excluded.

Each frame is cropped to 48 x 48 pixels, 45 um, around the tracked cell.  Three
planes are stored, phase contrast, dark-field and the mask of the tracked cell,
so that both modalities come from the same frames and the same masks.  The
network channels are derived from these at training time, see cytomorpheus.data.

    python -m preprocessing.build_dataset --manifest recordings.csv --out 00_data
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

from .config import BOX, CLASSES, POST, PRE, WINDOW, read_video


def crop_window(phase, dark, labels, track, anchor):
    """(15, 3, 48, 48) uint8 for one cell, or None if the window does not fit."""
    lo, hi = anchor - PRE, anchor + POST + 1
    if lo < 0 or hi > len(labels):
        return None
    h = BOX // 2
    out = np.zeros((WINDOW, 3, BOX, BOX), np.uint8)
    for k, t in enumerate(range(lo, hi)):
        m = labels[t] == track
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        y, x = int(round(ys.mean())), int(round(xs.mean()))
        P = np.pad(phase[t], h)
        D = np.pad(dark[t], h)
        L = np.pad(m.astype(np.uint8), h)
        sl = (slice(y, y + BOX), slice(x, x + BOX))
        out[k] = np.stack([P[sl], D[sl], L[sl] * 255])
    return out


def control_anchor(n_frames, rng):
    """A window position for a control cell, matched to the treated cells."""
    lo, hi = PRE, n_frames - POST - 1
    if hi < lo:
        return None
    return int(rng.integers(lo, hi + 1))


def build(manifest, out_dir, seed=0):
    """Build cache.npy, meta.npz and index.csv from a manifest of recordings.

    The manifest is a CSV with one row per recording and the columns
    recording, phase, dark, labels, truth, fold.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    man = pd.read_csv(manifest)

    windows, rows = [], []
    for _, r in man.iterrows():
        phase = read_video(r.phase)
        dark = read_video(r.dark)
        labels = np.load(r.labels)["labels"]
        truth = pd.read_csv(r.truth)
        kept = dropped = 0
        for _, c in truth.iterrows():
            if not isinstance(c.cls, str):
                continue
            anchor = (int(c.death) if not pd.isna(c.death)
                      else control_anchor(len(labels), rng))
            if anchor is None:
                dropped += 1
                continue
            w = crop_window(phase, dark, labels, int(c.track), anchor)
            if w is None:
                dropped += 1
                continue
            windows.append(w)
            rows.append(dict(uid=f"{r.recording}__c{int(c.track):05d}",
                             cls=c.cls, rec=r.recording, fold=r.fold,
                             anchor=anchor, n=len(labels)))
            kept += 1
        print(f"{r.recording}: {kept} cells kept, {dropped} without a full window")

    X = np.stack(windows)
    idx = pd.DataFrame(rows)
    y = np.array([CLASSES.index(c) for c in idx.cls], np.int64)

    np.save(out_dir / "cache.npy", X)
    np.savez(out_dir / "meta.npz", y=y, rec=idx.rec.values.astype(str),
             fold=idx.fold.values.astype(str), uid=idx.uid.values.astype(str),
             PRE=PRE, POST=POST, L=WINDOW)
    idx.to_csv(out_dir / "index.csv", index=False)
    print(f"\n{len(X)} cells  " +
          "  ".join(f"{c} {int((y == i).sum())}" for i, c in enumerate(CLASSES)))
    print("wrote", out_dir / "cache.npy")


def main():
    ap = argparse.ArgumentParser(description="Cut the anchored training windows.")
    ap.add_argument("--manifest", required=True,
                    help="CSV: recording, phase, dark, labels, truth, fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the control window positions")
    a = ap.parse_args()
    build(a.manifest, a.out, a.seed)


if __name__ == "__main__":
    main()
