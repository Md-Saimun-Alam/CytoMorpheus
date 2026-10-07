"""Voting, cross-modality fusion and the result tables.

Two levels of combination, both of them averages of class probabilities and
neither of them trained.

  voting            within one modality, average the probabilities of the four
                    architectures and take the highest mean
  cross-modality    for one cell, average the probabilities of the four
  fusion            phase-contrast models and the four dark-field models, so the
                    final call rests on eight models and two modalities

The eight networks are separate.  A phase model never sees a dark-field image,
nothing is trained jointly, and no weights are shared.  Fusion happens only at
prediction time, which is why the same models still work when only one modality
has been recorded.

    python -m cytomorpheus.evaluate --models 01_models --out 02_results \
           --protocol LORO --fold F1
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .models import NAMES

MODALITIES = ["phase", "dark"]
CLS = ["Apoptosis", "Control", "Necrosis"]


def softmax(logit):
    e = np.exp(logit - logit.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def metrics(y, P):
    """Balanced accuracy, accuracy, macro F1 and AUC, plus per-class values."""
    yp = P.argmax(1)
    Y = np.eye(3)[y]
    d = dict(balanced_acc=balanced_accuracy_score(y, yp),
             accuracy=accuracy_score(y, yp),
             macro_F1=f1_score(y, yp, average="macro"),
             macro_AUC=roc_auc_score(Y, P, average="macro"))
    for k, c in enumerate(CLS):
        d[f"{c}_recall"] = recall_score(y, yp, labels=[k], average=None, zero_division=0)[0]
        d[f"{c}_precision"] = precision_score(y, yp, labels=[k], average=None, zero_division=0)[0]
        d[f"{c}_F1"] = f1_score(y, yp, labels=[k], average=None, zero_division=0)[0]
        d[f"{c}_AUC"] = roc_auc_score(Y[:, k], P[:, k])
        d[f"{c}_AP"] = average_precision_score(Y[:, k], P[:, k])
    return d


def load_predictions(models_dir, protocol, fold, seeds=(0, 1, 2)):
    """{(model, modality): {y, rec, uid, Ps}} with one probability array per seed."""
    models_dir = pathlib.Path(models_dir)
    out = {}
    for name in NAMES:
        for mod in MODALITIES:
            zs = [np.load(models_dir / name / f"pred_{mod}_{protocol}_{fold}_s{s}.npz",
                          allow_pickle=True) for s in seeds]
            for z in zs[1:]:
                assert (z["uid"] == zs[0]["uid"]).all(), "seeds disagree on the test cells"
            out[(name, mod)] = dict(y=zs[0]["y"].astype(int), rec=zs[0]["rec"].astype(str),
                                    uid=zs[0]["uid"], Ps=[softmax(z["logit"]) for z in zs])
    return out


def add_voting(pred, seeds=(0, 1, 2)):
    """Average the four architectures within each modality."""
    for mod in MODALITIES:
        ref = pred[(NAMES[0], mod)]
        for name in NAMES[1:]:
            assert (pred[(name, mod)]["uid"] == ref["uid"]).all()
        pred[("Voting", mod)] = dict(
            y=ref["y"], rec=ref["rec"], uid=ref["uid"],
            Ps=[np.mean([pred[(n, mod)]["Ps"][s] for n in NAMES], 0) for s in range(len(seeds))])
    return pred


def cross_modality_fusion(pred, seeds=(0, 1, 2)):
    """Average all eight probability vectors of the same cell.

    Valid because both modalities were recorded from the same field without
    moving the stage, so a cell's two sequences come from the same frames and
    the same masks.
    """
    ref = pred[(NAMES[0], "phase")]
    assert (pred[(NAMES[0], "dark")]["uid"] == ref["uid"]).all(), \
        "the two modalities must be scored on the same cells in the same order"
    out = {}
    # per architecture, phase and dark-field of that architecture only
    for name in NAMES:
        out[(name, "fusion")] = dict(
            y=ref["y"], rec=ref["rec"], uid=ref["uid"],
            Ps=[(pred[(name, "phase")]["Ps"][s] + pred[(name, "dark")]["Ps"][s]) / 2
                for s in range(len(seeds))])
    # the full ensemble, eight models
    out[("Voting", "fusion")] = dict(
        y=ref["y"], rec=ref["rec"], uid=ref["uid"],
        Ps=[np.mean([pred[(n, m)]["Ps"][s] for n in NAMES for m in MODALITIES], 0)
            for s in range(len(seeds))])
    return out


def per_recording_recall(entry, seed):
    P, y, rec = entry["Ps"][seed], entry["y"], entry["rec"]
    pred = P.argmax(1)
    return {r: float((pred[rec == r] == y[rec == r]).mean()) for r in np.unique(rec)}


def main():
    ap = argparse.ArgumentParser(description="Voting, fusion and the result tables.")
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="LORO", choices=["LORO", "WITHIN"])
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(a.seeds)

    pred = add_voting(load_predictions(a.models, a.protocol, a.fold, seeds), seeds)
    rows = [dict(model=m, modality=mod, seed=s, **metrics(d["y"], d["Ps"][s]))
            for (m, mod), d in pred.items() for s in range(len(seeds))]

    # the held-out protocol also gets the cross-modality comparison
    if a.protocol != "WITHIN":
        fusion = cross_modality_fusion(pred, seeds)
        rows += [dict(model=m, modality="fusion", seed=s, **metrics(d["y"], d["Ps"][s]))
                 for (m, _), d in fusion.items() for s in range(len(seeds))]

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / f"metrics_per_seed_{a.protocol}.csv", index=False)

    table = (per_seed.drop(columns="seed").groupby(["modality", "model"])
             .agg(["mean", "std"]))
    table.to_csv(out / f"metrics_mean_std_{a.protocol}.csv")

    print(f"=== {a.protocol} · mean ± sd over {len(seeds)} seeds ===")
    for mod in per_seed.modality.unique():
        for m in NAMES + ["Voting"]:
            g = per_seed[(per_seed.model == m) & (per_seed.modality == mod)]
            if not len(g):
                continue
            print(f"  {mod:7s} {m:28s} balanced {g.balanced_acc.mean():.3f} "
                  f"± {g.balanced_acc.std():.3f}   F1 {g.macro_F1.mean():.3f}   "
                  f"AUC {g.macro_AUC.mean():.3f}")
    print("wrote", out / f"metrics_per_seed_{a.protocol}.csv")


if __name__ == "__main__":
    main()
