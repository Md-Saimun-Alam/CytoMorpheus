"""Training.

One architecture, one modality, one protocol, one seed per call.  Every run
uses the same data, augmentation, optimiser and splits, so differences in
performance reflect the architecture and the imaging modality and not the
training procedure.

    python -m cytomorpheus.train --data 00_data --out 01_models \
           --protocol LORO --fold F1 --seeds 0 1 2

Running the command above with both protocols reproduces the 24 runs of the
paper, four architectures x two modalities x three seeds.
"""
import argparse
import gc
import math
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import Dataset, build_batch, get_split, CLASSES
from .models import MODELS, NAMES

MODALITIES = ["phase", "dark"]


class EMA:
    """Exponential moving average of the weights, kept alongside training."""

    def __init__(self, model, decay=0.999):
        self.d, self.t = decay, 0
        self.s = {k: v.detach().float().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        self.t += 1
        d = min(self.d, (1. + self.t) / (10. + self.t))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.s[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.s[k] = v.detach().clone().float()

    def state(self, model):
        return {k: self.s[k].to(v.dtype) for k, v in model.state_dict().items()}


def balanced_accuracy(y_true, y_pred):
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


def run(ds, name, modality, protocol, fold, seed, out_root, device="cuda"):
    """Train one model and write its predictions, history and weights."""
    cfg = MODELS[name]
    R, BS, cl3d, epochs = cfg["R"], cfg["BS"], cfg["cl3d"], cfg["EP"]
    out_dir = pathlib.Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{MODALITIES[modality]}_{protocol}_{fold}_s{seed}"

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    tr, val, te = get_split(ds, protocol, fold, seed)

    # class-weighted loss, the classes are unbalanced
    cnt = np.bincount(ds.y[tr], minlength=3).astype(np.float64)
    cw = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    fmt = ((lambda t: t.contiguous(memory_format=torch.channels_last_3d)) if cl3d
           else (lambda t: t))

    @torch.no_grad()
    def evaluate(model, ix, tta=1, bs=64):
        """Balanced accuracy and probabilities, optionally averaged over rotations."""
        model.eval()
        out = []
        for i in range(0, len(ix), bs):
            b = torch.from_numpy(ix[i:i + bs]).to(device)
            acc = 0
            for r in range(tta):
                x = build_batch(ds, b, False, modality, R)
                if r:
                    x = torch.rot90(x, r, dims=(3, 4))
                with torch.autocast("cuda", torch.bfloat16):
                    acc = acc + model(fmt(x)).float()
            out.append((acc / tta).cpu())
        P = torch.cat(out).numpy()
        return balanced_accuracy(ds.y[ix], P.argmax(1)), P

    net = cfg["ctor"]().to(device)
    if cl3d:
        net = net.to(memory_format=torch.channels_last_3d)

    head_ids = {id(p) for p in net.head.parameters()}
    opt = torch.optim.AdamW(
        [{"params": [p for p in net.parameters() if id(p) not in head_ids], "lr": cfg["lr_bb"]},
         {"params": [p for p in net.parameters() if id(p) in head_ids], "lr": cfg["lr_hd"]}],
        weight_decay=.05)

    steps = max(1, len(tr) // BS)
    total, warm = steps * epochs, steps * 2
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    ema = EMA(net)
    hist, best, smooth, t0 = [], (-1, None), None, time.time()

    for ep in range(1, epochs + 1):
        net.train()
        perm = rng.permutation(tr)
        for i in range(steps):
            ii = perm[i * BS:(i + 1) * BS]
            if len(ii) < 2:
                continue
            b = torch.from_numpy(ii).to(device)
            x = build_batch(ds, b, True, modality, R)
            yt = torch.from_numpy(ds.y[ii]).to(device)
            t = F.one_hot(yt, 3).float() * .95 + .05 / 3          # label smoothing
            lam = float(np.random.beta(.2, .2))                   # mixup
            p_ = torch.randperm(len(b), device=device)
            x = fmt(lam * x + (1 - lam) * x[p_])
            t = lam * t + (1 - lam) * t[p_]
            with torch.autocast("cuda", torch.bfloat16):
                o = net(x)
            loss = -((t * cw) * F.log_softmax(o.float(), 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); ema.update(net)

        # keep whichever of the raw and the averaged weights validates better
        vb_raw, _ = evaluate(net, val)
        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema.state(net))
        vb_ema, _ = evaluate(net, val)
        use_ema = vb_ema >= vb_raw
        if not use_ema:
            net.load_state_dict(backup)
        vb = max(vb_ema, vb_raw)
        smooth = vb if smooth is None else .7 * smooth + .3 * vb
        if smooth > best[0]:
            best = (smooth, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        if use_ema:
            net.load_state_dict(backup)
        hist.append(dict(ep=ep, val_bal=vb))

    # score the test cells once, with four-rotation test-time augmentation
    net.load_state_dict(best[1]); net.to(device)
    test_bal, P = evaluate(net, te, tta=4)
    pred, y_true = P.argmax(1), ds.y[te]

    np.savez(out_dir / f"pred_{tag}.npz", logit=P, y=y_true,
             rec=ds.rec[te], uid=ds.uid[te])
    pd.DataFrame(hist).to_csv(out_dir / f"hist_{tag}.csv", index=False)
    torch.save(best[1], out_dir / f"model_{tag}.pt")

    row = dict(model=name, modality=MODALITIES[modality], protocol=protocol, fold=fold,
               seed=seed, test_bal=round(test_bal, 4), val_bal=round(best[0], 4),
               secs=round(time.time() - t0))
    for c in range(3):
        row[f"recall_{CLASSES[c]}"] = round(float((pred[y_true == c] == c).mean()), 3)

    del net, opt, ema, best
    gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description="Train the CytoMorpheus models.")
    ap.add_argument("--data", required=True, help="folder with cache.npy and meta.npz")
    ap.add_argument("--out", required=True, help="folder for weights and predictions")
    ap.add_argument("--protocol", default="LORO", choices=["LORO", "WITHIN"],
                    help="LORO = held-out recordings, WITHIN = within-experiment")
    ap.add_argument("--fold", default="F1", help="held-out fold name, ignored for WITHIN")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=NAMES, choices=NAMES)
    ap.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ds = Dataset(a.data, a.device)
    print(f"{len(ds)} cells  {ds.counts()}")

    rows = []
    for name in a.models:
        for mod in a.modalities:
            for seed in a.seeds:
                r = run(ds, name, MODALITIES.index(mod), a.protocol, a.fold, seed,
                        a.out, a.device)
                rows.append(r)
                print(f"  {r['model']:28s} {r['modality']:5s} seed {seed}  "
                      f"balanced accuracy {r['test_bal']:.3f}  ({r['secs']} s)")
    out = pathlib.Path(a.out) / f"runs_{a.protocol}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
