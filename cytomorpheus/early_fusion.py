"""Early fusion, the control experiment for the fusion strategy.

Here one network sees both modalities at once, as a six-channel input built by
stacking the three phase-contrast channels and the three dark-field channels of
the same cell, and is trained jointly on them.  This is the alternative to the
late fusion used in the paper, where the modalities stay independent and only
their probabilities are averaged.

Early fusion is worse on recordings the models have not seen, 0.959 against
0.981, because a jointly trained network can lean on whichever channel fits the
training dishes and that habit does not transfer.  It also cannot run when only
one modality was recorded.

    python -m cytomorpheus.early_fusion --data 00_data --out 01_models_fusion \
           --fold F1 --seeds 0 1 2
"""
import argparse
import gc
import math
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Dataset, build_batch, get_split, CLASSES
from .models import MODELS, NAMES
from .train import EMA, balanced_accuracy


def build_batch_6ch(ds, idx, aug, resolution):
    """Six channels, the three phase channels followed by the three dark-field ones.

    The same augmentation draw must apply to both modalities, so the random
    state is reset between the two calls.
    """
    state = torch.cuda.get_rng_state(ds.device) if aug and ds.device != "cpu" else None
    cpu_state = torch.get_rng_state() if aug else None
    a = build_batch(ds, idx, aug, 0, resolution)
    if aug:
        torch.set_rng_state(cpu_state)
        if state is not None:
            torch.cuda.set_rng_state(state, ds.device)
    b = build_batch(ds, idx, aug, 1, resolution)
    return torch.cat([a, b], dim=1)


def widen_first_conv(model):
    """Turn the first 3-channel convolution into a 6-channel one.

    The pretrained weights are duplicated and halved, so the layer starts from
    the same response as the 3-channel model.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)) and m.in_channels == 3:
            w = m.weight.data
            m.weight = nn.Parameter(torch.cat([w, w], 1) / 2)
            m.in_channels = 6
            return model
    raise RuntimeError("no 3-channel input convolution found")


def run_fusion(ds, name, fold, seed, out_root, protocol="LORO", device="cuda"):
    cfg = MODELS[name]
    R, BS, cl3d, epochs = cfg["R"], cfg["BS"], cfg["cl3d"], cfg["EP"]
    out_dir = pathlib.Path(out_root) / f"{name}_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fusion_{protocol}_{fold}_s{seed}"

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    tr, val, te = get_split(ds, protocol, fold, seed)

    cnt = np.bincount(ds.y[tr], minlength=3).astype(np.float64)
    cw = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    fmt = ((lambda t: t.contiguous(memory_format=torch.channels_last_3d)) if cl3d
           else (lambda t: t))

    @torch.no_grad()
    def evaluate(model, ix, tta=1, bs=64):
        model.eval()
        out = []
        for i in range(0, len(ix), bs):
            b = torch.from_numpy(ix[i:i + bs]).to(device)
            acc = 0
            for r in range(tta):
                x = build_batch_6ch(ds, b, False, R)
                if r:
                    x = torch.rot90(x, r, dims=(3, 4))
                with torch.autocast("cuda", torch.bfloat16):
                    acc = acc + model(fmt(x)).float()
            out.append((acc / tta).cpu())
        P = torch.cat(out).numpy()
        return balanced_accuracy(ds.y[ix], P.argmax(1)), P

    net = widen_first_conv(cfg["ctor"]()).to(device)
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
            x = build_batch_6ch(ds, b, True, R)
            yt = torch.from_numpy(ds.y[ii]).to(device)
            t = F.one_hot(yt, 3).float() * .95 + .05 / 3
            lam = float(np.random.beta(.2, .2))
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

    net.load_state_dict(best[1]); net.to(device)
    test_bal, P = evaluate(net, te, tta=4)
    np.savez(out_dir / f"pred_{tag}.npz", logit=P, y=ds.y[te], rec=ds.rec[te], uid=ds.uid[te])
    pd.DataFrame(hist).to_csv(out_dir / f"hist_{tag}.csv", index=False)
    torch.save(best[1], out_dir / f"model_{tag}.pt")

    row = dict(model=name, input="early fusion", fold=fold, seed=seed,
               test_bal=round(test_bal, 4), secs=round(time.time() - t0))
    del net, opt, ema, best
    gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description="Train the early-fusion control models.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--protocol", default="LORO", choices=["LORO", "WITHIN"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=NAMES, choices=NAMES)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ds = Dataset(a.data, a.device)
    rows = []
    for name in a.models:
        for seed in a.seeds:
            r = run_fusion(ds, name, a.fold, seed, a.out, a.protocol, a.device)
            rows.append(r)
            print(f"  {r['model']:28s} seed {seed}  balanced accuracy {r['test_bal']:.3f}")
    pd.DataFrame(rows).to_csv(pathlib.Path(a.out) / "runs_early_fusion.csv", index=False)


if __name__ == "__main__":
    main()
