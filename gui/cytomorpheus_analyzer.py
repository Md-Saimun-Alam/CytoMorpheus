"""
CytoMorpheus Analyzer — label-free classification of cell death from phase-contrast and/or dark-field video.

Give it one or two videos of the same field (any order); the modality of each is detected from the image itself.

Pipeline (identical to the training pipeline):
  1. crop valid rows 35:927 (timestamp / scale bar removed)
  2. Cellpose-SAM on every frame (full resolution, flow 0.4, cellprob +1) — on phase contrast when available
  3. link masks frame-to-frame: mutual-best IoU >= 0.10, gap closing <= 2 frames, 12 um step cap
  4. per track: 48 x 48 px crops (phase, dark-field, target mask) on the tracked centroid
  5. morphological death-event detector -> 15-frame window t-10 ... t+4 (no event -> middle of the track)
  6. three channels per modality: per-clip z-score, frame difference x3, mask dilated 2 px (+-1)
  7. four architectures x seeds per modality, 4-rotation averaging; probabilities averaged within a
     modality and, when both are present, across modalities

Usage:
  python cytomorpheus_v3.py                                     -> GUI
  python cytomorpheus_v3.py VIDEO [VIDEO2] [--out DIR] [--seeds 0 1 2] [--models DIR] [--no-video]
  python cytomorpheus_v3.py --open RESULTS_DIR
"""
import sys, os, json, time, math, argparse, pathlib, threading, queue, subprocess
os.environ["MPLBACKEND"] = "TkAgg"     # ignore an inherited Jupyter backend; this app draws into Tk
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
if sys.stdout is None: sys.stdout = open(os.devnull, "w")          # windowed exe has no console
if sys.stderr is None: sys.stderr = open(os.devnull, "w")
APP_DIR = pathlib.Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else pathlib.Path(__file__).resolve().parent   # next to the exe / script
RES_DIR = pathlib.Path(getattr(sys, "_MEIPASS", APP_DIR))                                                                                # files bundled into the exe

# ----------------------------------------------------------------------------- locked config
Y0, Y1 = 35, 927
UM_PX = 0.933
DT_MIN = 3
FLOW, CELLPROB = 0.4, 1.0
IOU_MIN, MAX_GAP, MAX_STEP_PX = 0.10, 2, 12.0 / UM_PX
BOX, PRE, POST = 48, 10, 4
LW = PRE + POST + 1                     # 15
STRENGTH_MIN = 6.0
DARK_MEDIAN_MAX = 100                   # a frame whose median intensity is below this is dark-field
CLS = ["Apoptosis", "Alive", "Necrosis"]   # "Alive" is the control class of the paper
CLS_RGB = {"Apoptosis": (0.16, 0.44, 0.75), "Alive": (0.13, 0.60, 0.33), "Necrosis": (0.82, 0.13, 0.15), "Unclassified": (0.90, 0.49, 0.13)}
CLS_HEX = {k: "#%02x%02x%02x" % tuple(int(255 * c) for c in v) for k, v in CLS_RGB.items()}
CLS_BGR = {k: tuple(int(255 * c) for c in v[::-1]) for k, v in CLS_RGB.items()}
NAVY, BLUE, LIGHTBLUE, ORANGE, INK, PAPER, CARD, LINE = "#12365a", "#2a6fba", "#9cc3e6", "#e8842c", "#1c2a38", "#e4ebf4", "#ffffff", "#c7d3e2"
PROB_COL = {"Phase contrast": "#9cc3e6", "Dark-field": "#12365a", "Combined": "#e8842c"}
MODN = {"phase": "Phase contrast", "dark": "Dark-field"}
NAMES = ["3DCNN", "AlexNet-BiLSTM", "MobileNetV2", "EfficientNetB0-Transformer"]
CFG = {"3DCNN": dict(R=48, cl3d=True), "AlexNet-BiLSTM": dict(R=96, cl3d=False),
       "MobileNetV2": dict(R=96, cl3d=False), "EfficientNetB0-Transformer": dict(R=96, cl3d=False)}
dev = "cuda" if torch.cuda.is_available() else "cpu"

def find_models():
    here = APP_DIR
    for c in [here.parent / "01_models", here / "01_models", here / "models", here.parent / "models"]:
        if (c / NAMES[0]).exists(): return c
    return None

class Cancelled(Exception): pass

# ----------------------------------------------------------------------------- models (as trained)
def blk(i, o, st):
    return nn.Sequential(nn.Conv3d(i,o,(1,3,3),(1,st,st),(0,1,1),bias=False), nn.BatchNorm3d(o), nn.SiLU(True),
                         nn.Conv3d(o,o,(3,1,1),1,(1,0,0),bias=False), nn.BatchNorm3d(o), nn.SiLU(True))
class Net3D(nn.Module):
    def __init__(self, nc=3):
        super().__init__(); self.f = nn.Sequential(blk(3,64,1), blk(64,96,2), blk(96,192,2), blk(192,320,2), blk(320,512,2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.BatchNorm1d(512), nn.Dropout(.3), nn.Linear(512, nc))
    def forward(self, x): return self.head(self.f(x))
def frames(back, x):
    B, C, T_, H, W = x.shape; return back(x.transpose(1, 2).reshape(B * T_, C, H, W)).reshape(B, T_, -1)
class BiLSTMPool(nn.Module):
    def __init__(self, d, hid=256): super().__init__(); self.lstm = nn.LSTM(d, hid, batch_first=True, bidirectional=True)
    def forward(self, f): return self.lstm(f)[0].mean(1)
class TransformerPool(nn.Module):
    def __init__(self, d, dm=256, heads=4, layers=2, T_=LW):
        super().__init__(); self.proj = nn.Linear(d, dm); self.pos = nn.Parameter(torch.zeros(1, T_, dm))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=.1, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dm)
    def forward(self, f): return self.norm(self.enc(self.proj(f) + self.pos)).mean(1)
class AlexNetBiLSTM(nn.Module):
    def __init__(self, nc=3):
        super().__init__(); from torchvision.models import alexnet; m = alexnet(weights=None)
        self.back = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()); self.head = nn.Sequential(BiLSTMPool(256), nn.Dropout(.5), nn.Linear(512, nc))
    def forward(self, x): return self.head(frames(self.back, x))
class MobileNetV2TP(nn.Module):
    def __init__(self, nc=3):
        super().__init__(); from torchvision.models import mobilenet_v2; m = mobilenet_v2(weights=None)
        self.back = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()); self.head = nn.Sequential(nn.Dropout(.5), nn.Linear(1280, nc))
    def forward(self, x): return self.head(frames(self.back, x).mean(1))
class EffB0Transformer(nn.Module):
    def __init__(self, nc=3):
        super().__init__(); from torchvision.models import efficientnet_b0; m = efficientnet_b0(weights=None)
        self.back = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()); self.head = nn.Sequential(TransformerPool(1280), nn.Dropout(.5), nn.Linear(256, nc))
    def forward(self, x): return self.head(frames(self.back, x))
CTOR = {"3DCNN": Net3D, "AlexNet-BiLSTM": AlexNetBiLSTM, "MobileNetV2": MobileNetV2TP, "EfficientNetB0-Transformer": EffB0Transformer}

def load_models(models_dir, modalities, seeds):
    if models_dir is None: raise FileNotFoundError("trained models not found — set the models folder in Settings")
    nets = {}
    for mod in modalities:
        for n in NAMES:
            for s in seeds:
                p = pathlib.Path(models_dir) / n / f"model_{mod}_LORO_F1_s{s}.pt"
                if not p.exists(): p = pathlib.Path(models_dir) / n / f"model_{mod}_LORO_F1v3_s{s}.pt"
                if not p.exists(): raise FileNotFoundError(p)
                net = CTOR[n]().to(dev); net.load_state_dict(torch.load(p, map_location=dev)); net.eval()
                if CFG[n]["cl3d"]: net = net.to(memory_format=torch.channels_last_3d)
                nets[(mod, n, s)] = net
    return nets

# ----------------------------------------------------------------------------- video + modality detection
def read_video(p, log=print):
    cap = cv2.VideoCapture(str(p)); S = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        S.append(g[Y0:Y1] if g.shape[0] >= Y1 else g)
    cap.release()
    if not S: raise IOError(f"no frames read from {p}")
    log(f"  {pathlib.Path(p).name}: {len(S)} frames, {S[0].shape[1]} x {S[0].shape[0]} px")
    return np.stack(S)

def modality_of(frame):
    """dark-field has a dark background (low median); phase contrast a bright one"""
    return "dark" if float(np.median(frame)) < DARK_MEDIAN_MAX else "phase"

def peek_modality(p):
    cap = cv2.VideoCapture(str(p)); ok, fr = cap.read(); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if not ok: return None, 0
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
    return modality_of(g), n

# ----------------------------------------------------------------------------- segmentation + tracking
def segment(img, log=print, progress=None, cancel=None):
    from cellpose import models
    model = models.CellposeModel(gpu=(dev == "cuda")); masks = np.zeros(img.shape, np.int32); t0 = time.time()
    for t in range(len(img)):
        if cancel and cancel(): raise Cancelled()
        m = model.eval(img[t], flow_threshold=FLOW, cellprob_threshold=CELLPROB)[0]; masks[t] = m
        if progress: progress("Finding cells", t + 1, len(img))
        if t % 10 == 0: log(f"  frame {t+1}/{len(img)}: {int(m.max())} cells  ({time.time()-t0:.0f} s)")
    return masks

def _props(lab):
    n = int(lab.max()); area = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    ys, xs = np.nonzero(lab); v = lab[ys, xs]
    cy = np.bincount(v, weights=ys, minlength=n + 1) / np.maximum(area, 1); cx = np.bincount(v, weights=xs, minlength=n + 1) / np.maximum(area, 1)
    return area, cy, cx

def _iou(a, b):
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    inter = np.bincount((a.astype(np.int64) * nb + b).ravel(), minlength=na * nb).reshape(na, nb).astype(float)
    aa, bb = inter.sum(1, keepdims=True), inter.sum(0, keepdims=True)
    return inter / np.maximum(aa + bb - inter, 1)

def link_tracks(masks, log=print, progress=None):
    T = len(masks); labels = np.zeros(masks.shape, np.int32); next_id = 1; last = {}
    for t in range(T):
        lab = masks[t]; area, cy, cx = _props(lab); n = int(lab.max()); assigned = np.zeros(n + 1, bool)
        for g in range(1, MAX_GAP + 2):
            if t - g < 0: break
            cand = [tid for tid, (f, l, y, x) in last.items() if f == t - g]
            if not cand: continue
            prev = masks[t - g]; iou = _iou(prev, lab)
            cand_lab = np.array([last[tid][1] for tid in cand]); sub = iou[cand_lab]; sub[:, 0] = 0; sub[:, assigned] = 0
            best_b = sub.argmax(1); best_a = sub.argmax(0)
            for i, tid in enumerate(cand):
                j = best_b[i]
                if j == 0 or sub[i, j] < IOU_MIN or best_a[j] != i: continue
                if math.hypot(cy[j] - last[tid][2], cx[j] - last[tid][3]) > MAX_STEP_PX * g: continue
                labels[t][lab == j] = tid; assigned[j] = True; last[tid] = (t, j, cy[j], cx[j])
        for j in range(1, n + 1):
            if not assigned[j]: labels[t][lab == j] = next_id; last[next_id] = (t, j, cy[j], cx[j]); next_id += 1
        if progress: progress("Following cells through time", t + 1, T)
    log(f"  {next_id - 1} cell tracks")
    return labels

# ----------------------------------------------------------------------------- crops, detector, window
def crop_all(labels, phase, dark):
    """{tid: (st (n,3,48,48) uint8 [phase-or-dark, dark-or-phase, mask], [(frame, cy, cx), ...])}"""
    h = BOX // 2; A = phase if phase is not None else dark; B = dark if dark is not None else phase; per = {}
    for t in range(len(labels)):
        lab = labels[t]; area, cy, cx = _props(lab); ids = np.nonzero(area[1:])[0] + 1
        P = np.pad(A[t], h); D = np.pad(B[t], h); L = np.pad(lab, h)
        for tid in ids:
            y, x = int(round(cy[tid])), int(round(cx[tid])); sl = (slice(y, y + BOX), slice(x, x + BOX))
            per.setdefault(int(tid), []).append((t, y, x, np.stack([P[sl], D[sl], (L[sl] == tid).astype(np.uint8) * 255])))
    return {tid: (np.stack([e[3] for e in v]), [(e[0], e[1], e[2]) for e in v]) for tid, v in per.items()}

def _series(st):
    T_ = st.shape[0]; cr = np.full(T_, np.nan); mr = np.full(T_, np.nan)
    d = st[:, 1].astype(np.float32); zd = (d - d.mean((1, 2), keepdims=True)) / (d.std((1, 2), keepdims=True) + 1e-5)
    for t in range(T_):
        m = st[t, 2] > 0
        if m.sum() < 30: continue
        core = ndi.binary_erosion(m, iterations=3); rim = m & ~core
        if core.sum() < 10 or rim.sum() < 10: continue
        g = st[t, 0].astype(np.float32); cr[t] = (g[core].mean() - g[rim].mean()) / (g[m].std() + 1e-3)
        if t > 0: mr[t] = np.abs(zd[t] - zd[t - 1])[rim].mean()
    return cr, mr
def _fill(f): return pd.Series(f).interpolate(limit_direction="both").bfill().ffill().values
def _step(f, sign, W=3):
    f = _fill(f); d = np.diff(f); mad = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-6; s = np.full(len(f), np.nan)
    for t in range(W, len(f) - W): s[t] = sign * (np.median(f[t:t + W]) - np.median(f[t - W:t])) / mad
    return s
def _spike(f):
    f = _fill(f); med = np.median(f); mad = 1.4826 * np.median(np.abs(f - med)) + 1e-6; return (f - med) / mad

def anchor_window(st, has_phase=True):
    n = st.shape[0]
    if n < LW: return None
    cr, mr = _series(st)
    S = np.clip(np.nan_to_num(_spike(mr), nan=0), 0, None)
    if has_phase: S = S + np.nan_to_num(_step(cr, -1), nan=-np.inf)      # the phase-contrast collapse is only defined in phase
    S[:PRE] = -np.inf; S[n - POST:] = -np.inf
    if np.isfinite(S).any() and S.max() > STRENGTH_MIN: a = int(np.argmax(S)); strength = float(S.max())
    else: a = int(min(max(n // 2, PRE), n - POST - 1)); strength = float(S.max()) if np.isfinite(S).any() else 0.0
    return a, strength, slice(a - PRE, a - PRE + LW)

# ----------------------------------------------------------------------------- inference
def build(clip, ch, R, HALO=2):
    """clip (B,15,3,48,48) uint8 -> (B,3,15,R,R) [z, 3*dz, 2*mask-1] from channel ch — as in training"""
    xb = torch.from_numpy(clip).to(dev); B, T_ = xb.shape[:2]
    g = xb[:, :, ch].float() / 255.; m = (xb[:, :, 2] > 0).float()
    m = F.max_pool2d(m.reshape(-1, 1, BOX, BOX), 2 * HALO + 1, 1, HALO).reshape(B, T_, BOX, BOX)
    z = (g - g.mean((1, 2, 3), keepdim=True)) / (g.std((1, 2, 3), keepdim=True) + 1e-5)
    dz = z - torch.roll(z, 1, dims=1); dz[:, 0] = 0
    x = torch.stack([z, 3 * dz, 2 * m - 1], 1)
    if R != BOX: x = F.interpolate(x, size=(T_, R, R), mode="trilinear", align_corners=False)
    return x

@torch.no_grad()
def predict(nets, clips, chan, tta=4, bs=32, progress=None, cancel=None):
    """chan: {modality: channel index in the clip}. returns {modality: (N,3) probabilities}"""
    out = {}; total = len(nets); done = 0
    for mod, ch in chan.items():
        acc = np.zeros((len(clips), 3)); k = 0
        for (m_, n, s), net in nets.items():
            if m_ != mod: continue
            if cancel and cancel(): raise Cancelled()
            R = CFG[n]["R"]; fmt = (lambda t: t.contiguous(memory_format=torch.channels_last_3d)) if CFG[n]["cl3d"] else (lambda t: t); P = []
            for i in range(0, len(clips), bs):
                x = build(clips[i:i + bs], ch, R); a = 0
                for r in range(tta):
                    xr = torch.rot90(x, r, dims=(3, 4)) if r else x
                    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")): a = a + net(fmt(xr)).float()
                P.append((a / tta).cpu().numpy())
            P = np.concatenate(P); e = np.exp(P - P.max(1, keepdims=True)); acc += e / e.sum(1, keepdims=True); k += 1; done += 1
            if progress: progress(f"Classifying ({MODN[mod]}: {n}, seed {s})", done, total)
        out[mod] = acc / max(k, 1)
    return out

# ----------------------------------------------------------------------------- main pipeline
def analyze_full(videos, out_dir=None, models_dir=None, seeds=(0, 1, 2), log=print, progress=None, cancel=None, write_video=True):
    videos = [str(v) for v in ([videos] if isinstance(videos, (str, pathlib.Path)) else videos) if v]
    if not videos: raise ValueError("no video given")
    if len(videos) > 2: raise ValueError("give at most two videos (one field of view, one or two modalities)")
    models_dir = models_dir or find_models(); t_all = time.time()
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path(videos[0]).parent / f"CytoMorpheus_{pathlib.Path(videos[0]).stem}"; out_dir.mkdir(parents=True, exist_ok=True)
    log(f"CytoMorpheus Analyzer · {dev.upper()}"); log("reading videos"); src = {}
    for v in videos:
        arr = read_video(v, log); mod = modality_of(arr[0]); log(f"    detected: {MODN[mod]} (median intensity {int(np.median(arr[0]))})")
        if mod in src: raise ValueError(f"both videos look like {MODN[mod]} — give one phase-contrast and one dark-field video of the same field")
        src[mod] = (arr, v)
    phase = src["phase"][0] if "phase" in src else None; dark = src["dark"][0] if "dark" in src else None
    if phase is not None and dark is not None and phase.shape != dark.shape:
        n = min(len(phase), len(dark)); log(f"  WARNING frame counts differ ({len(phase)} vs {len(dark)}); using the first {n}"); phase, dark = phase[:n], dark[:n]
    modalities = [m for m in ["phase", "dark"] if m in src]; seg_src = "phase" if phase is not None else "dark"
    log(f"finding cells (Cellpose-SAM on {MODN[seg_src]})"); masks = segment(phase if phase is not None else dark, log, progress, cancel); np.savez_compressed(out_dir / "masks.npz", masks=masks.astype(np.uint16))
    log("following cells through time"); labels = link_tracks(masks, log, progress); np.savez_compressed(out_dir / "tracks.npz", labels=labels.astype(np.uint16))
    log("loading classifiers"); nets = load_models(models_dir, modalities, seeds)
    df, clips_by_tid, cents = classify_tracks(labels, phase, dark, nets, modalities, log, progress, cancel); df.to_csv(out_dir / "cells.csv", index=False)
    summary = make_summary(df, src, len(labels), modalities, seg_src, seeds, time.time() - t_all); json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    log("result: " + ", ".join(f"{k} {v}" for k, v in summary["counts"].items()))
    if write_video: log("writing annotated video"); write_annotated(phase if phase is not None else dark, labels, df, out_dir / "annotated.avi", progress)
    log(f"done in {time.time() - t_all:.0f} s -> {out_dir}")
    return dict(df=df, summary=summary, phase=phase, dark=dark, labels=labels, clips=clips_by_tid, cents=cents, out_dir=out_dir)

def classify_tracks(labels, phase, dark, nets, modalities, log=print, progress=None, cancel=None):
    rows, clips, keep, clips_by_tid = [], [], [], {}
    log("cutting out each cell"); crops = crop_all(labels, phase, dark); has_phase = phase is not None; cents = {tid: c for tid, (st, c) in crops.items()}
    chan = {"phase": 0, "dark": (1 if has_phase else 0)} if dark is not None else {"phase": 0}
    if not has_phase: chan = {"dark": 0}
    for tid in sorted(crops):
        st, cent = crops[tid]; aw = anchor_window(st, has_phase); n = st.shape[0]
        base = dict(track=int(tid), n_frames=n, first_frame=cent[0][0], last_frame=cent[-1][0], cy=int(np.mean([c[1] for c in cent])), cx=int(np.mean([c[2] for c in cent])))
        if aw is None:
            rows.append(dict(base, anchor_frame=-1, event_strength=np.nan, window_start=-1, window_end=-1, status="too short (<15 frames)")); continue
        a, strength, sl = aw; clips.append(st[sl]); keep.append(len(rows)); clips_by_tid[int(tid)] = st[sl]
        rows.append(dict(base, anchor_frame=int(cent[a][0]), event_strength=round(strength, 2), window_start=int(cent[sl.start][0]), window_end=int(cent[sl.stop - 1][0]),
                         status="event" if strength > STRENGTH_MIN else "no event (window centred)"))
    df = pd.DataFrame(rows); df["class"] = "Unclassified"; df["confidence"] = 0.0; log(f"  {len(clips)} cells long enough to classify, {len(df) - len(clips)} too short")
    if clips:
        log("classifying"); P = predict(nets, np.stack(clips), chan, progress=progress, cancel=cancel)
        for mod in modalities:
            for c, cn in enumerate(CLS): df.loc[keep, f"p_{mod}_{cn}"] = P[mod][:, c]
            df.loc[keep, f"class_{mod}"] = [CLS[i] for i in P[mod].argmax(1)]
        Pf = np.mean([P[m] for m in modalities], 0)
        for c, cn in enumerate(CLS): df.loc[keep, f"p_fused_{cn}"] = Pf[:, c]
        df.loc[keep, "class"] = [CLS[i] for i in Pf.argmax(1)]; df.loc[keep, "confidence"] = Pf.max(1)
    return df, clips_by_tid, cents

def make_summary(df, src, n_frames, modalities, seg_src, seeds, secs):
    counts = df["class"].value_counts().to_dict(); n_cls = int((df["class"] != "Unclassified").sum())
    return dict(videos={MODN[m]: v for m, (a, v) in src.items()}, phase_video=src["phase"][1] if "phase" in src else None, dark_video=src["dark"][1] if "dark" in src else None,
                frames=int(n_frames), minutes=int(n_frames * DT_MIN), tracks=int(len(df)), classified=n_cls, counts={k: int(v) for k, v in counts.items()},
                fractions={k: round(v / max(n_cls, 1), 4) for k, v in counts.items() if k != "Unclassified"}, modalities=[MODN[m] for m in modalities],
                segmentation=MODN[seg_src], seeds=list(seeds), models=NAMES, decision="both modalities combined" if len(modalities) == 2 else f"{MODN[modalities[0]]} only", seconds=round(secs))

def analyze(*a, **k):
    r = analyze_full(*a, **k); return r["df"], r["summary"]

def load_results(out_dir, log=print):
    out_dir = pathlib.Path(out_dir); summary = json.load(open(out_dir / "summary.json")); df = pd.read_csv(out_dir / "cells.csv")
    if "window_start" not in df.columns:
        w = df.get("window", pd.Series([""] * len(df))).astype(str).str.split("-", expand=True)
        df["window_start"] = pd.to_numeric(w[0], errors="coerce").fillna(-1).astype(int); df["window_end"] = pd.to_numeric(w[1] if w.shape[1] > 1 else np.nan, errors="coerce").fillna(-1).astype(int)
    for c in ["class", "class_phase", "class_dark"]:
        if c in df.columns: df[c] = df[c].replace({"Control": "Alive"})
    if "counts" in summary: summary["counts"] = {("Alive" if k == "Control" else k): v for k, v in summary["counts"].items()}
    if "fractions" in summary: summary["fractions"] = {("Alive" if k == "Control" else k): v for k, v in summary["fractions"].items()}
    for c, dflt in {"class": "Unclassified", "confidence": 0.0, "status": "", "anchor_frame": -1, "event_strength": np.nan, "cy": -1, "cx": -1}.items():
        if c not in df.columns: df[c] = dflt
    phase = read_video(summary["phase_video"], log) if summary.get("phase_video") else None; dark = read_video(summary["dark_video"], log) if summary.get("dark_video") else None
    labels = np.load(out_dir / "tracks.npz")["labels"].astype(np.int32); crops = crop_all(labels, phase, dark); clips = {}; cents = {tid: c for tid, (st, c) in crops.items()}
    for r in df.itertuples():
        if r.anchor_frame >= 0 and r.track in crops:
            st, cent = crops[r.track]; fr = [c[0] for c in cent]; i0 = fr.index(int(r.window_start)); clips[int(r.track)] = st[i0:i0 + LW]
            if r.cy < 0: df.loc[df.track == r.track, ["cy", "cx"]] = [int(np.mean([c[1] for c in cent])), int(np.mean([c[2] for c in cent]))]
    return dict(df=df, summary=summary, phase=phase, dark=dark, labels=labels, clips=clips, cents=cents, out_dir=out_dir)

def class_lut(df, max_tid):
    """track id -> (class index 0..3 with 3 = Unclassified, window start, death frame).

    A cell is drawn in the Alive colour before its 15-frame window, blends into its class
    colour across the window, and keeps the class colour from the detected death frame on.
    Alive and Unclassified cells keep one colour throughout.
    """
    lut = np.full(max_tid + 1, 3, np.int64); t0 = np.full(max_tid + 1, -1, np.int64); ta = np.full(max_tid + 1, -1, np.int64)
    idx = {c: i for i, c in enumerate(CLS + ["Unclassified"])}
    t = df["track"].astype(int).values; ok = t <= max_tid
    lut[t[ok]] = [idx.get(c, 3) for c in df["class"].values[ok]]
    if "window_start" in df.columns: t0[t[ok]] = df["window_start"].fillna(-1).astype(int).values[ok]
    if "anchor_frame" in df.columns: ta[t[ok]] = df["anchor_frame"].fillna(-1).astype(int).values[ok]
    ta = np.where(ta < 0, t0, ta)
    return lut, t0, ta

def frame_colours(lut, t):
    """RGB colour of every track at frame t, following class_lut"""
    lut, t0, ta = lut
    cols = np.array([CLS_RGB[c] for c in CLS + ["Unclassified"]]); base = cols[lut]; alive = np.array(CLS_RGB["Alive"])
    dying = (lut == 0) | (lut == 2)
    span = np.maximum(ta - t0, 1); f = np.clip((t - t0) / span, 0, 1); f = np.where(t >= ta, 1.0, f); f = np.where(t0 < 0, 1.0, f)
    f = np.where(dying, f, 1.0)[:, None]
    return (1 - f) * alive + f * base

def render_frame(img_stack, labels, df, t, show=None, selected=None, lo=None, hi=None, lut=None):
    if lo is None: lo, hi = np.percentile(img_stack[0], [.5, 99.5])
    g = np.clip((img_stack[t].astype(float) - lo) / (hi - lo + 1e-6), 0, 1); img = np.repeat(g[..., None], 3, 2)
    lab = labels[t]
    if lut is None: lut = class_lut(df, int(labels.max()))
    er = cv2.erode(lab.astype(np.float32), np.ones((3, 3), np.uint8)); border = (lab != er) & (lab > 0)
    border = cv2.dilate(border.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & (lab > 0)
    tid = np.minimum(lab[border], len(lut[0]) - 1); ci = lut[0][tid]; cols_t = frame_colours(lut, t)
    keep = np.ones(len(ci), bool)
    if show is not None: keep = np.isin(ci, [i for i, c in enumerate(CLS + ["Unclassified"]) if c in show])
    ys, xs = np.nonzero(border); img[ys[keep], xs[keep]] = cols_t[tid[keep]]
    if selected is not None and (lab == selected).any():
        m = (lab == selected).astype(np.uint8); ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) - cv2.dilate(m, np.ones((3, 3), np.uint8)); img[ring > 0] = (1, 0.85, 0)
    return img

def write_annotated(img_stack, labels, df, path, progress=None):
    H, W = img_stack.shape[1:]; path = pathlib.Path(path); vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (W, H))
    if not vw.isOpened(): path = path.with_suffix(".mp4"); vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (W, H))
    if not vw.isOpened(): print("  WARNING: no video codec available, annotated video skipped"); return
    lo, hi = np.percentile(img_stack[0], [.5, 99.5])
    for t in range(len(img_stack)):
        fr = cv2.cvtColor((render_frame(img_stack, labels, df, t, lo=lo, hi=hi) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(fr, f"{t * DT_MIN} min", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2); y = 55
        for c in CLS: cv2.putText(fr, c, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .6, CLS_BGR[c], 2); y += 25
        vw.write(fr)
        if progress: progress("Writing annotated video", t + 1, len(img_stack))
    vw.release()

# ----------------------------------------------------------------------------- GUI
LOGO_URL = "https://www.rayresearchlab.com/images/logo.png"

def get_logo(here):
    for p in [here / "logo.png", here / "logo.gif", here / "logo.jpg", here / "logo.jpeg", RES_DIR / "logo.png"]:
        if p.exists(): return p
    if (here / ".no_logo").exists(): return None
    import urllib.request
    for url in [LOGO_URL, LOGO_URL.replace("https://", "http://")]:
        try:
            urllib.request.urlretrieve(url, here / "logo.png")
            if (here / "logo.png").stat().st_size > 500: return here / "logo.png"
        except Exception: pass
    try: (here / ".no_logo").touch()
    except Exception: pass
    return None

def gui(open_dir=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    import matplotlib; matplotlib.use("TkAgg", force=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk(); root.title("CytoMorpheus Analyzer"); root.geometry("1560x940"); root.minsize(1200, 760); root.configure(bg=PAPER)
    for ic in [APP_DIR / "app_icon.ico", RES_DIR / "app_icon.ico"]:                   # window / taskbar icon
        if ic.exists():
            try: root.iconbitmap(default=str(ic)); break
            except Exception: pass
    style = ttk.Style(root)
    try: style.theme_use("clam")
    except Exception: pass
    style.configure(".", background=PAPER, foreground=INK, font=("Segoe UI", 10))
    style.configure("TFrame", background=PAPER); style.configure("Card.TFrame", background=CARD, relief="flat")
    style.configure("TLabel", background=PAPER, foreground=INK); style.configure("Card.TLabel", background=CARD, foreground=INK)
    style.configure("CardTitle.TLabel", background=CARD, foreground=NAVY, font=("Segoe UI", 10, "bold")); style.configure("Sub.TLabel", background=CARD, foreground="#5a6a7a", font=("Segoe UI", 9))
    style.configure("Detail.TLabel", background=CARD, foreground=NAVY, font=("Segoe UI", 10, "bold")); style.configure("Stage.TLabel", background=CARD, foreground=NAVY, font=("Segoe UI", 10, "bold"))
    style.configure("TButton", padding=5, background="#eef2f7", foreground=INK, bordercolor=LINE); style.map("TButton", background=[("active", "#d9e3ef"), ("disabled", "#eef1f5")], foreground=[("disabled", "#9aa7b5")])
    style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=9, background=BLUE, foreground="white", bordercolor=BLUE)
    style.map("Accent.TButton", background=[("active", NAVY), ("disabled", "#b7c9dd")], foreground=[("disabled", "white")])
    style.configure("Play.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 4), background=BLUE, foreground="white", bordercolor=BLUE); style.map("Play.TButton", background=[("active", NAVY)])
    style.configure("TNotebook", background=PAPER, bordercolor=LINE, tabmargins=(2, 4, 2, 0)); style.configure("TNotebook.Tab", padding=(16, 7), font=("Segoe UI", 10, "bold"), background="#d5dfeb", foreground=NAVY)
    style.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", BLUE)])
    style.configure("Treeview", rowheight=24, font=("Segoe UI", 10), background=CARD, fieldbackground=CARD, foreground=INK, bordercolor=LINE)
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#eef2f7", foreground=NAVY, relief="flat"); style.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", "white")])
    style.configure("Horizontal.TProgressbar", troughcolor="#dce4ee", background=BLUE, bordercolor="#dce4ee", lightcolor=BLUE, darkcolor=BLUE)
    style.configure("TCheckbutton", background=CARD); style.configure("Card.TRadiobutton", background=CARD); style.configure("TEntry", fieldbackground="white")
    style.configure("Vertical.TScrollbar", background="#d5dfeb", troughcolor=PAPER, bordercolor=PAPER, arrowcolor=NAVY)
    state = dict(res=None, sel=None, cancel=False, q=queue.Queue(), videos=[], models=find_models(), seeds=(0, 1, 2), write_video=True, lut=None, lohi=None)
    play = {"on": False, "job": None}

    def card(parent, title, **pk):
        outer = tk.Frame(parent, bg=LINE, padx=1, pady=1); outer.pack(fill="x", pady=(0, 10), **pk)
        f = ttk.Frame(outer, style="Card.TFrame", padding=10); f.pack(fill="both", expand=True)
        if title: ttk.Label(f, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 6))
        return f

    # ---------------- header band
    top = tk.Frame(root, bg=NAVY, padx=18, pady=8); top.pack(side="top", fill="x")
    tk.Label(top, text="CytoMorpheus Analyzer", bg=NAVY, fg="white", font=("Segoe UI", 17, "bold")).pack(side="left")
    tk.Label(top, text="label-free classification of cell death from phase-contrast and dark-field video", bg=NAVY, fg=LIGHTBLUE, font=("Segoe UI", 10)).pack(side="left", padx=18, pady=(5, 0))
    b_set = tk.Button(top, text="⚙  Settings", bg=NAVY, fg="white", activebackground=BLUE, activeforeground="white", bd=0, font=("Segoe UI", 10), cursor="hand2"); b_set.pack(side="right", padx=(14, 0))
    logo_file = get_logo(APP_DIR)
    if logo_file:
        try:
            from PIL import Image, ImageTk
            im = Image.open(logo_file).convert("RGBA"); im = im.resize((int(im.width * 50 / im.height), 50), Image.LANCZOS)
            state["logo"] = ImageTk.PhotoImage(im); tk.Label(top, image=state["logo"], bg=NAVY).pack(side="right", padx=(0, 10))
        except Exception as e: print("logo not shown:", e)

    # ---------------- left column
    left = ttk.Frame(root, padding=(12, 12, 6, 12)); left.pack(side="left", fill="y")
    cv = card(left, "Videos of one field of view")
    ttk.Label(cv, text="Add one or two videos (phase contrast and/or dark-field). The modality is detected automatically.", style="Sub.TLabel", wraplength=380).pack(anchor="w")
    vlist = tk.Listbox(cv, height=3, width=52, bg="white", fg=INK, bd=0, highlightthickness=1, highlightbackground=LINE, font=("Segoe UI", 10), selectbackground=LIGHTBLUE, selectforeground=INK, activestyle="none"); vlist.pack(fill="x", pady=6)
    vb = ttk.Frame(cv, style="Card.TFrame"); vb.pack(fill="x")
    def refresh_videos():
        vlist.delete(0, "end")
        for v in state["videos"]: vlist.insert("end", f"  {v['mod_name']:14s}  {v['n']} frames   {pathlib.Path(v['path']).name}")
        if state["videos"] and not v_out.get(): v_out.set(str(pathlib.Path(state["videos"][0]["path"]).parent / f"CytoMorpheus_{pathlib.Path(state['videos'][0]['path']).stem}"))
    def add_videos():
        for p in filedialog.askopenfilenames(title="Choose video(s)", filetypes=[("video", "*.avi *.mp4 *.mov *.mkv *.tif *.tiff"), ("all", "*.*")]):
            if any(v["path"] == p for v in state["videos"]): continue
            mod, n = peek_modality(p)
            if mod is None: messagebox.showerror("CytoMorpheus", f"Cannot read {p}"); continue
            if len(state["videos"]) >= 2: messagebox.showwarning("CytoMorpheus", "At most two videos: one phase-contrast and one dark-field of the same field."); break
            if any(v["mod"] == mod for v in state["videos"]): messagebox.showwarning("CytoMorpheus", f"You already added a {MODN[mod].lower()} video. The second one must be the other modality of the same field."); continue
            state["videos"].append(dict(path=p, mod=mod, mod_name=MODN[mod], n=n))
        refresh_videos()
    def remove_video():
        for i in reversed(vlist.curselection()): state["videos"].pop(i)
        refresh_videos()
    ttk.Button(vb, text="＋ Add video(s)…", command=add_videos).pack(side="left"); ttk.Button(vb, text="Remove", command=remove_video).pack(side="left", padx=6)
    ttk.Button(vb, text="Clear", command=lambda: (state["videos"].clear(), v_out.set(""), refresh_videos())).pack(side="left")
    co = card(left, "Output folder"); v_out = tk.StringVar(); of = ttk.Frame(co, style="Card.TFrame"); of.pack(fill="x")
    ttk.Entry(of, textvariable=v_out, width=48).pack(side="left", fill="x", expand=True); ttk.Button(of, text="…", width=3, command=lambda: v_out.set(filedialog.askdirectory() or v_out.get())).pack(side="left", padx=(4, 0))
    ttk.Label(co, text="filled in automatically next to the first video; change it if you like", style="Sub.TLabel").pack(anchor="w", pady=(4, 0))
    ca = card(left, None); b_run = ttk.Button(ca, text="▶   Run analysis", style="Accent.TButton"); b_run.pack(fill="x")
    rb = ttk.Frame(ca, style="Card.TFrame"); rb.pack(fill="x", pady=(6, 0)); b_cancel = ttk.Button(rb, text="Cancel", state="disabled"); b_cancel.pack(side="left"); b_open = ttk.Button(rb, text="Open previous results…"); b_open.pack(side="right")
    cp = card(left, "Progress"); v_stage = tk.StringVar(value="ready"); ttk.Label(cp, textvariable=v_stage, style="Stage.TLabel", wraplength=380).pack(anchor="w")
    pb = ttk.Progressbar(cp, length=380, mode="determinate"); pb.pack(fill="x", pady=(6, 0))
    cl = card(left, "Log"); txt = scrolledtext.ScrolledText(cl, height=12, width=52, font=("Consolas", 9), bg="#f6f8fb", fg=INK, relief="flat"); txt.pack(fill="both", expand=True)

    # ---------------- right: notebook
    right = ttk.Frame(root, padding=(6, 12, 12, 12)); right.pack(side="right", fill="both", expand=True)
    nb = ttk.Notebook(right); nb.pack(fill="both", expand=True)
    tab_field, tab_cells, tab_sum = (ttk.Frame(nb, style="Card.TFrame", padding=8) for _ in range(3))
    nb.add(tab_field, text="  Field view  "); nb.add(tab_cells, text="  Cells  "); nb.add(tab_sum, text="  Summary  ")

    # ================= field view = player
    v_frame = tk.IntVar(value=0); v_show = {c: tk.BooleanVar(value=True) for c in CLS + ["Unclassified"]}; v_img = tk.StringVar(value="phase"); v_follow = tk.BooleanVar(value=True); v_speed = tk.StringVar(value="normal")
    ctl2 = ttk.Frame(tab_field, style="Card.TFrame"); ctl2.pack(fill="x", pady=(0, 4))
    ttk.Label(ctl2, text="Show:", style="Card.TLabel").pack(side="left")
    for c in CLS + ["Unclassified"]:
        tk.Checkbutton(ctl2, text=c, variable=v_show[c], fg=CLS_HEX[c], bg=CARD, activebackground=CARD, selectcolor="white", font=("Segoe UI", 10, "bold")).pack(side="left", padx=6)
    img_sel = ttk.Frame(ctl2, style="Card.TFrame"); img_sel.pack(side="left", padx=(18, 0))
    r_phase = ttk.Radiobutton(img_sel, text="phase contrast", value="phase", variable=v_img, style="Card.TRadiobutton"); r_dark = ttk.Radiobutton(img_sel, text="dark-field", value="dark", variable=v_img, style="Card.TRadiobutton")
    ttk.Button(ctl2, text="Save view as PNG", command=lambda: save_view()).pack(side="right"); ttk.Button(ctl2, text="Fit", width=5, command=lambda: fit_view()).pack(side="right", padx=6)
    body = ttk.Frame(tab_field, style="Card.TFrame"); body.pack(fill="both", expand=True)
    side = ttk.Frame(body, style="Card.TFrame", padding=(10, 0, 0, 0), width=300); side.pack(side="right", fill="y"); side.pack_propagate(False)
    viewer = ttk.Frame(body, style="Card.TFrame"); viewer.pack(side="left", fill="both", expand=True)
    tr = ttk.Frame(viewer, style="Card.TFrame"); tr.pack(side="bottom", fill="x", pady=(2, 0))
    tr0 = ttk.Frame(viewer, style="Card.TFrame"); tr0.pack(side="bottom", fill="x", pady=(4, 0))
    sl = ttk.Scale(tr0, from_=0, to=0, orient="horizontal", command=lambda v: goto(int(float(v)), from_slider=True)); sl.pack(side="left", fill="x", expand=True, padx=4)
    b_first = ttk.Button(tr, text="⏮", width=3, command=lambda: goto(0)); b_prev = ttk.Button(tr, text="◀", width=3, command=lambda: goto(v_frame.get() - 1))
    b_play = ttk.Button(tr, text="▶  Play", style="Play.TButton", command=lambda: toggle_play()); b_next = ttk.Button(tr, text="▶", width=3, command=lambda: goto(v_frame.get() + 1)); b_last = ttk.Button(tr, text="⏭", width=3, command=lambda: goto(10 ** 9))
    for b in [b_first, b_prev, b_play, b_next, b_last]: b.pack(side="left", padx=2)
    ttk.OptionMenu(tr, v_speed, "normal", "slow", "normal", "fast").pack(side="left", padx=(8, 2))
    v_ftxt = tk.StringVar(value=""); ttk.Label(tr, textvariable=v_ftxt, style="Card.TLabel").pack(side="left", padx=(10, 0))
    fig_f = Figure(figsize=(9, 6.2), dpi=100, facecolor=CARD); ax_f = fig_f.add_subplot(111); ax_f.set_axis_off(); fig_f.subplots_adjust(0, 0, 1, 1)
    can_f = FigureCanvasTkAgg(fig_f, master=viewer); can_f.get_tk_widget().pack(side="top", fill="both", expand=True); im_handle = {"im": None}; drag = {"p": None, "moved": False, "lim": None}
    # side panel: selected cell
    ttk.Label(side, text="Selected cell", style="CardTitle.TLabel").pack(anchor="w")
    v_cell = tk.StringVar(value="click a cell in the field"); ttk.Label(side, textvariable=v_cell, style="Sub.TLabel", wraplength=280, justify="left").pack(anchor="w")
    v_res = tk.StringVar(value=""); lbl_res = tk.Label(side, textvariable=v_res, bg=CARD, fg=NAVY, font=("Segoe UI", 15, "bold")); lbl_res.pack(anchor="w", pady=(4, 2))
    fig_p = Figure(figsize=(2.9, 2.0), dpi=100, facecolor=CARD); can_p = FigureCanvasTkAgg(fig_p, master=side); can_p.get_tk_widget().pack(anchor="w")
    v_when = tk.StringVar(value=""); ttk.Label(side, textvariable=v_when, style="Sub.TLabel", wraplength=280, justify="left").pack(anchor="w", pady=(4, 6))
    sbtn = ttk.Frame(side, style="Card.TFrame"); sbtn.pack(fill="x")
    ttk.Button(sbtn, text="▶ Play this cell", style="Play.TButton", command=lambda: play_cell()).pack(side="left"); ttk.Button(sbtn, text="Deselect", command=lambda: deselect()).pack(side="left", padx=6)
    ttk.Checkbutton(side, text="follow the cell while playing", variable=v_follow).pack(anchor="w", pady=(8, 0))
    ttk.Label(side, text="Legend", style="CardTitle.TLabel").pack(anchor="w", pady=(14, 2))
    for c, note in [("Apoptosis", "programmed death"), ("Alive", "no death event, control-like"), ("Necrosis", "rupture / lysis"), ("Unclassified", "tracked < 15 frames")]:
        row_ = ttk.Frame(side, style="Card.TFrame"); row_.pack(anchor="w"); tk.Label(row_, text="●", fg=CLS_HEX[c], bg=CARD, font=("Segoe UI", 12)).pack(side="left"); ttk.Label(row_, text=f"{c} — {note}", style="Card.TLabel").pack(side="left", padx=4)
    row_ = ttk.Frame(side, style="Card.TFrame"); row_.pack(anchor="w"); tk.Label(row_, text="●", fg="#ffd900", bg=CARD, font=("Segoe UI", 12)).pack(side="left"); ttk.Label(row_, text="selected cell", style="Card.TLabel").pack(side="left", padx=4)
    ttk.Label(side, text="mouse wheel = zoom · drag = pan · click a cell to select it\nspace = play / pause · ← → = step one frame", style="Sub.TLabel", justify="left", wraplength=280).pack(anchor="w", pady=(14, 0))

    def stack_for_view(res): return res["phase"] if (v_img.get() == "phase" and res["phase"] is not None) else (res["dark"] if res["dark"] is not None else res["phase"])
    def draw_field(*_):
        res = state["res"]
        if res is None: return
        t = int(v_frame.get()); show = {c for c, v in v_show.items() if v.get()}; stk = stack_for_view(res)
        key = ("lohi", v_img.get())
        if state.get("lohi_key") != key: state["lohi"] = tuple(np.percentile(stk[0], [.5, 99.5])); state["lohi_key"] = key
        img = render_frame(stk, res["labels"], res["df"], t, show=show, selected=state["sel"], lo=state["lohi"][0], hi=state["lohi"][1], lut=state["lut"])
        if im_handle["im"] is None: ax_f.clear(); ax_f.set_axis_off(); im_handle["im"] = ax_f.imshow(img); ax_f.set_xlim(0, img.shape[1]); ax_f.set_ylim(img.shape[0], 0)
        else: im_handle["im"].set_data(img)
        if state["sel"] is not None and v_follow.get() and play["on"]: centre_on(state["sel"], t, keep_zoom=True)
        v_ftxt.set(f"{t * DT_MIN} min   (frame {t}/{len(res['labels']) - 1})"); can_f.draw_idle()
        if int(float(sl.get())) != t: sl.set(t)
    v_frame.trace_add("write", draw_field); v_img.trace_add("write", draw_field)
    for v in v_show.values(): v.trace_add("write", draw_field)
    def goto(i, from_slider=False):
        res = state["res"]
        if res is None: return
        i = int(np.clip(i, 0, len(res["labels"]) - 1))
        if from_slider and i == int(v_frame.get()): return
        v_frame.set(i)
    def toggle_play():
        play["on"] = not play["on"]; b_play.configure(text="⏸  Pause" if play["on"] else "▶  Play")
        if play["on"]: tick()
        elif play["job"]: root.after_cancel(play["job"]); play["job"] = None
    def tick():
        res = state["res"]
        if not play["on"] or res is None: return
        t = int(v_frame.get()) + 1; lo_, hi_ = 0, len(res["labels"]) - 1
        if state["sel"] is not None and v_follow.get() and state["sel"] in res["cents"]:
            fr = [c[0] for c in res["cents"][state["sel"]]]; lo_, hi_ = fr[0], fr[-1]
        if t > hi_: t = lo_
        goto(t); play["job"] = root.after({"slow": 500, "normal": 220, "fast": 90}[v_speed.get()], tick)
    def centre_on(tid, t, keep_zoom=False, half=170):
        res = state["res"]; cent = res["cents"].get(tid)
        if not cent: return
        fr = [c[0] for c in cent]; j = int(np.argmin(np.abs(np.array(fr) - t))); _, cy, cx = cent[j]
        H, W = res["labels"].shape[1:]
        if keep_zoom: x0, x1 = ax_f.get_xlim(); y0, y1 = ax_f.get_ylim(); hw, hh = (x1 - x0) / 2, (y0 - y1) / 2
        else: hw = hh = half
        hw, hh = min(hw, W / 2), min(hh, H / 2); cx = float(np.clip(cx, hw, W - hw)); cy = float(np.clip(cy, hh, H - hh))   # keep the window inside the field
        ax_f.set_xlim(cx - hw, cx + hw); ax_f.set_ylim(cy + hh, cy - hh)
    def play_cell():
        res = state["res"]
        if res is None or state["sel"] is None: return
        fr = [c[0] for c in res["cents"][state["sel"]]]
        if play["on"]: toggle_play()
        v_follow.set(True); centre_on(state["sel"], fr[0]); goto(fr[0]); toggle_play()
    def deselect():
        state["sel"] = None; v_cell.set("click a cell in the field"); v_res.set(""); v_when.set(""); fig_p.clear(); can_p.draw_idle(); draw_field()
    def fit_view():
        res = state["res"]
        if res is None: return
        H, W = res["labels"].shape[1:]; ax_f.set_xlim(0, W); ax_f.set_ylim(H, 0); can_f.draw_idle()
    def on_scroll(ev):
        if state["res"] is None or ev.inaxes != ax_f or ev.xdata is None: return
        f = 1 / 1.25 if ev.button == "up" else 1.25; x0, x1 = ax_f.get_xlim(); y0, y1 = ax_f.get_ylim(); H, W = state["res"]["labels"].shape[1:]
        nx0 = ev.xdata - (ev.xdata - x0) * f; nx1 = ev.xdata + (x1 - ev.xdata) * f; ny0 = ev.ydata - (ev.ydata - y0) * f; ny1 = ev.ydata + (y1 - ev.ydata) * f
        if nx1 - nx0 > W: nx0, nx1, ny0, ny1 = 0, W, H, 0
        ax_f.set_xlim(nx0, nx1); ax_f.set_ylim(ny0, ny1); can_f.draw_idle()
    def on_press(ev):
        if ev.inaxes != ax_f or ev.xdata is None: return
        drag.update(p=(ev.x, ev.y), moved=False, lim=(ax_f.get_xlim(), ax_f.get_ylim(), ev.xdata, ev.ydata))
    def on_motion(ev):
        if drag["p"] is None or ev.inaxes != ax_f or ev.xdata is None: return
        if abs(ev.x - drag["p"][0]) + abs(ev.y - drag["p"][1]) > 4: drag["moved"] = True
        if drag["moved"]:
            (x0, x1), (y0, y1), px, py = drag["lim"]; dx, dy = ev.xdata - px, ev.ydata - py; ax_f.set_xlim(x0 - dx, x1 - dx); ax_f.set_ylim(y0 - dy, y1 - dy); can_f.draw_idle()
    def on_release(ev):
        moved = drag["moved"]; drag["p"] = None
        if moved or state["res"] is None or ev.inaxes != ax_f or ev.xdata is None: return
        t = int(v_frame.get()); y, x = int(ev.ydata), int(ev.xdata); lab = state["res"]["labels"][t]
        if 0 <= y < lab.shape[0] and 0 <= x < lab.shape[1] and lab[y, x] > 0: select_cell(int(lab[y, x]), from_field=True)
    can_f.mpl_connect("scroll_event", on_scroll); can_f.mpl_connect("button_press_event", on_press); can_f.mpl_connect("motion_notify_event", on_motion); can_f.mpl_connect("button_release_event", on_release)
    root.bind("<Left>", lambda e: goto(v_frame.get() - 1)); root.bind("<Right>", lambda e: goto(v_frame.get() + 1)); root.bind("<space>", lambda e: toggle_play() if state["res"] is not None and not isinstance(e.widget, (tk.Entry, ttk.Entry, scrolledtext.ScrolledText)) else None)
    def save_view():
        res = state["res"]
        if res is None: return
        p = filedialog.asksaveasfilename(defaultextension=".png", initialdir=str(res["out_dir"]), initialfile=f"field_{int(v_frame.get()) * DT_MIN}min.png")
        if p: fig_f.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); log(f"saved {p}")
    def show_prediction(tid):
        res = state["res"]; r = res["df"][res["df"]["track"] == tid]; fig_p.clear()
        if r.empty: return
        r = r.iloc[0]; cls_ = r["class"]; has_p, has_d = res["phase"] is not None, res["dark"] is not None; fr = [c[0] for c in res["cents"][tid]]
        v_cell.set(f"Cell {tid}  ·  tracked {len(fr)} frames ({fr[0] * DT_MIN}–{fr[-1] * DT_MIN} min)"); v_res.set(cls_ if cls_ != "Unclassified" else "not classified"); lbl_res.configure(fg=CLS_HEX[cls_])
        srcs = ([("phase", "Phase")] if has_p else []) + ([("dark", "Dark-field")] if has_d else []) + ([("fused", "Both")] if has_p and has_d else [])
        if r["window_start"] >= 0 and srcs:
            axp = fig_p.add_subplot(111); axp.set_facecolor(CARD); ys = np.arange(3)[::-1]; h = .8 / len(srcs)
            for j, (key, lab_) in enumerate(srcs):
                vals = [float(r.get(f"p_{key}_{c}", 0)) for c in CLS]; bars = axp.barh(ys + (j - (len(srcs) - 1) / 2) * h, vals, h * .9, color=PROB_COL[{"Phase": "Phase contrast", "Dark-field": "Dark-field", "Both": "Combined"}[lab_]], edgecolor="white", label=lab_)
                for b_, v in zip(bars, vals): axp.text(min(v + .02, .98), b_.get_y() + b_.get_height() / 2, f"{v:.2f}", va="center", fontsize=7, color=INK)
            axp.set_yticks(ys); axp.set_yticklabels(CLS, fontsize=8); axp.set_xlim(0, 1.18); axp.set_xticks([])
            for lab_, tick in zip(CLS, axp.get_yticklabels()): tick.set_color(CLS_HEX[lab_]); tick.set_fontweight("bold")
            if len(srcs) > 1: axp.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-.45, -.32), ncol=3, handlelength=1.2, columnspacing=1.0)
            for s_ in ["top", "right", "bottom"]: axp.spines[s_].set_visible(False)
            fig_p.subplots_adjust(left=.36, right=.98, top=.98, bottom=.26 if len(srcs) > 1 else .05)
            v_when.set(f"decision made on the {LW} frames {int(r['window_start']) * DT_MIN}–{int(r['window_end']) * DT_MIN} min" + (f"; death event detected at {int(r['anchor_frame']) * DT_MIN} min" if r["status"] == "event" else "; no clear death event in this track"))
        else: v_when.set(f"tracked for only {len(fr)} frames; at least {LW} are needed")
        can_p.draw_idle()
    def select_cell(tid, from_field=False):
        res = state["res"]
        if res is None or tid not in res.get("cents", {}): return
        state["sel"] = tid; show_prediction(tid)
        if not from_field:
            fr = [c[0] for c in res["cents"][tid]]; t = int(v_frame.get())
            if not (fr[0] <= t <= fr[-1]): goto(fr[0])
            centre_on(tid, int(v_frame.get())); nb.select(tab_field)
        draw_field()
        try: tv.selection_set(str(tid)); tv.see(str(tid))
        except Exception: pass

    # ================= cells tab = table
    fl = ttk.Frame(tab_cells, style="Card.TFrame"); fl.pack(fill="x", pady=(0, 4)); v_filter = tk.StringVar(value="All")
    ttk.Label(fl, text="Show", style="Card.TLabel").pack(side="left", padx=(0, 6))
    for c in ["All"] + CLS + ["Unclassified"]: ttk.Radiobutton(fl, text=c, value=c, variable=v_filter, style="Card.TRadiobutton", command=lambda: fill_table()).pack(side="left", padx=4)
    ttk.Label(fl, text="double-click a row to see the cell in the field view", style="Sub.TLabel").pack(side="left", padx=16)
    ttk.Button(fl, text="Export table (CSV)", command=lambda: export_table()).pack(side="right")
    tf = ttk.Frame(tab_cells, style="Card.TFrame"); tf.pack(fill="both", expand=True)
    cols = ("track", "class", "class_phase", "class_dark", "n_frames", "span"); heads = {"track": "Cell", "class": "Result", "class_phase": "Phase contrast says", "class_dark": "Dark-field says", "n_frames": "Frames tracked", "span": "Time span (min)"}
    tv = ttk.Treeview(tf, columns=cols, show="headings", selectmode="browse"); vsb = ttk.Scrollbar(tf, orient="vertical", command=tv.yview); tv.configure(yscrollcommand=vsb.set)
    for c in cols: tv.heading(c, text=heads[c], command=lambda c=c: sort_tv(c)); tv.column(c, width=170 if c in ("class_phase", "class_dark") else 120, anchor="center")
    tv.pack(side="left", fill="both", expand=True); vsb.pack(side="right", fill="y")
    for c in CLS + ["Unclassified"]: tv.tag_configure(c, foreground=CLS_HEX[c])
    sort_state = {"col": "track", "rev": False}
    def fill_table():
        res = state["res"]; tv.delete(*tv.get_children())
        if res is None: return
        d = res["df"]; f = v_filter.get()
        if f != "All": d = d[d["class"] == f]
        col = sort_state["col"] if sort_state["col"] in d.columns else "track"; d = d.sort_values(col, ascending=not sort_state["rev"], na_position="last")
        for r in d.to_dict("records"):
            pp = r.get("class_phase", ""); pp = pp if isinstance(pp, str) else "—"; pdk = r.get("class_dark", ""); pdk = pdk if isinstance(pdk, str) else "—"
            tv.insert("", "end", iid=str(int(r["track"])), tags=(r["class"],), values=(int(r["track"]), r["class"], pp, pdk, int(r["n_frames"]), f"{int(r['first_frame']) * DT_MIN}–{int(r['last_frame']) * DT_MIN}"))
    def sort_tv(c): sort_state["rev"] = not sort_state["rev"] if sort_state["col"] == c else False; sort_state["col"] = c; fill_table()
    def export_table():
        res = state["res"]
        if res is None: return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialdir=str(res["out_dir"]), initialfile="cells_export.csv")
        if p: res["df"].to_csv(p, index=False); log(f"saved {p}")
    tv.bind("<Double-1>", lambda e: select_cell(int(tv.selection()[0])) if tv.selection() else None)
    tv.bind("<<TreeviewSelect>>", lambda e: (state.update(sel=int(tv.selection()[0])), show_prediction(int(tv.selection()[0]))) if tv.selection() and state["res"] is not None and int(tv.selection()[0]) in state["res"].get("cents", {}) else None)

    # ================= summary tab
    sb = ttk.Frame(tab_sum, style="Card.TFrame"); sb.pack(fill="x", pady=(0, 6))
    ttk.Button(sb, text="Save summary figure (PNG)", command=lambda: save_summary()).pack(side="left"); ttk.Button(sb, text="Open output folder", command=lambda: open_folder()).pack(side="left", padx=8)
    v_sumtxt = tk.StringVar(value=""); ttk.Label(tab_sum, textvariable=v_sumtxt, style="Card.TLabel", font=("Segoe UI", 10), justify="left", wraplength=980).pack(anchor="w", padx=6, pady=(0, 6))
    fig_s = Figure(figsize=(9, 5), dpi=100, facecolor=CARD); can_s = FigureCanvasTkAgg(fig_s, master=tab_sum); can_s.get_tk_widget().pack(fill="both", expand=True)
    def draw_summary():
        res = state["res"]; fig_s.clear()
        if res is None: return
        df, s = res["df"], res["summary"]; d = df[df["class"] != "Unclassified"]; cnt = d["class"].value_counts().reindex(CLS, fill_value=0); n = int(cnt.sum())
        both = "class_phase" in df.columns and "class_dark" in df.columns and d["class_phase"].notna().any() and d["class_dark"].notna().any()
        gs = fig_s.add_gridspec(1, 3 if both else 2, width_ratios=[1, 1.35, 1] if both else [1, 1.6], left=.06, right=.98, top=.9, bottom=.16, wspace=.45)
        ax1 = fig_s.add_subplot(gs[0]); ax1.set_facecolor(CARD)
        ax1.pie(cnt.values, colors=[CLS_RGB[c] for c in CLS], startangle=90, radius=.8, wedgeprops=dict(edgecolor="white", linewidth=1.5))
        ax1.set_title(f"{n} cells classified", fontsize=11, color=NAVY, fontweight="bold")
        for k, (c, v) in enumerate(cnt.items()):   # counts as a list under the pie, so small slices never overlap
            ax1.text(0, -1.02 - 0.17 * k, f"{c}   {int(v)}   ({100 * v / max(n, 1):.1f} %)", ha="center", va="top", fontsize=9, color=CLS_HEX[c], fontweight="bold", transform=ax1.transData)
        ax1.set_ylim(-1.7, 1.0)
        ax2 = fig_s.add_subplot(gs[1]); ax2.set_facecolor(CARD); H, W = res["labels"].shape[1:]
        for c in CLS:
            e = d[d["class"] == c]
            if len(e): ax2.scatter(e["cx"] * UM_PX, e["cy"] * UM_PX, s=14, color=CLS_RGB[c], alpha=.8, linewidths=0)
        ax2.set_xlim(0, W * UM_PX); ax2.set_ylim(H * UM_PX, 0); ax2.set_aspect("equal"); ax2.set_xlabel("µm"); ax2.set_ylabel("µm"); ax2.set_title("Where the cells are", fontsize=11, color=NAVY, fontweight="bold")
        for s_ in ["top", "right"]: ax2.spines[s_].set_visible(False)
        if both:
            ax3 = fig_s.add_subplot(gs[2]); ax3.set_facecolor(CARD); xs = np.arange(3); w = .26
            for j, (col, lab_) in enumerate([("class_phase", "Phase contrast"), ("class_dark", "Dark-field"), ("class", "Combined")]):
                v = d[col].value_counts().reindex(CLS, fill_value=0).values; bars = ax3.bar(xs + (j - 1) * w, v, w * .92, color=PROB_COL[lab_], edgecolor="white")
                for b_, vv in zip(bars, v): ax3.text(b_.get_x() + b_.get_width() / 2, vv + max(n * .01, 1), str(int(vv)), ha="center", va="bottom", fontsize=7.5, color=INK)
            ax3.set_xticks(xs); ax3.set_xticklabels(CLS, fontsize=9); ax3.set_ylabel("cells"); ax3.set_title("Each modality on its own", fontsize=11, color=NAVY, fontweight="bold")
            for lab_, tick in zip(CLS, ax3.get_xticklabels()): tick.set_color(CLS_HEX[lab_]); tick.set_fontweight("bold")
            ax3.set_ylim(0, max(1, d[["class_phase", "class_dark", "class"]].apply(lambda c: c.value_counts().max()).max() * 1.15))
            for s_ in ["top", "right"]: ax3.spines[s_].set_visible(False)
            try: ax3.set_box_aspect(1.1)
            except Exception: pass
        from matplotlib.patches import Patch; from matplotlib.lines import Line2D
        hs = [Line2D([], [], marker="o", ls="", color=CLS_RGB[c], label=c) for c in CLS] + ([Patch(color=PROB_COL[k], label=k) for k in ["Phase contrast", "Dark-field", "Combined"]] if both else [])
        fig_s.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, fontsize=9, bbox_to_anchor=(.5, .01)); can_s.draw_idle()
        agree = ""
        if both: dd = d.dropna(subset=["class_phase", "class_dark"]); agree = f"   ·   phase contrast and dark-field agree on {(dd['class_phase'] == dd['class_dark']).mean() * 100:.1f} % of cells"
        vids = "  +  ".join(f"{k}: {pathlib.Path(v).name}" for k, v in s.get("videos", {}).items())
        v_sumtxt.set(f"{vids}\n{s['frames']} frames = {s.get('minutes', s['frames'] * DT_MIN)} min   ·   {s['tracks']} cells tracked, {s['classified']} classified, {s['tracks'] - s['classified']} too short   ·   decision: {s.get('decision', '')}{agree}\nanalysis time {s['seconds']} s   ·   results in {res['out_dir']}")
    def save_summary():
        res = state["res"]
        if res is None: return
        p = filedialog.asksaveasfilename(defaultextension=".png", initialdir=str(res["out_dir"]), initialfile="summary.png")
        if p: fig_s.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); log(f"saved {p}")
    def open_folder():
        res = state["res"]
        if res is None: return
        p = str(res["out_dir"])
        if sys.platform.startswith("win"): os.startfile(p)
        elif sys.platform == "darwin": subprocess.Popen(["open", p])
        else: subprocess.Popen(["xdg-open", p])

    # ================= settings
    def settings():
        w = tk.Toplevel(root); w.title("Settings"); w.configure(bg=PAPER); w.resizable(False, False); w.transient(root); w.grab_set()
        f = ttk.Frame(w, style="Card.TFrame", padding=14); f.pack(fill="both", expand=True, padx=10, pady=10)
        v_m = tk.StringVar(value=str(state["models"] or "")); v_s = tk.StringVar(value=" ".join(map(str, state["seeds"]))); v_v = tk.BooleanVar(value=state["write_video"])
        ttk.Label(f, text="Trained models folder", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w"); ttk.Entry(f, textvariable=v_m, width=60).grid(row=1, column=0, sticky="w"); ttk.Button(f, text="…", width=3, command=lambda: v_m.set(filedialog.askdirectory() or v_m.get())).grid(row=1, column=1, padx=4)
        ttk.Label(f, text="Model seeds to average (0 = fastest, 0 1 2 = as in the paper)", style="CardTitle.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 0)); ttk.Entry(f, textvariable=v_s, width=20).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(f, text="Write an annotated video", variable=v_v).grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Label(f, text=f"Window t−{PRE} … t+{POST} frames around the death event · {DT_MIN} min/frame · computing on {dev.upper()}", style="Sub.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 0))
        def ok():
            try: state["seeds"] = tuple(int(x) for x in v_s.get().split())
            except ValueError: messagebox.showerror("Settings", "Seeds must be integers, e.g. 0 1 2"); return
            state["models"] = pathlib.Path(v_m.get()) if v_m.get() else None; state["write_video"] = v_v.get(); w.destroy()
        ttk.Button(f, text="OK", command=ok, style="Accent.TButton").grid(row=6, column=0, sticky="e", pady=(14, 0))
    b_set.configure(command=settings)

    # ================= wiring
    def log(s): state["q"].put(("log", s))
    def progress(stage, i, n): state["q"].put(("prog", (stage, i, n)))
    def pump():
        try:
            while True:
                kind, val = state["q"].get_nowait()
                if kind == "log": txt.insert("end", val + "\n"); txt.see("end")
                elif kind == "prog": v_stage.set(f"{val[0]}  ·  {val[1]}/{val[2]}"); pb["maximum"] = val[2]; pb["value"] = val[1]
                elif kind == "done": show_results(val)
                elif kind == "error": messagebox.showerror("CytoMorpheus", val); v_stage.set("failed — see log"); finish()
                elif kind == "cancelled": v_stage.set("cancelled"); finish()
        except queue.Empty: pass
        root.after(100, pump)
    def finish(): b_run["state"] = "normal"; b_cancel["state"] = "disabled"; b_open["state"] = "normal"
    def show_results(res):
        if play["on"]: toggle_play()
        state["res"] = res; state["sel"] = None; im_handle["im"] = None; state["lut"] = class_lut(res["df"], int(res["labels"].max())); state["lohi_key"] = None
        r_phase.pack_forget(); r_dark.pack_forget()
        if res["phase"] is not None and res["dark"] is not None: r_phase.pack(side="left"); r_dark.pack(side="left", padx=6)
        v_img.set("phase" if res["phase"] is not None else "dark")
        sl.configure(to=len(res["labels"]) - 1); v_frame.set(0); deselect(); fill_table(); draw_summary(); finish(); nb.select(tab_field)
        c = res["summary"]["counts"]; v_stage.set("done  ·  " + ", ".join(f"{k} {v}" for k, v in c.items())); pb["value"] = pb["maximum"]
    def run():
        if not state["videos"]: messagebox.showerror("CytoMorpheus", "Add at least one video first."); return
        if state["models"] is None: messagebox.showerror("CytoMorpheus", "Trained models were not found next to the program. Set the models folder in Settings."); return
        b_run["state"] = "disabled"; b_cancel["state"] = "normal"; b_open["state"] = "disabled"; state["cancel"] = False; txt.delete("1.0", "end"); pb["value"] = 0
        vids = [v["path"] for v in state["videos"]]; out = v_out.get() or None
        def work():
            try: state["q"].put(("done", analyze_full(vids, out, state["models"], state["seeds"], log, progress, lambda: state["cancel"], state["write_video"])))
            except Cancelled: state["q"].put(("cancelled", None))
            except Exception as e:
                import traceback; log(traceback.format_exc()); state["q"].put(("error", str(e)))
        threading.Thread(target=work, daemon=True).start()
    def cancel(): state["cancel"] = True; v_stage.set("cancelling after the current step…")
    def open_prev(p=None):
        p = p or filedialog.askdirectory(title="Choose a results folder (contains summary.json)")
        if not p: return
        if not (pathlib.Path(p) / "summary.json").exists(): messagebox.showerror("CytoMorpheus", "No summary.json in that folder."); return
        b_run["state"] = "disabled"; b_open["state"] = "disabled"; v_stage.set("loading previous results…")
        def work():
            try: state["q"].put(("done", load_results(p, log)))
            except Exception as e:
                import traceback; log(traceback.format_exc()); state["q"].put(("error", str(e)))
        threading.Thread(target=work, daemon=True).start()
    b_run.configure(command=run); b_cancel.configure(command=cancel); b_open.configure(command=lambda: open_prev())
    if open_dir: root.after(200, lambda: open_prev(open_dir))
    if os.environ.get("CYTO_SHOTS"):                                        # headless self-test
        def _shots():
            if state["res"] is None: root.after(500, _shots); return
            shots = os.environ["CYTO_SHOTS"]
            for i, name in enumerate(["field", "cells", "summary"]):
                nb.select(i); root.update(); time.sleep(.6); subprocess.run(["import", "-window", "root", os.path.join(shots, f"tab_{name}.png")])
            select_cell(int(state["res"]["df"]["track"].iloc[0])); goto(12); root.update(); time.sleep(.6); subprocess.run(["import", "-window", "root", os.path.join(shots, "tab_field_selected.png")])
            root.destroy()
        root.after(1000, _shots)
    root.after(100, pump); root.mainloop()

if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="*", help="one or two videos of the same field"); ap.add_argument("--phase"); ap.add_argument("--dark"); ap.add_argument("--out")
    ap.add_argument("--models"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--no-video", action="store_true"); ap.add_argument("--open", help="show a finished analysis folder"); a = ap.parse_args()
    vids = list(a.videos) + [v for v in [a.phase, a.dark] if v]
    if vids: analyze(vids, a.out, a.models, tuple(a.seeds), write_video=not a.no_video)
    else: gui(a.open)
