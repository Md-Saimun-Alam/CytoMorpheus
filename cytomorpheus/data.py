"""Dataset, input channels and the two evaluation protocols.

The dataset is one array of anchored windows, (N, 15, 3, 48, 48) uint8, where
the three stored planes are phase contrast, dark-field and the cell mask of the
tracked cell.  `build_batch` turns a batch of windows into the three network
channels described in the paper, the intensity normalised within the sequence,
the difference between consecutive frames, and the dilated mask.

Both modalities come from the same windows and the same masks, so a cell's
phase and dark-field sequences differ only in which stored plane is read.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

CLASSES = ["APOPTOSIS", "CONTROL", "NECROSIS"]
CROP = 48          # stored crop, 48 px = 45 um at 0.933 um/px
PRE, POST = 10, 4  # window is t-10 ... t+4 around the death frame
T_FRAMES = PRE + POST + 1

_BLUR = None       # 3x3 binomial kernel, built on first use


class Dataset:
    """Windows held on the GPU, with the metadata needed by the splits."""

    def __init__(self, data_dir, device="cuda"):
        import pathlib
        d = pathlib.Path(data_dir)
        meta = np.load(d / "meta.npz", allow_pickle=True)
        self.y = meta["y"].astype(int)
        self.rec = meta["rec"].astype(str)
        self.fold = meta["fold"].astype(str)
        self.uid = meta["uid"].astype(str)
        self.device = device
        self.X = torch.from_numpy(np.load(d / "cache.npy")).to(device)
        self.T = self.X.shape[1]

    def __len__(self):
        return len(self.y)

    def counts(self):
        return dict(zip(CLASSES, np.bincount(self.y, minlength=3).tolist()))


def _blur_kernel(device):
    global _BLUR
    if _BLUR is None or _BLUR.device != torch.device(device):
        k = torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]], device=device) / 16
        _BLUR = k.view(1, 1, 3, 3)
    return _BLUR


def build_batch(ds, idx, aug, modality, resolution, halo=2):
    """Windows -> network input (B, 3, T, R, R).

    modality  0 = phase contrast, 1 = dark-field
    aug       apply the training augmentation, one random transform per window
              applied identically to all of its frames
    """
    dev = ds.device
    xb = ds.X[idx]
    b, t = xb.shape[0], ds.T
    g = xb[:, :, modality].float() / 255.
    m = (xb[:, :, 2] > 0).float()
    m = F.max_pool2d(m.reshape(-1, 1, CROP, CROP), 2 * halo + 1, 1, halo).reshape(b, t, CROP, CROP)

    if aug:
        # 90-degree rotations, flips and shifts of up to 3 px, one draw per window
        k = torch.randint(0, 4, (b,), device=dev).float() * (math.pi / 2)
        ca, sa = torch.cos(k), torch.sin(k)
        fx = torch.where(torch.rand(b, device=dev) < .5, -1., 1.)
        fy = torch.where(torch.rand(b, device=dev) < .5, -1., 1.)
        th = torch.zeros(b, 2, 3, device=dev)
        th[:, 0, 0] = ca * fx; th[:, 0, 1] = -sa * fx
        th[:, 0, 2] = (torch.rand(b, device=dev) * 2 - 1) * (3 / 24)
        th[:, 1, 0] = sa * fy; th[:, 1, 1] = ca * fy
        th[:, 1, 2] = (torch.rand(b, device=dev) * 2 - 1) * (3 / 24)
        gr = F.affine_grid(th, (b, t, CROP, CROP), align_corners=False)
        g = F.grid_sample(g, gr, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = F.grid_sample(m, gr, mode="nearest", padding_mode="zeros", align_corners=False)
        # contrast and gamma
        bl = F.conv2d(g.reshape(b * t, 1, CROP, CROP), _blur_kernel(dev), padding=1)
        bl = bl.reshape(b, t, CROP, CROP)
        g = g + (torch.rand(b, 1, 1, 1, device=dev) - .5) * (g - bl)
        g = g.clamp(0, 1) ** (.75 + torch.rand(b, 1, 1, 1, device=dev) * .60)

    z = (g - g.mean((1, 2, 3), keepdim=True)) / (g.std((1, 2, 3), keepdim=True) + 1e-5)
    if aug:
        z = z + .02 * torch.randn_like(z)
    dz = z - torch.roll(z, 1, dims=1)
    dz[:, 0] = 0
    x = torch.stack([z, 3 * dz, 2 * m - 1], 1)
    if resolution != CROP:
        x = F.interpolate(x, size=(t, resolution, resolution), mode="trilinear",
                          align_corners=False)
    return x


# ------------------------------------------------------------------- splits
def split_heldout(ds, fold, seed):
    """Held-out-recording protocol.

    Every cell of the recordings in `fold` is the test set.  Of the remaining
    recordings, 15 % of the cells of each recording and class form the
    validation set, used only to select the checkpoint.
    """
    base = fold[:2]
    rng = np.random.default_rng(seed)
    te = np.where(ds.fold == base)[0]
    rest = np.where(ds.fold != base)[0]
    val = []
    for r in np.unique(ds.rec[rest]):
        for c in np.unique(ds.y[rest]):
            ii = rest[(ds.rec[rest] == r) & (ds.y[rest] == c)]
            if len(ii):
                val += list(rng.choice(ii, max(1, int(round(.15 * len(ii)))), replace=False))
    val = np.array(sorted(set(val)))
    return np.setdiff1d(rest, val), val, te


def split_within(ds, seed):
    """Within-experiment protocol.

    20 % of the cells of every recording and class are the test set, then 15 %
    of what remains is the validation set.  Every recording contributes to
    every split.
    """
    rng = np.random.default_rng(seed)
    tr, val, te = [], [], []
    for r in np.unique(ds.rec):
        for c in np.unique(ds.y):
            ii = rng.permutation(np.where((ds.rec == r) & (ds.y == c))[0])
            if len(ii) == 0:
                continue
            n_te = int(round(.20 * len(ii)))
            n_val = int(round(.15 * (len(ii) - n_te)))
            te += list(ii[:n_te])
            val += list(ii[n_te:n_te + n_val])
            tr += list(ii[n_te + n_val:])
    return np.array(sorted(tr)), np.array(sorted(val)), np.array(sorted(te))


def get_split(ds, protocol, fold, seed):
    if protocol == "WITHIN":
        return split_within(ds, seed)
    return split_heldout(ds, fold, seed)
