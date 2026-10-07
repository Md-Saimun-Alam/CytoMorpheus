"""The four spatiotemporal architectures used in the paper.

Every model takes a (B, 3, T, R, R) tensor and returns three logits
(apoptosis, control, necrosis).  T = 15 frames.  R is 48 for the 3D-CNN,
which is trained from scratch, and 96 for the three ImageNet backbones,
so that the input matches the scale of the pretrained filters.
"""
import torch
import torch.nn as nn
from torchvision.models import (
    alexnet, AlexNet_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

N_CLASSES = 3
T_FRAMES = 15


# --------------------------------------------------------------------- 3D-CNN
def _block(cin, cout, stride):
    """One factorised space-then-time block."""
    return nn.Sequential(
        nn.Conv3d(cin, cout, (1, 3, 3), (1, stride, stride), (0, 1, 1), bias=False),
        nn.BatchNorm3d(cout), nn.SiLU(True),
        nn.Conv3d(cout, cout, (3, 1, 1), 1, (1, 0, 0), bias=False),
        nn.BatchNorm3d(cout), nn.SiLU(True),
    )


class Net3D(nn.Module):
    """Five factorised blocks, 64 -> 512 filters, trained from scratch at 48 px."""

    def __init__(self, nc=N_CLASSES):
        super().__init__()
        self.f = nn.Sequential(
            _block(3, 64, 1), _block(64, 96, 2), _block(96, 192, 2),
            _block(192, 320, 2), _block(320, 512, 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.BatchNorm1d(512), nn.Dropout(0.3), nn.Linear(512, nc),
        )

    def forward(self, x):
        return self.head(self.f(x))


# ------------------------------------------------- shared per-frame backbones
def _per_frame(backbone, x):
    """Apply a 2D backbone to every frame with shared weights -> (B, T, D)."""
    b, c, t, h, w = x.shape
    return backbone(x.transpose(1, 2).reshape(b * t, c, h, w)).reshape(b, t, -1)


class BiLSTMPool(nn.Module):
    def __init__(self, d, hid=256):
        super().__init__()
        self.lstm = nn.LSTM(d, hid, batch_first=True, bidirectional=True)

    def forward(self, f):
        return self.lstm(f)[0].mean(1)


class TransformerPool(nn.Module):
    def __init__(self, d, dm=256, heads=4, layers=2, t=T_FRAMES):
        super().__init__()
        self.proj = nn.Linear(d, dm)
        self.pos = nn.Parameter(torch.zeros(1, t, dm))
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=0.1,
                                       batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dm)

    def forward(self, f):
        return self.norm(self.enc(self.proj(f) + self.pos)).mean(1)


class AlexNetBiLSTM(nn.Module):
    """AlexNet features per frame, read in order by a bidirectional LSTM."""

    def __init__(self, nc=N_CLASSES):
        super().__init__()
        m = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.back = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(BiLSTMPool(256), nn.Dropout(0.5), nn.Linear(512, nc))

    def forward(self, x):
        return self.head(_per_frame(self.back, x))


class MobileNetV2TP(nn.Module):
    """MobileNetV2 features per frame, averaged over time."""

    def __init__(self, nc=N_CLASSES):
        super().__init__()
        m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.back = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, nc))

    def forward(self, x):
        return self.head(_per_frame(self.back, x).mean(1))


class EffB0Transformer(nn.Module):
    """EfficientNet-B0 features per frame, related across time by a transformer."""

    def __init__(self, nc=N_CLASSES, t=T_FRAMES):
        super().__init__()
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.back = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(TransformerPool(1280, t=t), nn.Dropout(0.5),
                                  nn.Linear(256, nc))

    def forward(self, x):
        return self.head(_per_frame(self.back, x))


# ------------------------------------------------------------------ registry
# R    input resolution              BS  batch size
# lr_bb backbone learning rate       lr_hd head learning rate
# EP   epochs                        cl3d use channels_last_3d memory format
MODELS = {
    "3DCNN": dict(
        ctor=Net3D, R=48, BS=96, lr_bb=3e-4, lr_hd=3e-4, EP=50, cl3d=True),
    "AlexNet-BiLSTM": dict(
        ctor=AlexNetBiLSTM, R=96, BS=32, lr_bb=1e-4, lr_hd=1e-3, EP=30, cl3d=False),
    "MobileNetV2": dict(
        ctor=MobileNetV2TP, R=96, BS=32, lr_bb=1e-4, lr_hd=1e-3, EP=30, cl3d=False),
    "EfficientNetB0-Transformer": dict(
        ctor=EffB0Transformer, R=96, BS=32, lr_bb=1e-4, lr_hd=1e-3, EP=30, cl3d=False),
}

NAMES = list(MODELS)
