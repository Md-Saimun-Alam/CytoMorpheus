"""Locked acquisition and preprocessing parameters.

Every value here was fixed before the dataset was built and none of it was
tuned afterwards.  The same numbers are given in the Experimental Section of the paper.
"""
import cv2
import numpy as np

# ----------------------------------------------------------------- imaging
Y0, Y1 = 35, 927        # valid rows; the timestamp occupies 2-34 and the scale bar 927-967
UM_PX = 0.933           # 10x objective, verified against the burned-in scale bar
DT_MIN = 3              # minutes between frames

# ------------------------------------------------------------ segmentation
SEG_MODEL = "cellpose-SAM"
FLOW = 0.4              # cellpose flow_threshold, phase contrast
CELLPROB = 1.0          # cellpose cellprob_threshold, phase contrast
PI_FLOW = 0.4           # cellpose on the fluorescence channel
PI_CELLPROB = 0.0

# ---------------------------------------------------------------- tracking
IOU_MIN = 0.10          # mutual-best overlap needed to link two masks
MAX_GAP = 2             # frames a track may be missing and still be linked
MAX_STEP_UM = 12.0      # displacement cap between linked frames
MAX_STEP_PX = MAX_STEP_UM / UM_PX

# ------------------------------------------------------- PI ground truth
PI_COVERAGE = 0.50      # a nucleus must cover this fraction of the mask
PI_DEBOUNCE = 2         # for this many consecutive frames
PI_PRE_CLEAN = 3        # and the cell must be PI-negative for its first 3 frames

# ------------------------------------------------------------------ window
BOX = 48                # crop side in pixels, 45 um
PRE, POST = 10, 4       # window is t-10 ... t+4 around the death frame
WINDOW = PRE + POST + 1  # 15 frames, 45 minutes
HALO = 2                # the mask channel is dilated by this many pixels

CLASSES = ["APOPTOSIS", "CONTROL", "NECROSIS"]


def read_frame(path, t):
    """One frame of a video, cropped to the valid imaging region."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"frame {t} of {path}")
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
    return g[Y0:Y1]


def read_video(path):
    """A whole video as (T, H, W) uint8, cropped to the valid imaging region."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        frames.append(g[Y0:Y1] if g.shape[0] >= Y1 else g)
    cap.release()
    if not frames:
        raise IOError(f"no frames read from {path}")
    return np.stack(frames)
