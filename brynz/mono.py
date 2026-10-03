"""Metric monocular depth (MoGe-2, Microsoft, MIT licence) - the scale source of the RGB tiers.

Measured against iPhone LiDAR on 40 level frames of three captures (tools/depth_scale_eval.py):
with the true field of view MoGe-2 reads 0.955x LiDAR, the same on every capture (0.947-0.956),
per-frame spread +-7 %. MapAnything's own metric depth read 0.76x with +-20 %. So the RGB tiers
take scale from MoGe-2 (divided by the measured 0.955) and averaged over many frames.
"""
import os
import sys
from pathlib import Path

import warnings

import numpy as np

# MoGe asks for an fp32 autocast region; PyTorch's MPS backend notes it only autocasts to fp16/bf16 and
# simply runs it in fp32 - harmless, silenced so the run log stays readable
warnings.filterwarnings('ignore', message='.*MPS Autocast only supports.*')
warnings.filterwarnings('ignore', message='In MPS autocast.*')

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = str(ROOT / 'weights' / 'hf')
MOGE_ID = 'Ruicheng/moge-2-vitl-normal'
SCALE_CAL = 0.955        # MoGe-2 depth / LiDAR depth (median, 3 captures)
SCALE_CAL_SPREAD = 0.01  # spread of that median across captures (0.947-0.956)
NUM_TOKENS = 1200        # MoGe's low end of its suggested 1200-2500 (2x faster; input is only 392x518)


class MetricDepth:
    def __init__(self, device=None):
        import torch
        sys.path.insert(0, str(ROOT / 'third_party' / 'MoGe'))
        from huggingface_hub import snapshot_download
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        path = snapshot_download(MOGE_ID, cache_dir=HF_CACHE)
        self.model = MoGeModel.from_pretrained(os.path.join(path, 'model.pt')).to(self.device).eval()

    def infer(self, rgb, fov_x_deg=None):
        """rgb uint8 HxWx3 -> (metric depth HxW float32, 0 = invalid; horizontal FOV in degrees)."""
        x = self.torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).to(self.device)
        with self.torch.no_grad():
            o = self.model.infer(x, fov_x=fov_x_deg, use_fp16=False, num_tokens=NUM_TOKENS)
        d = o['depth'].float().cpu().numpy()
        d[~np.isfinite(d)] = 0
        K = o['intrinsics'].float().cpu().numpy()            # normalised intrinsics
        fov = float(np.degrees(2 * np.arctan(0.5 / K[0, 0])))
        return (d / SCALE_CAL).astype(np.float32), fov

    def free(self):
        del self.model
        if self.device == 'mps':
            self.torch.mps.empty_cache()
