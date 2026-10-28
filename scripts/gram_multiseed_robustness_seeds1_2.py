#!/usr/bin/env python3
"""Multi-seed robustness runner derived from the validated master script.

This version intentionally preserves the original preprocessing, frozen encoder/decoder,
mapper architecture, composite loss, checkpoint selection, stopping criteria, metric
definitions, and mean-handling semantics so that seeds 1 and 2 are directly comparable
with the already completed seed-0 results.

It runs ONLY five representative narrowband configurations on BOTH Chikusei and ICVL:
    1) single_nir_801
    2) two_vis_461_549
    3) two_nir_801_935
    4) spanning_549_801
    5) spanning_461_549_652_801

New training workload: 5 configurations x 2 new seeds x 2 datasets = 20 runs.
The analysis stage then combines these with the existing seed-0 results to produce
three-seed mean +/- SD tables.

Original master description follows below.

Master runner: both datasets, new 31-band RGB products, all 16 representative inputs.

This is the consolidated version to keep for future reruns. It combines the
corrected RGB preparation with the representative narrowband experiments for
BOTH Chikusei and ICVL in one script.

Per dataset (16 runs):
    2 RGB-related
        - RGB
        - RGB + ~801 nm

    5 representative single bands
        - ~461, ~549, ~652, ~801, ~935 nm

    7 representative two-band configurations
        - 461+549       visible+visible control
        - 801+935       NIR+NIR control
        - 461+801       visible+NIR
        - 549+801       reference visible+NIR pair
        - 652+801       visible+NIR
        - 549+714       NIR-location test
        - 549+935       NIR-location test

    1 three-band configuration
        - 461+549+801

    1 four-band configuration
        - 461+549+652+801

Total if both datasets are enabled: 32 physical runs.

DATA PREPARATION
----------------
The existing 40-band VIS-NIR cubes are kept unchanged. The newly regenerated
31-band visible cubes are used for RGB synthesis.

Chikusei:
    40-band: Chikusei/Chikusei_1000_40bands.h5
    31-band: Chikusei/Chikusei_1000_visible31_new.h5

ICVL:
    40-band: ICVL_raw/ICVL_1020_1223_40band.mat
    31-band: ICVL_raw/ICVL_1020_1223_visible31_new.mat

For ICVL, BOTH the 40-band and new 31-band cubes are derived in the original
raw spatial orientation, so the SAME spatial transform is applied to both:

    rotate 90 degrees -> horizontal mirror -> crop [392:1192, 100:900]

The RGB/HSI visualization check confirmed that these products are aligned.

NORMALIZATION
-------------
For each dataset, the aligned/cropped 31-band and 40-band cubes are divided
ONCE by the same mutual maximum:

    common_max = max(max(full_40_raw), max(visible_31_raw))

There is no second per-cube normalization. RGB is generated from the normalized
31-band visible cube with the fixed 3x31 B/G/R SRF, and RGB+801 concatenates
that exact RGB with the ~801-nm channel from the same normalized 40-band cube.

TRAINING
--------
- seed = 0
- frozen pretrained 40-band encoder and decoder
- only the input-to-40-band mapper is optimized
- same Gram-matrix loss as the representative study
- original composite loss is used for optimization
- best checkpoint is selected by minimum composite loss every epoch
- centered abundance-Gram SAM is used as one unsupervised early-stopping signal
      meaningful decrease >= 0.001
      patience = 5000 epochs
- predicted-spectrum convergence is a second unsupervised early-stopping signal:
      compare the full predicted 40-band HSI across a 2000-epoch window
      prediction is placed on an absolute scale using the OBSERVED LR-HSI mean (not GT)
      stop after 2 consecutive 2000-epoch windows with relative L2 change < 1e-3
      (minimum epoch 5000; no ground truth is used for this criterion)
- reconstruction metrics and interactive visualization every 200 epochs
- restart-safe: completed runs are skipped and incomplete runs can resume
- NO reuse of results from older roots; this master script writes a clean study

The built-in analysis regenerates the representative mapping, location scans,
RGB comparison, per-wavelength error profiles, and 1-to-4-band curve.

Run directly:
    python fullBandStudy_bothDatasets_new31_all16_compositeLoss_GramSAM_reconWindowStop_consoleViz200_seed0.py

or in Spyder/IPython:
    %run -i fullBandStudy_bothDatasets_new31_all16_compositeLoss_GramSAM_reconWindowStop_consoleViz200_seed0.py

No argparse/main guard is used.
"""


from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib
# Interactive backend: figures are shown with plt.show(); they are not saved.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage, signal
from scipy.io import loadmat


# =============================================================================
# USER SETTINGS
# =============================================================================
RUN_TRAINING = True
RUN_ANALYSIS = True
RUN_CHIKUSEI = True
RUN_ICVL = True

DEVICE = "cuda:0"

# Seed 0 already exists in the original full study and is NOT retrained here.
# Keep SEED=0 only for compatibility with untouched helper/legacy-analysis code.
SEED = 0
SEEDS_TO_RUN = (1, 2)

ROBUSTNESS_CONFIGS = (
    "single_nir_801",
    "two_vis_461_549",
    "two_nir_801_935",
    "spanning_549_801",
    "spanning_461_549_652_801",
)

DETERMINISTIC = False

MAX_EPOCHS = 1_000_001
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 200
VIZ_EVERY = 2000
SAM_PATIENCE_EPOCHS = 5_000
SAM_MIN_DECREASE = 1e-3

# Predicted-spectrum convergence stopping. Compare the full predicted 40-band HSI
# across a LONG window, rather than against the immediately previous 200-epoch
# snapshot. The prediction is put on an absolute scale with the observed LR-HSI
# spectral mean, so this criterion remains fully unsupervised and is better aligned
# with whether the predicted spectra have actually stopped changing.
RECON_WINDOW_EPOCHS = 2_000
RECON_REL_CHANGE_THRESHOLD = 1e-3   # <0.1% full-cube change over 2000 epochs
RECON_STABLE_WINDOWS = 2            # require two consecutive stable windows
RECON_EARLY_STOP_MIN_EPOCH = 5_000

SAVE_LATEST_EVERY = 10_000
SCALE_FACTOR = 8
GRAD_CLIP = 0.0

# Fresh root for ONLY the new seed-1/seed-2 robustness runs.
OUTPUT_ROOT = Path("fullBandStudy_multiseed_robustness_seeds1_2")

# Existing authoritative seed-0 full study. It is read only during robustness analysis;
# nothing in this script writes to or retrains this root.
BASELINE_SEED0_ROOT = Path(
    "fullBandStudy_new31_both_all16_compositeLoss_GramSAMstop_seed0"
)

# If True, result.json marks a run as complete and it will not be retrained.
SKIP_COMPLETED = True
RESUME_INCOMPLETE = True

# Reuse physically identical runs from the earlier 822-run mutual-max script when
# possible. This is safe only when seed, normalization and source/band indices match.
REUSE_COMPATIBLE_ROOTS: Tuple[Path, ...] = ()  # clean master: never import legacy results

# Per-band profiles are needed only for analysis item 4.
PROFILE_EXPERIMENTS = (
    "rgb_only",
    "rgb_plus_nir_801",
    "single_nir_801",
    "two_nir_801_935",
    "spanning_549_801",
)

# Dataset paths and pretrained decomposition checkpoints.
CHIKUSEI_40_PATH = Path("Chikusei/Chikusei_1000_40bands.h5")
CHIKUSEI_31_PATH = Path("Chikusei/Chikusei_1000_visible31_new.h5")
CHIKUSEI_KEY = "chikusei"
CHIKUSEI_ENCODER = Path(
    "unsupervised_CRF_learning_encoder_uav_chikusei_CE_9_endmember_sam_4spc40_visNir_g54_01.pth"
)
CHIKUSEI_DECODER = Path(
    "unsupervised_CRF_learning_decoder_uav_chikusei_CE_9_endmember_sam_4spc40_visNir_g54_01.pth"
)

ICVL_40_PATH = Path("ICVL_raw/ICVL_1020_1223_40band.mat")
ICVL_31_PATH = Path("ICVL_raw/ICVL_1020_1223_visible31_new.mat")
ICVL_ENCODER = Path(
    "unsupervised_CRF_learning_encoder_uav_icvl_CE_9_endmember_sam_4spc40_visNir_g54_01.pth"
)
ICVL_DECODER = Path(
    "unsupervised_CRF_learning_decoder_uav_icvl_CE_9_endmember_sam_4spc40_visNir_g54_01.pth"
)

VISIBLE_MAX_NM = 700.0
DPI = 300


# =============================================================================
# WAVELENGTHS AND FIXED 31-SAMPLE RGB RESPONSE
# =============================================================================
CHIKUSEI_WAVELENGTHS_NM = np.array(
    [
        363.00, 378.47, 399.10, 414.57, 430.05, 445.52, 460.99,
        481.62, 497.09, 512.57, 533.20, 548.67,
        564.14, 579.61, 600.24, 615.72, 631.19, 651.82,
        667.29, 682.76, 698.24, 713.71, 734.34, 749.81, 765.28,
        780.76, 801.39, 816.86, 832.33, 847.80, 868.43, 883.91,
        899.38, 914.85, 935.48, 950.95, 966.43, 981.90, 1002.53,
        1018.00,
    ],
    dtype=np.float32,
)

# This is the fixed srf_np used in the supplied Chikusei/ICVL code.
# Rows are kept in the original B/G/R ordering.
RGB_SRF_BGR = np.array(
    [
        [
            0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026,
            0.024, 0.019, 0.010, 0.004, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        ],
        [
            0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003,
            0.005, 0.007, 0.012, 0.013, 0.015, 0.016, 0.017, 0.020,
            0.013, 0.011, 0.009, 0.005, 0.001, 0.001, 0.001, 0.001,
            0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003,
        ],
        [
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.001, 0.003, 0.010, 0.012, 0.013, 0.022, 0.020, 0.020,
            0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013,
        ],
    ],
    dtype=np.float64,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass(frozen=True)
class Experiment:
    name: str
    source: str  # narrowband | rgb | rgb_plus_narrowband
    band_indices: Tuple[int, ...]
    analysis_role: str

    @property
    def n_channels(self) -> int:
        if self.source == "rgb":
            return 3
        if self.source == "rgb_plus_narrowband":
            return 3 + len(self.band_indices)
        return len(self.band_indices)


@dataclass
class PreparedDataset:
    name: str
    wavelengths_nm: np.ndarray
    hsi_hwc: np.ndarray
    rgb_hwc: np.ndarray
    raw_full_max: float
    raw_visible_max: float
    common_max: float
    encoder_checkpoint: Path
    decoder_checkpoint: Path
    root: Path


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class EncoderLRHSI(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv11 = nn.Linear(40, 5, bias=False)
        self.conv12 = nn.Linear(45, 5, bias=False)
        self.conv13 = nn.Linear(50, 40, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_11 = self.relu(self.conv11(x))
        stack_layer_11 = torch.cat((x, layer_11), dim=1)
        layer_12 = self.relu(self.conv12(stack_layer_11))
        stack_layer_12 = torch.cat((stack_layer_11, layer_12), dim=1)
        return self.relu(self.conv13(stack_layer_12))


class DecoderHSI(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv31 = nn.Linear(40, 40, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv31(x)


class InputToHSIMapper(nn.Module):
    """Residual input-to-40-band mapper matching the supplied definitions.

    1 input  -> [40]
    2 inputs -> [20, 20]
    3 inputs -> [13, 13, 14]
    4 inputs -> [10, 10, 10, 10]
    """

    def __init__(self, n_inputs: int, n_outputs: int = 40) -> None:
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.conv11 = nn.Linear(self.n_inputs, self.n_outputs, bias=False)

        base = self.n_outputs // self.n_inputs
        counts = [base] * self.n_inputs
        counts[-1] += self.n_outputs - sum(counts)
        self.register_buffer("repeat_counts", torch.tensor(counts, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        learned = self.conv11(x)
        residual = torch.repeat_interleave(x, self.repeat_counts, dim=1)
        return learned + residual


# =============================================================================
# GENERIC DATA HELPERS
# =============================================================================
def reshape_cube(array: np.ndarray, bands: int, expected_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 2:
        if array.shape[-1] != bands:
            raise ValueError(f"Expected flattened data with {bands} bands, got {array.shape}")
        if expected_hw is not None:
            h, w = expected_hw
            if h * w != array.shape[0]:
                raise ValueError(f"Cannot reshape {array.shape} to {(h, w, bands)}")
            return array.reshape(h, w, bands)
        n = array.shape[0]
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError(f"Flattened cube is not square: {array.shape}")
        return array.reshape(side, side, bands)

    if array.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D cube, got {array.shape}")

    if array.shape[-1] == bands:
        cube = array
    elif array.shape[0] == bands:
        cube = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Could not identify {bands}-band spectral axis in {array.shape}")

    if expected_hw is not None and cube.shape[:2] != expected_hw:
        raise ValueError(f"Expected spatial size {expected_hw}, got {cube.shape[:2]}")
    return cube


def generate_msi(img: np.ndarray, spectral_matrix: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC cube, got {img.shape}")
    if spectral_matrix.shape[0] != img.shape[2]:
        raise ValueError(
            f"Spectral matrix {spectral_matrix.shape} incompatible with image {img.shape}"
        )
    h, w, c = img.shape
    out = np.dot(img.reshape(h * w, c), spectral_matrix)
    return out.reshape(h, w, spectral_matrix.shape[1]).astype(np.float32, copy=False)


def matlab_style_gaussian_2d(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    total = kernel.sum()
    if total != 0:
        kernel /= total
    return kernel


def generate_lr_hsi(img: np.ndarray, scale_factor: int) -> np.ndarray:
    kernel = matlab_style_gaussian_2d((scale_factor, scale_factor), sigma=0.5)
    h, w, c = img.shape
    out = np.zeros((h // scale_factor, w // scale_factor, c), dtype=np.float32)
    for band in range(c):
        filtered = signal.convolve2d(img[:, :, band], kernel, mode="valid")
        out[:, :, band] = filtered[::scale_factor, ::scale_factor]
    return out


def fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# =============================================================================
# EXACT DATASET PREPARATION WITH MUTUAL-MAX NORMALIZATION
# =============================================================================
def prepare_chikusei() -> PreparedDataset:
    if not CHIKUSEI_40_PATH.exists():
        raise FileNotFoundError(CHIKUSEI_40_PATH)
    if not CHIKUSEI_31_PATH.exists():
        raise FileNotFoundError(
            f"New Chikusei 31-band visible cube not found: {CHIKUSEI_31_PATH}"
        )

    # Existing 40-band product: do not alter its spectral preparation.
    with h5py.File(CHIKUSEI_40_PATH, "r") as f:
        full_raw = reshape_cube(np.asarray(f[CHIKUSEI_KEY]), 40, (1000, 1000))
        if "selected_wavelengths_nm" in f:
            wavelengths = np.asarray(
                f["selected_wavelengths_nm"], dtype=np.float32
            ).ravel()
        else:
            wavelengths = CHIKUSEI_WAVELENGTHS_NM.copy()

    # Newly rebuilt 31-band visible product from the original 128-band cube.
    with h5py.File(CHIKUSEI_31_PATH, "r") as f:
        visible_raw = reshape_cube(np.asarray(f[CHIKUSEI_KEY]), 31, (1000, 1000))

    if wavelengths.size != 40 or not np.all(np.diff(wavelengths) > 0):
        raise ValueError(f"Invalid Chikusei wavelength vector: {wavelengths.shape}")
    if full_raw.shape[:2] != visible_raw.shape[:2]:
        raise ValueError(
            f"Chikusei 40/31 spatial mismatch: {full_raw.shape} vs {visible_raw.shape}"
        )

    full_max = float(np.max(full_raw))
    visible_max = float(np.max(visible_raw))
    common_max = max(full_max, visible_max)
    if not np.isfinite(common_max) or common_max <= 0:
        raise ValueError(f"Invalid Chikusei mutual maximum: {common_max}")

    # ONE shared normalization only.
    full_norm = (full_raw / common_max).astype(np.float32, copy=False)
    visible_norm = (visible_raw / common_max).astype(np.float64, copy=False)
    rgb = generate_msi(visible_norm, RGB_SRF_BGR.T)

    i801 = int(np.argmin(np.abs(wavelengths - 801.39)))

    print("\n[Chikusei unified preparation]")
    print(f"  40-band source   : {CHIKUSEI_40_PATH}")
    print(f"  31-band source   : {CHIKUSEI_31_PATH}")
    print(f"  raw 40 max       : {full_max:.10g}")
    print(f"  raw 31 max       : {visible_max:.10g}")
    print(f"  raw max ratio 40/31: {full_max / visible_max:.8f}")
    print(f"  mutual max       : {common_max:.10g}")
    print(f"  normalized 40 max: {full_norm.max():.8f}")
    print(f"  normalized 31 max: {visible_norm.max():.8f}")
    print(f"  RGB shape/range  : {rgb.shape}, ({rgb.min():.8f}, {rgb.max():.8f})")
    print(f"  RGB sha256       : {fingerprint(rgb)}")
    print(f"  ~801 band        : index={i801}, wavelength={wavelengths[i801]:.4f} nm")

    root = OUTPUT_ROOT / "Chikusei"
    root.mkdir(parents=True, exist_ok=True)
    return PreparedDataset(
        name="Chikusei",
        wavelengths_nm=wavelengths,
        hsi_hwc=full_norm,
        rgb_hwc=rgb,
        raw_full_max=full_max,
        raw_visible_max=visible_max,
        common_max=common_max,
        encoder_checkpoint=CHIKUSEI_ENCODER,
        decoder_checkpoint=CHIKUSEI_DECODER,
        root=root,
    )


def prepare_icvl() -> PreparedDataset:
    if not ICVL_40_PATH.exists():
        raise FileNotFoundError(ICVL_40_PATH)
    if not ICVL_31_PATH.exists():
        raise FileNotFoundError(
            f"New ICVL 31-band visible cube not found: {ICVL_31_PATH}"
        )

    full_payload = loadmat(ICVL_40_PATH)
    if "rad" not in full_payload or "wavelengths" not in full_payload:
        raise KeyError(f"Expected 'rad' and 'wavelengths' in {ICVL_40_PATH}")
    full_original = np.asarray(full_payload["rad"], dtype=np.float32)
    wavelengths = np.asarray(full_payload["wavelengths"], dtype=np.float32).ravel()

    visible_payload = loadmat(ICVL_31_PATH)
    if "rad" not in visible_payload:
        raise KeyError(f"Expected 'rad' in {ICVL_31_PATH}")
    visible_original = np.asarray(visible_payload["rad"], dtype=np.float32)

    if full_original.ndim != 3 or full_original.shape[-1] != 40:
        raise ValueError(f"Expected original ICVL HxWx40, got {full_original.shape}")
    if visible_original.ndim != 3 or visible_original.shape[-1] != 31:
        raise ValueError(f"Expected original ICVL HxWx31, got {visible_original.shape}")
    if full_original.shape[:2] != visible_original.shape[:2]:
        raise ValueError(
            "ICVL 40-band and new 31-band cubes do not share the same raw spatial size: "
            f"40={full_original.shape}, 31={visible_original.shape}"
        )
    if wavelengths.size != 40 or not np.all(np.diff(wavelengths) > 0):
        raise ValueError(f"Invalid ICVL wavelength vector: {wavelengths.shape}")

    # Both products are in the original raw spatial orientation. Apply the SAME
    # transform to both. This is the geometry verified by the alignment script.
    full_aligned = ndimage.rotate(full_original, 90)
    full_aligned = full_aligned[:, ::-1, :]
    full_raw = full_aligned[392:1192, 100:900, :]

    visible_aligned = ndimage.rotate(visible_original, 90)
    visible_aligned = visible_aligned[:, ::-1, :]
    visible_raw = visible_aligned[392:1192, 100:900, :]

    if full_raw.shape != (800, 800, 40):
        raise ValueError(
            f"Expected ICVL 40-band crop (800,800,40), got {full_raw.shape}"
        )
    if visible_raw.shape != (800, 800, 31):
        raise ValueError(
            f"Expected ICVL 31-band crop (800,800,31), got {visible_raw.shape}"
        )

    full_max = float(np.max(full_raw))
    visible_max = float(np.max(visible_raw))
    common_max = max(full_max, visible_max)
    if not np.isfinite(common_max) or common_max <= 0:
        raise ValueError(f"Invalid ICVL mutual maximum: {common_max}")

    # ONE shared normalization only.
    full_norm = (full_raw / common_max).astype(np.float32, copy=False)
    visible_norm = (visible_raw / common_max).astype(np.float64, copy=False)
    rgb = generate_msi(visible_norm, RGB_SRF_BGR.T)

    i801 = int(np.argmin(np.abs(wavelengths - 801.39)))

    print("\n[ICVL unified preparation]")
    print(f"  40-band source   : {ICVL_40_PATH}")
    print(f"  31-band source   : {ICVL_31_PATH}")
    print(f"  raw source shapes: 40={full_original.shape}, 31={visible_original.shape}")
    print("  common geometry  : rotate90 -> mirror-x -> crop[392:1192,100:900]")
    print(f"  raw 40 max       : {full_max:.10g}")
    print(f"  raw 31 max       : {visible_max:.10g}")
    print(f"  raw max ratio 40/31: {full_max / visible_max:.8f}")
    print(f"  mutual max       : {common_max:.10g}")
    print(f"  normalized 40 max: {full_norm.max():.8f}")
    print(f"  normalized 31 max: {visible_norm.max():.8f}")
    print(f"  RGB shape/range  : {rgb.shape}, ({rgb.min():.8f}, {rgb.max():.8f})")
    print(f"  RGB sha256       : {fingerprint(rgb)}")
    print(f"  ~801 band        : index={i801}, wavelength={wavelengths[i801]:.4f} nm")

    root = OUTPUT_ROOT / "ICVL"
    root.mkdir(parents=True, exist_ok=True)
    return PreparedDataset(
        name="ICVL",
        wavelengths_nm=wavelengths,
        hsi_hwc=full_norm,
        rgb_hwc=rgb,
        raw_full_max=full_max,
        raw_visible_max=visible_max,
        common_max=common_max,
        encoder_checkpoint=ICVL_ENCODER,
        decoder_checkpoint=ICVL_DECODER,
        root=root,
    )


# =============================================================================
# EXPERIMENT SET: EXACTLY WHAT ANALYSES 1-4 REQUIRE
# =============================================================================
def nearest_idx(wavelengths: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths - float(nm))))


def build_experiments(wavelengths: np.ndarray) -> List[Experiment]:
    """Return the compact 16-run representative design.

    Target wavelengths are converted to the nearest actual band separately for
    each dataset, so the same scientific design is used on Chikusei and ICVL.
    """
    i461 = nearest_idx(wavelengths, 461.0)
    i549 = nearest_idx(wavelengths, 549.0)
    i652 = nearest_idx(wavelengths, 652.0)
    i714 = nearest_idx(wavelengths, 714.0)
    i801 = nearest_idx(wavelengths, 801.39)
    i935 = nearest_idx(wavelengths, 935.0)

    experiments = [
        # RGB comparison.
        Experiment("rgb_only", "rgb", (), "RGB baseline"),
        Experiment("rgb_plus_nir_801", "rgb_plus_narrowband", (i801,), "RGB + direct NIR anchor"),

        # Representative single-band mapping.
        Experiment("single_vis_461", "narrowband", (i461,), "representative blue-visible single"),
        Experiment("single_vis_549", "narrowband", (i549,), "representative green-visible single"),
        Experiment("single_vis_652", "narrowband", (i652,), "representative red-visible single"),
        Experiment("single_nir_801", "narrowband", (i801,), "representative NIR single + profile"),
        Experiment("single_nir_935", "narrowband", (i935,), "representative far-NIR single"),

        # Representative pair controls and controlled location scans.
        Experiment("two_vis_461_549", "narrowband", tuple(sorted((i461, i549))), "visible+visible control"),
        Experiment("two_nir_801_935", "narrowband", tuple(sorted((i801, i935))), "NIR+NIR control + profile"),
        Experiment("spanning_461_801", "narrowband", tuple(sorted((i461, i801))), "visible-anchor scan"),
        Experiment("spanning_549_801", "narrowband", tuple(sorted((i549, i801))), "reference visible+NIR pair + profile"),
        Experiment("spanning_652_801", "narrowband", tuple(sorted((i652, i801))), "visible-anchor scan"),
        Experiment("spanning_549_714", "narrowband", tuple(sorted((i549, i714))), "NIR-anchor scan"),
        Experiment("spanning_549_935", "narrowband", tuple(sorted((i549, i935))), "NIR-anchor scan"),

        # 3- and 4-band spanning configurations retained for the band-count curve.
        Experiment("spanning_461_549_801", "narrowband", tuple(sorted((i461, i549, i801))), "three-band spanning"),
        Experiment("spanning_461_549_652_801", "narrowband", tuple(sorted((i461, i549, i652, i801))), "four-band spanning"),
    ]

    # Guard against accidental duplicate physical inputs caused by nearest-band mapping.
    keys = [(e.source, tuple(e.band_indices)) for e in experiments]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Representative targets collapsed onto duplicate physical bands. "
            f"Actual wavelengths: {wavelengths.tolist()}"
        )
    if len(experiments) != 16:
        raise RuntimeError(f"Expected 16 representative runs, got {len(experiments)}")
    return experiments


def save_manifest(dataset: PreparedDataset, experiments: Sequence[Experiment]) -> None:
    rows = []
    for exp in experiments:
        rows.append(
            {
                "experiment": exp.name,
                "source": exp.source,
                "n_channels": exp.n_channels,
                "band_indices": ";".join(map(str, exp.band_indices)),
                "wavelengths_nm": ";".join(f"{dataset.wavelengths_nm[i]:.4f}" for i in exp.band_indices),
                "analysis_role": exp.analysis_role,
            }
        )
    pd.DataFrame(rows).to_csv(dataset.root / "manifest.csv", index=False)

    with open(dataset.root / "normalization.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset.name,
                "raw_full_40band_max": dataset.raw_full_max,
                "raw_visible_31band_max": dataset.raw_visible_max,
                "mutual_common_max": dataset.common_max,
                "normalization": "both 31-band and 40-band cubes divided once by mutual_common_max",
                "seeds_to_run": list(SEEDS_TO_RUN),
                "baseline_seed0_root": str(BASELINE_SEED0_ROOT),
                "robustness_configs": list(ROBUSTNESS_CONFIGS),
                "evaluation_every": EVAL_EVERY,
                "visualization_every": VIZ_EVERY,
                "early_stop_metrics": ["centered_abundance_gram_sam", "reconstruction_relative_change"],
                "early_stop_patience_epochs": SAM_PATIENCE_EPOCHS,
                "early_stop_min_sam_decrease": SAM_MIN_DECREASE,
                "recon_window_epochs": RECON_WINDOW_EPOCHS,
                "recon_change_threshold": RECON_REL_CHANGE_THRESHOLD,
                "recon_stable_windows": RECON_STABLE_WINDOWS,
                "recon_early_stop_min_epoch": RECON_EARLY_STOP_MIN_EPOCH,
                "checkpoint_selection_metric": "original_composite_loss",
            },
            f,
            indent=2,
        )


# =============================================================================
# LOSSES, ABUNDANCE, METRICS
# =============================================================================
def get_stick_segments(v: torch.Tensor) -> torch.Tensor:
    one_minus = 1.0 - v
    cumulative = torch.cumprod(one_minus, dim=1)
    previous = torch.cat([torch.ones_like(cumulative[:, :1]), cumulative[:, :-1]], dim=1)
    return v * previous


def l_2_1_loss(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    # Exact old working implementation.
    error = ((((x_pred - x_true).pow(2)).sum(1)).pow(1 / 2)).sum(0)
    return error


def sam_mean(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    # Exact old working implementation.
    n = x_true.size()[0]
    numerator = torch.sum(torch.mul(x_true, x_pred), dim=1)
    denom1 = torch.sqrt(torch.sum(torch.pow(x_true, 2), dim=1))
    denom2 = torch.sqrt(torch.sum(torch.pow(x_pred, 2), dim=1))
    sam = torch.acos(
        torch.div(numerator, torch.mul(denom1, denom2)).clamp(
            -1.0 + 1e-8, 1.0 - 1e-8
        )
    )
    sam = torch.mul(torch.div(sam, np.pi), 180)
    sam = torch.div(torch.sum(sam), n)
    return sam


def ergas_loss(x_true: torch.Tensor, x_pred: torch.Tensor, scale: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    means = x_true.mean(dim=0)
    mse = ((x_true - x_pred) ** 2).mean(dim=0)
    return 100.0 / scale * torch.sqrt((mse / (means ** 2 + eps)).mean())


def psnr_mean(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    # Exact form used in the old working script.
    msr = ((x_true - x_pred) ** 2).mean(dim=1)
    max2 = (x_true ** 2).max()
    psnrall = 10.0 * torch.log10(max2 / msr)
    return psnrall.mean()


def centered_gram(abundances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gram = abundances.T @ abundances / abundances.shape[0]
    centered = gram - gram.mean(dim=0, keepdim=True)
    diagonal = torch.diag(gram).unsqueeze(0)
    return centered, diagonal


def compute_training_loss(
    mapped_hsi: torch.Tensor,
    reconstructed_hsi: torch.Tensor,
    hr_gram_centered: torch.Tensor,
    hr_gram_diagonal: torch.Tensor,
    lr_gram_centered: torch.Tensor,
    lr_gram_diagonal: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    reconstruction_l21 = l_2_1_loss(mapped_hsi + 1e-8, reconstructed_hsi)

    gram_l21 = l_2_1_loss(hr_gram_centered, lr_gram_centered)
    gram_ergas = ergas_loss(hr_gram_centered, lr_gram_centered)
    gram_sam = sam_mean(hr_gram_centered, lr_gram_centered)
    gram_sam_t = sam_mean(hr_gram_centered.T, lr_gram_centered.T)

    diagonal_l21 = l_2_1_loss(hr_gram_diagonal, lr_gram_diagonal)
    diagonal_sam = sam_mean(hr_gram_diagonal, lr_gram_diagonal)

    loss_multi = gram_l21 + gram_ergas + 10.0 * gram_sam + gram_sam_t
    loss_diagonal = diagonal_l21 + 10.0 * diagonal_sam
    total = 0.1 * reconstruction_l21 + 10.0 * loss_diagonal + 100.0 * loss_multi

    return total, {
        "reconstruction_l21": reconstruction_l21,
        "gram_l21": gram_l21,
        "gram_ergas": gram_ergas,
        "gram_sam": gram_sam,
        "gram_sam_t": gram_sam_t,
        "diagonal_l21": diagonal_l21,
        "diagonal_sam": diagonal_sam,
    }


@torch.no_grad()
def evaluate_reconstruction(gt: torch.Tensor, pred: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    error = pred - gt
    rmse = torch.sqrt(torch.mean(error ** 2, dim=0))
    mae = torch.mean(torch.abs(error), dim=0)
    mse = torch.mean(error ** 2, dim=0)
    psnr_band = 10.0 * torch.log10((gt ** 2).max() / (mse + 1e-8))
    gt_mean = torch.mean(gt, dim=0)

    metrics = {
        "SAM": float(sam_mean(gt, pred).item()),
        "PSNR": float(psnr_mean(gt, pred).item()),
        "ERGAS": float(ergas_loss(gt, pred).item()),
    }
    profiles = {
        "RMSE": rmse.cpu().numpy(),
        "MAE": mae.cpu().numpy(),
        "PSNR": psnr_band.cpu().numpy(),
        "GT_mean": gt_mean.cpu().numpy(),
    }
    return metrics, profiles


# =============================================================================
# CHECKPOINT / TRAINING HELPERS
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def load_module_checkpoint(module: nn.Module, path: Path, device: torch.device) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state, strict=True)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_hr_input(dataset: PreparedDataset, exp: Experiment) -> np.ndarray:
    if exp.source == "rgb":
        return dataset.rgb_hwc.astype(np.float32, copy=True)
    if exp.source == "narrowband":
        return dataset.hsi_hwc[..., list(exp.band_indices)].astype(np.float32, copy=True)
    if exp.source == "rgb_plus_narrowband":
        narrow = dataset.hsi_hwc[..., list(exp.band_indices)].astype(np.float32, copy=False)
        return np.concatenate((dataset.rgb_hwc, narrow), axis=2).astype(np.float32, copy=False)
    raise ValueError(exp.source)


def exp_dir(dataset: PreparedDataset, exp: Experiment, seed: int = SEED) -> Path:
    return dataset.root / "experiments" / exp.name / f"seed_{int(seed)}"


def save_profile_csv(path: Path, wavelengths: np.ndarray, profiles: Dict[str, np.ndarray]) -> None:
    frame = pd.DataFrame(
        {
            "band_index": np.arange(len(wavelengths), dtype=int),
            "wavelength_nm": wavelengths,
            "RMSE": profiles["RMSE"],
            "MAE": profiles["MAE"],
            "PSNR": profiles["PSNR"],
            "GT_mean": profiles["GT_mean"],
        }
    )
    frame.to_csv(path, index=False)


def reuse_compatible_completed_run(dataset: PreparedDataset, exp: Experiment, out_dir: Path) -> bool:
    """Optional legacy-result reuse helper. Disabled in this clean master script."""
    for root in REUSE_COMPATIBLE_ROOTS:
        old_exp_root = root / dataset.name / "experiments"
        if not old_exp_root.exists():
            continue
        for result_path in old_exp_root.glob(f"*/seed_{SEED}/result.json"):
            try:
                old = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if old.get("status") != "complete":
                continue
            if old.get("dataset") != dataset.name:
                continue
            if old.get("source") != exp.source:
                continue
            if tuple(sorted(int(x) for x in old.get("band_indices", []))) != tuple(sorted(exp.band_indices)):
                continue
            if int(old.get("seed", -1)) != SEED:
                continue
            old_common = float(old.get("common_max", float("nan")))
            if not np.isfinite(old_common) or not np.isclose(old_common, dataset.common_max, rtol=1e-7, atol=1e-10):
                continue

            old_dir = result_path.parent
            shutil.copytree(old_dir, out_dir, dirs_exist_ok=True)
            copied_result = out_dir / "result.json"
            payload = json.loads(copied_result.read_text(encoding="utf-8"))
            payload["experiment"] = exp.name
            payload["reused_from"] = str(old_dir)
            copied_result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  [reuse] {exp.name} <- {old_dir}")
            return True
    return False


def show_training_visualization(
    dataset: PreparedDataset,
    exp: Experiment,
    epoch: int,
    gt_absolute: torch.Tensor,
    reconstruction_absolute: torch.Tensor,
    hr_gram_diagonal: torch.Tensor,
    lr_gram_diagonal: torch.Tensor,
    recon_sam: float,
) -> None:
    """Show the same style of diagnostic plots in the interactive console.

    Nothing is written to disk.
    """
    h, w, c = dataset.hsi_hwc.shape
    gt_np = gt_absolute.detach().cpu().numpy().reshape(h, w, c)
    pr_np = reconstruction_absolute.detach().cpu().numpy().reshape(h, w, c)
    xx = dataset.wavelengths_nm

    # Spectra at three representative pixels.
    plt.figure(figsize=(12, 8))
    for fy, fx, col, tag in [(0.2, 0.2, "b", "1"), (0.5, 0.5, "g", "2"), (0.8, 0.8, "r", "3")]:
        yi = min(h - 1, int(h * fy))
        xi = min(w - 1, int(w * fx))
        plt.plot(xx, gt_np[yi, xi, :], color=col, linestyle="-.", label=f"gt{tag}")
        plt.plot(xx, pr_np[yi, xi, :], color=col, linestyle="-", label=f"rec{tag}")
    plt.title(f"{dataset.name} | {exp.name} | epoch {epoch} | recon-SAM {recon_sam:.3f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance / normalized intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Per-pixel reconstruction SAM map + Gram diagonal overlap.
    gt_flat = gt_np.reshape(-1, c).astype(np.float64)
    pr_flat = pr_np.reshape(-1, c).astype(np.float64)
    eps = 1e-8
    nom = (gt_flat * pr_flat).sum(axis=1)
    den = np.sqrt((gt_flat ** 2).sum(axis=1)) * np.sqrt((pr_flat ** 2).sum(axis=1)) + eps
    sam_map = np.degrees(np.arccos(np.clip(nom / den, -1.0, 1.0))).reshape(h, w)

    figure, axarr = plt.subplots(1, 2, figsize=(10, 4))
    im = axarr[0].imshow(sam_map)
    axarr[0].set_title(f"recon SAM map E{epoch}")
    figure.colorbar(im, ax=axarr[0], fraction=0.046)

    gmd_hr = hr_gram_diagonal.detach().cpu().squeeze().numpy()
    gmd_lr = lr_gram_diagonal.detach().cpu().squeeze().numpy()
    axarr[1].plot(gmd_hr, marker="o", label="GMD_hr_hsi")
    axarr[1].plot(gmd_lr, marker="o", label="GMD_lr_hsi")
    axarr[1].set_title(f"Gram diagonal E{epoch}")
    axarr[1].legend()
    figure.tight_layout()
    plt.show()


def train_dataset(dataset: PreparedDataset) -> None:
    print(f"\n{'='*90}\nROBUSTNESS TRAINING DATASET: {dataset.name}\n{'='*90}")

    all_experiments = build_experiments(dataset.wavelengths_nm)
    exp_by_name = {e.name: e for e in all_experiments}

    missing = set(ROBUSTNESS_CONFIGS) - set(exp_by_name)
    if missing:
        raise RuntimeError(f"Missing requested robustness configurations: {sorted(missing)}")

    experiments = [exp_by_name[name] for name in ROBUSTNESS_CONFIGS]
    save_manifest(dataset, experiments)

    total_runs = len(experiments) * len(SEEDS_TO_RUN)
    print(
        f"[{dataset.name}] robustness runs: {len(experiments)} configs x "
        f"{len(SEEDS_TO_RUN)} new seeds = {total_runs}"
    )
    print(f"[{dataset.name}] configs: {[e.name for e in experiments]}")
    print(f"[{dataset.name}] seeds  : {SEEDS_TO_RUN}")

    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={DEVICE}, but CUDA is unavailable")

    encoder = EncoderLRHSI().to(device).float()
    decoder = DecoderHSI().to(device).float()
    load_module_checkpoint(encoder, dataset.encoder_checkpoint, device)
    load_module_checkpoint(decoder, dataset.decoder_checkpoint, device)

    # Fixed GT and LR-HSI representation for every seed/configuration.
    # Intentionally unchanged from seed 0 for strict comparability.
    hsi = dataset.hsi_hwc
    lr_hsi = generate_lr_hsi(hsi, SCALE_FACTOR)

    gt_raw_cpu = torch.from_numpy(hsi.reshape(-1, 40)).float()
    gt_mean_cpu = gt_raw_cpu.mean(dim=0, keepdim=True)
    gt_centered_cpu = gt_raw_cpu - gt_mean_cpu

    lr_raw_cpu = torch.from_numpy(lr_hsi.reshape(-1, 40)).float()
    lr_mean_cpu = lr_raw_cpu.mean(dim=0, keepdim=True)
    lr_centered = (lr_raw_cpu - lr_mean_cpu).to(device)

    with torch.no_grad():
        lr_latent = encoder(lr_centered)
        lr_abundance = get_stick_segments(lr_latent).clamp(1e-8, 1.0 - 1e-8)
        lr_gram_centered, lr_gram_diagonal = centered_gram(lr_abundance)

    gt_centered = gt_centered_cpu.to(device)
    gt_mean = gt_mean_cpu.to(device)
    gt_absolute = gt_centered + gt_mean

    # Observed LR-HSI mean used by the unsupervised convergence criterion.
    lr_mean = lr_mean_cpu.to(device)

    run_i = 0

    for seed in SEEDS_TO_RUN:
        for exp in experiments:
            run_i += 1

            out_dir = exp_dir(dataset, exp, seed=seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path = out_dir / "result.json"
            best_path = out_dir / "best_mapper.pth"
            latest_path = out_dir / "latest_mapper.pth"
            history_path = out_dir / "history.csv"

            if SKIP_COMPLETED and result_path.exists():
                try:
                    old = json.loads(result_path.read_text(encoding="utf-8"))
                    if old.get("status") == "complete":
                        print(
                            f"[{dataset.name} {run_i:03d}/{total_runs}] "
                            f"skip complete: {exp.name} seed={seed}"
                        )
                        continue
                except Exception:
                    pass

            print(
                f"\n[{dataset.name} {run_i:03d}/{total_runs}] "
                f"{exp.name} | seed={seed} | "
                f"source={exp.source} | channels={exp.n_channels} | "
                f"bands={[float(dataset.wavelengths_nm[i]) for i in exp.band_indices]}"
            )

            # ONLY stochastic factor changed relative to seed 0.
            set_seed(seed)

            hr_input_hwc = build_hr_input(dataset, exp)
            hr_raw_cpu = torch.from_numpy(
                hr_input_hwc.reshape(-1, hr_input_hwc.shape[2])
            ).float()
            hr_mean_cpu = hr_raw_cpu.mean(dim=0, keepdim=True)
            hr_input = (hr_raw_cpu - hr_mean_cpu).to(device)

            mapper = InputToHSIMapper(exp.n_channels, 40).to(device).float()
            optimizer = torch.optim.Adam(
                mapper.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )

            start_epoch = 0
            best_loss = float("inf")
            saved_gram_sam = float("inf")
            saved_gram_psnr = float("nan")
            saved_epoch = -1

            plateau_gram_sam = float("inf")
            plateau_epoch = 0

            recon_window_reference = None
            recon_window_reference_epoch = None
            recon_stable_windows = 0
            recon_relative_change = float("nan")

            if RESUME_INCOMPLETE and latest_path.exists():
                payload = torch.load(latest_path, map_location=device)
                mapper.load_state_dict(payload["state_dict"])
                optimizer.load_state_dict(payload["optimizer"])
                start_epoch = int(payload["epoch"]) + 1
                plateau_gram_sam = float(
                    payload.get("plateau_gram_sam", plateau_gram_sam)
                )
                plateau_epoch = int(
                    payload.get("plateau_epoch", max(0, start_epoch - 1))
                )
                print(f"  [resume] latest state from epoch {start_epoch}")

            if best_path.exists():
                best_payload_existing = torch.load(best_path, map_location="cpu")
                best_loss = float(best_payload_existing.get("best_loss", best_loss))
                saved_gram_sam = float(
                    best_payload_existing.get("saved_gram_sam", saved_gram_sam)
                )
                saved_gram_psnr = float(
                    best_payload_existing.get("saved_gram_psnr", saved_gram_psnr)
                )
                saved_epoch = int(best_payload_existing.get("epoch", saved_epoch))

            history_fields = [
                "epoch", "total_loss", "recon_SAM", "recon_PSNR", "recon_ERGAS",
                "reconstruction_l21", "gram_l21", "gram_ergas", "gram_sam",
                "gram_sam_t", "diagonal_l21", "diagonal_sam", "gram_psnr",
                "best_loss", "saved_gram_sam", "saved_gram_psnr", "saved_epoch",
                "plateau_gram_sam", "plateau_epoch",
                "recon_relative_change", "recon_stable_count", "seconds",
            ]

            run_start = time.time()
            mapper.train()
            last_epoch = start_epoch - 1

            for epoch in range(start_epoch, MAX_EPOCHS):
                last_epoch = epoch
                tick = time.time()
                optimizer.zero_grad(set_to_none=True)

                mapped_hsi = mapper(hr_input).clamp(-1.0 + 1e-8, 1.0 - 1e-8)
                hr_latent = encoder(mapped_hsi)
                hr_abundance = get_stick_segments(hr_latent).clamp(
                    1e-8, 1.0 - 1e-8
                )
                reconstruction = decoder(hr_abundance).clamp(
                    -1.0 + 1e-8, 1.0 - 1e-8
                )
                hr_gram_centered, hr_gram_diagonal = centered_gram(hr_abundance)

                total_loss, parts = compute_training_loss(
                    mapped_hsi,
                    reconstruction,
                    hr_gram_centered,
                    hr_gram_diagonal,
                    lr_gram_centered,
                    lr_gram_diagonal,
                )

                loss_value = float(total_loss.detach().item())
                current_gram_sam = float(parts["gram_sam"].detach().item())
                current_gram_psnr = float(
                    psnr_mean(
                        hr_gram_centered + 1e-8,
                        lr_gram_centered + 1e-8,
                    ).detach().item()
                )

                if loss_value < best_loss:
                    best_loss = loss_value
                    saved_gram_sam = current_gram_sam
                    saved_gram_psnr = current_gram_psnr
                    saved_epoch = epoch

                    atomic_torch_save(
                        {
                            "epoch": saved_epoch,
                            "best_loss": best_loss,
                            "saved_gram_sam": saved_gram_sam,
                            "saved_gram_psnr": saved_gram_psnr,
                            "state_dict": mapper.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "experiment": asdict(exp),
                            "seed": int(seed),
                            "selection_metric": "original_composite_loss",
                            "gt_metrics_used_for_selection": False,
                        },
                        best_path,
                    )
                    print(f"Updating loss {best_loss} at epoch of {epoch}")

                print(
                    "last saved sam: {}, psnr: {} at epoch: {}".format(
                        saved_gram_sam, saved_gram_psnr, saved_epoch
                    )
                )

                if not np.isfinite(plateau_gram_sam):
                    plateau_gram_sam = current_gram_sam
                    plateau_epoch = epoch
                elif current_gram_sam < plateau_gram_sam - SAM_MIN_DECREASE:
                    plateau_gram_sam = current_gram_sam
                    plateau_epoch = epoch

                evaluate_now = (epoch % EVAL_EVERY == 0) or (
                    epoch == MAX_EPOCHS - 1
                )
                recon_change_stop = False

                if evaluate_now:
                    current_reconstruction_eval = (
                        reconstruction.detach() + lr_mean
                    ).to("cpu", dtype=torch.float32)

                    if recon_window_reference is None:
                        recon_window_reference = current_reconstruction_eval
                        recon_window_reference_epoch = epoch
                        recon_relative_change = float("nan")
                        recon_stable_windows = 0

                    elif (
                        epoch - recon_window_reference_epoch
                    ) >= RECON_WINDOW_EPOCHS:
                        diff_norm = torch.linalg.vector_norm(
                            current_reconstruction_eval - recon_window_reference
                        )
                        ref_norm = torch.linalg.vector_norm(
                            recon_window_reference
                        ).clamp_min(1e-12)
                        recon_relative_change = float(
                            (diff_norm / ref_norm).item()
                        )

                        if (
                            epoch >= RECON_EARLY_STOP_MIN_EPOCH
                            and recon_relative_change < RECON_REL_CHANGE_THRESHOLD
                        ):
                            recon_stable_windows += 1
                        else:
                            recon_stable_windows = 0

                        print(
                            f"  [recon-window] epochs "
                            f"{recon_window_reference_epoch}->{epoch}: "
                            f"relative full-HSI change={recon_relative_change:.3e}; "
                            f"stable={recon_stable_windows}/{RECON_STABLE_WINDOWS}"
                        )

                        recon_window_reference = current_reconstruction_eval
                        recon_window_reference_epoch = epoch

                        if (
                            epoch >= RECON_EARLY_STOP_MIN_EPOCH
                            and recon_stable_windows >= RECON_STABLE_WINDOWS
                        ):
                            recon_change_stop = True

                    # Intentionally unchanged from the completed seed-0 study
                    # so the robustness comparison isolates random seed only.
                    with torch.no_grad():
                        reconstruction_absolute = reconstruction + gt_mean
                        metrics, _ = evaluate_reconstruction(
                            gt_absolute, reconstruction_absolute
                        )

                    row = {
                        "epoch": epoch,
                        "total_loss": loss_value,
                        "recon_SAM": metrics["SAM"],
                        "recon_PSNR": metrics["PSNR"],
                        "recon_ERGAS": metrics["ERGAS"],
                        "gram_psnr": current_gram_psnr,
                        "best_loss": best_loss,
                        "saved_gram_sam": saved_gram_sam,
                        "saved_gram_psnr": saved_gram_psnr,
                        "saved_epoch": saved_epoch,
                        "plateau_gram_sam": plateau_gram_sam,
                        "plateau_epoch": plateau_epoch,
                        "recon_relative_change": recon_relative_change,
                        "recon_stable_count": recon_stable_windows,
                        "seconds": time.time() - tick,
                    }
                    row.update(
                        {k: float(v.detach().item()) for k, v in parts.items()}
                    )

                    write_header = not history_path.exists()
                    with open(
                        history_path, "a", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.DictWriter(f, fieldnames=history_fields)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)

                    print(
                        f"  [{dataset.name}/{exp.name}/seed={seed}] "
                        f"epoch={epoch:07d} loss={loss_value:.6f} "
                        f"gram-SAM={current_gram_sam:.6f} "
                        f"plateau={plateau_gram_sam:.6f}@{plateau_epoch} | "
                        f"recon-SAM={metrics['SAM']:.4f} "
                        f"recon-PSNR={metrics['PSNR']:.4f} "
                        f"recon-ERGAS={metrics['ERGAS']:.4f} | "
                        f"recon-change={recon_relative_change:.3e} "
                        f"stable-windows={recon_stable_windows}/{RECON_STABLE_WINDOWS}"
                    )

                    if epoch % VIZ_EVERY == 0:
                        show_training_visualization(
                            dataset=dataset,
                            exp=exp,
                            epoch=epoch,
                            gt_absolute=gt_absolute,
                            reconstruction_absolute=reconstruction_absolute,
                            hr_gram_diagonal=hr_gram_diagonal,
                            lr_gram_diagonal=lr_gram_diagonal,
                            recon_sam=metrics["SAM"],
                        )

                    if recon_change_stop:
                        print(
                            f"  [early-stop] predicted spectra stable: "
                            f"{recon_relative_change:.3e} over "
                            f"{RECON_WINDOW_EPOCHS} epochs for "
                            f"{RECON_STABLE_WINDOWS} consecutive windows."
                        )
                        break

                if (
                    epoch >= SAM_PATIENCE_EPOCHS
                    and (epoch - plateau_epoch) >= SAM_PATIENCE_EPOCHS
                ):
                    print(
                        f"  [early-stop] centered Gram-SAM has not improved by "
                        f">= {SAM_MIN_DECREASE} for "
                        f"{SAM_PATIENCE_EPOCHS} epochs."
                    )
                    break

                total_loss.backward()

                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(
                        mapper.parameters(), GRAD_CLIP
                    )

                optimizer.step()

                if (
                    epoch > 0
                    and SAVE_LATEST_EVERY > 0
                    and epoch % SAVE_LATEST_EVERY == 0
                ):
                    atomic_torch_save(
                        {
                            "epoch": epoch,
                            "state_dict": mapper.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "plateau_gram_sam": plateau_gram_sam,
                            "plateau_epoch": plateau_epoch,
                            "recon_relative_change": recon_relative_change,
                            "recon_stable_windows": recon_stable_windows,
                            "experiment": asdict(exp),
                            "seed": int(seed),
                        },
                        latest_path,
                    )

            atomic_torch_save(
                {
                    "epoch": last_epoch,
                    "state_dict": mapper.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "plateau_gram_sam": plateau_gram_sam,
                    "plateau_epoch": plateau_epoch,
                    "recon_relative_change": recon_relative_change,
                    "recon_stable_windows": recon_stable_windows,
                    "experiment": asdict(exp),
                    "seed": int(seed),
                },
                latest_path,
            )

            if not best_path.exists():
                raise RuntimeError(
                    f"No best checkpoint for "
                    f"{dataset.name}/{exp.name}/seed_{seed}"
                )

            best_payload = torch.load(best_path, map_location=device)
            mapper.load_state_dict(best_payload["state_dict"])
            mapper.eval()

            with torch.no_grad():
                mapped = mapper(hr_input).clamp(-1.0 + 1e-8, 1.0 - 1e-8)
                abundance = get_stick_segments(encoder(mapped)).clamp(
                    1e-8, 1.0 - 1e-8
                )
                recon_centered = decoder(abundance).clamp(
                    -1.0 + 1e-8, 1.0 - 1e-8
                )
                recon_absolute = recon_centered + gt_mean
                metrics, profiles = evaluate_reconstruction(
                    gt_absolute, recon_absolute
                )

            if exp.name in PROFILE_EXPERIMENTS:
                save_profile_csv(
                    out_dir / "per_band_metrics.csv",
                    dataset.wavelengths_nm,
                    profiles,
                )

            result = {
                "status": "complete",
                "dataset": dataset.name,
                "experiment": exp.name,
                "source": exp.source,
                "n_channels": exp.n_channels,
                "band_indices": list(exp.band_indices),
                "wavelengths_nm": [
                    float(dataset.wavelengths_nm[i])
                    for i in exp.band_indices
                ],
                "seed": int(seed),
                "best_epoch": int(best_payload["epoch"]),
                "best_composite_loss": float(best_payload["best_loss"]),
                "saved_gram_sam": float(best_payload["saved_gram_sam"]),
                "saved_gram_psnr": float(best_payload["saved_gram_psnr"]),
                "SAM": metrics["SAM"],
                "PSNR": metrics["PSNR"],
                "ERGAS": metrics["ERGAS"],
                "runtime_seconds": time.time() - run_start,
                "common_max": dataset.common_max,
                "raw_full_max": dataset.raw_full_max,
                "raw_visible_max": dataset.raw_visible_max,
                "checkpoint_selection": {
                    "metric": "original_composite_loss",
                    "checked_every_epoch": True,
                    "gt_metrics_used": False,
                },
                "early_stopping": {
                    "logic": "stop_if_either_criterion_plateaus",
                    "gram_sam": {
                        "metric": "centered_abundance_gram_sam",
                        "checked_every_epoch": True,
                        "min_decrease": SAM_MIN_DECREASE,
                        "patience_epochs": SAM_PATIENCE_EPOCHS,
                    },
                    "reconstruction_change": {
                        "metric": (
                            "relative_L2_change_of_full_predicted_HSI_"
                            "LRmean_anchored"
                        ),
                        "evaluation_every_epochs": EVAL_EVERY,
                        "window_epochs": RECON_WINDOW_EPOCHS,
                        "threshold": RECON_REL_CHANGE_THRESHOLD,
                        "consecutive_stable_windows": RECON_STABLE_WINDOWS,
                        "minimum_epoch": RECON_EARLY_STOP_MIN_EPOCH,
                        "last_relative_change": recon_relative_change,
                        "last_stable_windows": recon_stable_windows,
                        "uses_hr_gt": False,
                        "absolute_scale_anchor": "observed_LR_HSI_mean",
                    },
                    "gt_metrics_used": False,
                },
                "robustness_run": True,
                "baseline_seed0_not_retrained": True,
            }

            result_path.write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

            print(
                f"  [complete] seed={seed} "
                f"recon-SAM={metrics['SAM']:.4f}, "
                f"recon-PSNR={metrics['PSNR']:.4f}, "
                f"recon-ERGAS={metrics['ERGAS']:.4f}, "
                f"best-loss epoch={best_payload['epoch']}"
            )

            del hr_input_hwc, hr_raw_cpu, hr_mean_cpu, hr_input, mapper, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del encoder, decoder, lr_centered, lr_latent, lr_abundance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# COLLECT RESULTS
# =============================================================================
def _read_result_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if row.get("status") != "complete":
        return None
    row["result_path"] = str(path)
    return row


def collect_results(dataset: PreparedDataset) -> pd.DataFrame:
    """Collect only the new seed-1/seed-2 robustness runs."""
    rows: List[Dict[str, object]] = []
    exp_root = dataset.root / "experiments"
    if not exp_root.exists():
        return pd.DataFrame()

    for exp_name in ROBUSTNESS_CONFIGS:
        for seed in SEEDS_TO_RUN:
            path = exp_root / exp_name / f"seed_{seed}" / "result.json"
            if not path.exists():
                continue
            row = _read_result_json(path)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def collect_three_seed_robustness(
    prepared: Sequence[PreparedDataset],
) -> pd.DataFrame:
    """
    Combine seed 0 from the existing full study with new seeds 1 and 2.
    Seed 0 is read only; it is never retrained by this script.
    """
    rows: List[Dict[str, object]] = []

    for dataset in prepared:
        seed0_exp_root = (
            BASELINE_SEED0_ROOT / dataset.name / "experiments"
        )

        for exp_name in ROBUSTNESS_CONFIGS:
            p0 = (
                seed0_exp_root
                / exp_name
                / "seed_0"
                / "result.json"
            )
            if p0.exists():
                row = _read_result_json(p0)
                if row is not None:
                    row = dict(row)
                    row["seed"] = 0
                    row["source_root"] = "existing_seed0_full_study"
                    rows.append(row)

        new_exp_root = dataset.root / "experiments"
        for exp_name in ROBUSTNESS_CONFIGS:
            for seed in SEEDS_TO_RUN:
                p = (
                    new_exp_root
                    / exp_name
                    / f"seed_{seed}"
                    / "result.json"
                )
                if not p.exists():
                    continue
                row = _read_result_json(p)
                if row is not None:
                    row = dict(row)
                    row["seed"] = int(seed)
                    row["source_root"] = "new_multiseed_robustness"
                    rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS 1: REPRESENTATIVE SINGLE/PPAIR MAPPING
# =============================================================================
def analysis_systematic_mapping(dataset: PreparedDataset, frame: pd.DataFrame, out_root: Path) -> None:
    """Summarize the compact representative spectral design.

    This is intentionally NOT described as exhaustive mapping. It provides
    representative single-band behavior plus controlled pair-category contrasts.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    wavelengths = dataset.wavelengths_nm

    singles = frame[(frame["source"] == "narrowband") & (frame["n_channels"] == 1)].copy()
    pairs = frame[(frame["source"] == "narrowband") & (frame["n_channels"] == 2)].copy()

    single_rows = []
    for _, row in singles.iterrows():
        i = int(row["band_indices"][0])
        single_rows.append({
            "band_index": i,
            "wavelength_nm": float(wavelengths[i]),
            "SAM": float(row["SAM"]),
            "PSNR": float(row["PSNR"]),
            "ERGAS": float(row["ERGAS"]),
            "experiment": row["experiment"],
        })
    single_df = pd.DataFrame(single_rows).sort_values("wavelength_nm") if single_rows else pd.DataFrame()
    single_df.to_csv(out_root / "representative_single_band_scan.csv", index=False)

    if not single_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(single_df["wavelength_nm"], single_df["SAM"], marker="o")
        ax.axvline(VISIBLE_MAX_NM, linestyle="--", linewidth=1)
        ax.set_xlabel("Representative wavelength (nm)")
        ax.set_ylabel("SAM (degree)")
        ax.set_title(f"{dataset.name}: representative single-band reconstruction")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    pair_rows = []
    for _, row in pairs.iterrows():
        i, j = sorted(int(x) for x in row["band_indices"])
        wi, wj = float(wavelengths[i]), float(wavelengths[j])
        if wi <= VISIBLE_MAX_NM and wj <= VISIBLE_MAX_NM:
            region = "visible+visible"
        elif wi > VISIBLE_MAX_NM and wj > VISIBLE_MAX_NM:
            region = "NIR+NIR"
        else:
            region = "visible+NIR"
        pair_rows.append({
            "band_i": i, "band_j": j,
            "wavelength_i_nm": wi, "wavelength_j_nm": wj,
            "pair_label": f"{wi:.0f}+{wj:.0f}",
            "region": region,
            "SAM": float(row["SAM"]),
            "PSNR": float(row["PSNR"]),
            "ERGAS": float(row["ERGAS"]),
            "experiment": row["experiment"],
        })

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values("SAM")
        pair_df.to_csv(out_root / "representative_two_band_pairs.csv", index=False)
        region_summary = pair_df.groupby("region", as_index=False).agg(
            n_pairs=("SAM", "count"),
            SAM_mean=("SAM", "mean"),
            SAM_median=("SAM", "median"),
            SAM_min=("SAM", "min"),
            SAM_max=("SAM", "max"),
        )
        region_summary.to_csv(out_root / "representative_pair_region_summary.csv", index=False)

        plot_df = pair_df.sort_values(["region", "wavelength_i_nm", "wavelength_j_nm"])
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(np.arange(len(plot_df)), plot_df["SAM"].to_numpy(float))
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df["pair_label"], rotation=45, ha="right")
        ax.set_ylabel("SAM (degree)")
        ax.set_xlabel("Representative wavelength pair (nm)")
        ax.set_title(f"{dataset.name}: representative two-band configurations")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# ANALYSIS 2: CONTROLLED LOCATION SCANS
# =============================================================================
def analysis_location_scans(dataset: PreparedDataset, frame: pd.DataFrame, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    w = dataset.wavelengths_nm
    i549 = nearest_idx(w, 549.0)
    i801 = nearest_idx(w, 801.39)

    visible_targets = (461.0, 549.0, 652.0)
    nir_targets = (714.0, 801.39, 935.0)

    vis_rows = []
    for target in visible_targets:
        vi = nearest_idx(w, target)
        row = result_by_bands(frame, tuple(sorted((vi, i801))))
        if row is not None:
            vis_rows.append({
                "visible_band_index": vi,
                "visible_wavelength_nm": float(w[vi]),
                "fixed_nir_wavelength_nm": float(w[i801]),
                "SAM": float(row["SAM"]),
                "PSNR": float(row["PSNR"]),
                "ERGAS": float(row["ERGAS"]),
            })
    vis_df = pd.DataFrame(vis_rows).drop_duplicates("visible_band_index").sort_values("visible_wavelength_nm") if vis_rows else pd.DataFrame()
    vis_df.to_csv(out_root / "visible_anchor_scan_fixed_801.csv", index=False)

    nir_rows = []
    for target in nir_targets:
        ni = nearest_idx(w, target)
        row = result_by_bands(frame, tuple(sorted((i549, ni))))
        if row is not None:
            nir_rows.append({
                "fixed_visible_wavelength_nm": float(w[i549]),
                "nir_band_index": ni,
                "nir_wavelength_nm": float(w[ni]),
                "SAM": float(row["SAM"]),
                "PSNR": float(row["PSNR"]),
                "ERGAS": float(row["ERGAS"]),
            })
    nir_df = pd.DataFrame(nir_rows).drop_duplicates("nir_band_index").sort_values("nir_wavelength_nm") if nir_rows else pd.DataFrame()
    nir_df.to_csv(out_root / "nir_anchor_scan_fixed_549.csv", index=False)

    if not vis_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(vis_df["visible_wavelength_nm"], vis_df["SAM"], marker="o")
        ax.set_xlabel("Visible anchor wavelength (nm)")
        ax.set_ylabel("SAM (degree)")
        ax.set_title(f"{dataset.name}: visible anchor scan, NIR fixed at ~801 nm")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    if not nir_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(nir_df["nir_wavelength_nm"], nir_df["SAM"], marker="o")
        ax.set_xlabel("NIR anchor wavelength (nm)")
        ax.set_ylabel("SAM (degree)")
        ax.set_title(f"{dataset.name}: NIR anchor scan, visible fixed at ~549 nm")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# ANALYSIS 3: RGB / RGB+NIR / 549+801
# =============================================================================
def analysis_rgb_comparison(datasets_frames: Sequence[Tuple[PreparedDataset, pd.DataFrame]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    names = ("rgb_only", "rgb_plus_nir_801", "spanning_549_801")

    for dataset, frame in datasets_frames:
        selected = {}
        for name in names:
            hit = frame[frame["experiment"] == name]
            if not hit.empty:
                selected[name] = hit.iloc[0]
        if "rgb_only" not in selected:
            continue
        rgb_sam = float(selected["rgb_only"]["SAM"])
        for name in names:
            if name not in selected:
                continue
            row = selected[name]
            sam = float(row["SAM"])
            rows.append({
                "dataset": dataset.name,
                "experiment": name,
                "SAM": sam,
                "PSNR": float(row["PSNR"]),
                "ERGAS": float(row["ERGAS"]),
                "SAM_improvement_vs_RGB_absolute": rgb_sam - sam,
                "SAM_improvement_vs_RGB_percent": 100.0 * (rgb_sam - sam) / rgb_sam,
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_root / "rgb_comparison.csv", index=False)

    if not out.empty:
        pivot = out.pivot(index="experiment", columns="dataset", values="SAM")
        order = [x for x in names if x in pivot.index]
        pivot = pivot.loc[order]
        ax = pivot.plot(kind="bar", figsize=(8, 4.8))
        ax.set_ylabel("SAM (degree)")
        ax.set_xlabel("")
        ax.set_title("RGB and NIR-anchor comparison")
        ax.set_xticklabels(["RGB", "RGB + 801 nm", "549 + 801 nm"][: len(order)], rotation=0)
        ax.grid(axis="y", alpha=0.25)
        fig = ax.get_figure()
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# ANALYSIS 4: PER-WAVELENGTH ERROR PROFILES
# =============================================================================
def analysis_per_wavelength(dataset: PreparedDataset, frame: pd.DataFrame, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    labels = {
        "rgb_only": "RGB",
        "rgb_plus_nir_801": "RGB + 801 nm",
        "single_nir_801": "801 nm",
        "two_nir_801_935": "801 + 935 nm",
        "spanning_549_801": "549 + 801 nm",
    }

    curves: Dict[str, pd.DataFrame] = {}
    summary_rows = []
    visible = dataset.wavelengths_nm <= VISIBLE_MAX_NM

    for exp_name in PROFILE_EXPERIMENTS:
        hit = frame[frame["experiment"] == exp_name]
        if hit.empty:
            continue
        path = exp_dir(dataset, Experiment(exp_name, "", (), "")) / "per_band_metrics.csv"
        # exp_dir uses only the name, so the dummy Experiment is safe here.
        if not path.exists():
            print(f"[audit] missing per-band profile: {path}")
            continue
        prof = pd.read_csv(path)
        curves[exp_name] = prof
        rmse = prof["RMSE"].to_numpy(float)
        summary_rows.append({
            "dataset": dataset.name,
            "experiment": exp_name,
            "visible_mean_RMSE": float(np.mean(rmse[visible])),
            "NIR_mean_RMSE": float(np.mean(rmse[~visible])),
            "full_mean_RMSE": float(np.mean(rmse)),
        })

    pd.DataFrame(summary_rows).to_csv(out_root / "visible_vs_NIR_RMSE_summary.csv", index=False)

    if curves:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, prof in curves.items():
            ax.plot(prof["wavelength_nm"], prof["RMSE"], label=labels[name])
        ax.axvline(VISIBLE_MAX_NM, linestyle="--", linewidth=1)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{dataset.name}: per-wavelength reconstruction error")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# BONUS: 1-TO-4 BAND SPANNING CURVE (NO EXTRA TRAINING)
# =============================================================================
def analysis_band_count(dataset: PreparedDataset, frame: pd.DataFrame, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    names = [
        (1, "single_nir_801"),
        (2, "spanning_549_801"),
        (3, "spanning_461_549_801"),
        (4, "spanning_461_549_652_801"),
    ]
    rows = []
    for n_bands, name in names:
        hit = frame[frame["experiment"] == name]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append({
            "n_bands": n_bands,
            "experiment": name,
            "SAM": float(row["SAM"]),
            "PSNR": float(row["PSNR"]),
            "ERGAS": float(row["ERGAS"]),
        })
    out = pd.DataFrame(rows).sort_values("n_bands") if rows else pd.DataFrame()
    out.to_csv(out_root / "band_count_1_to_4.csv", index=False)
    if not out.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(out["n_bands"], out["SAM"], marker="o")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Number of HR narrow bands")
        ax.set_ylabel("SAM (degree)")
        ax.set_title(f"{dataset.name}: representative 1-to-4-band curve")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# COMPLETION AUDIT + ALL ANALYSES
# =============================================================================
def run_analysis(prepared: Sequence[PreparedDataset]) -> None:
    """
    Build three-seed robustness outputs.

    Expected:
      20 new rows = 5 configs x 2 new seeds x 2 datasets
      30 combined rows = 5 configs x 3 seeds x 2 datasets
    """
    analysis_root = OUTPUT_ROOT / "robustness_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    df = collect_three_seed_robustness(prepared)

    if df.empty:
        print("[robustness analysis] no completed results found")
        return

    keep_cols = [
        "dataset",
        "experiment",
        "seed",
        "SAM",
        "PSNR",
        "ERGAS",
        "best_epoch",
        "best_composite_loss",
        "runtime_seconds",
        "source_root",
        "result_path",
    ]
    available = [c for c in keep_cols if c in df.columns]
    raw = df[available].copy()

    raw["seed"] = pd.to_numeric(
        raw["seed"], errors="coerce"
    ).astype("Int64")

    for col in ("SAM", "PSNR", "ERGAS"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    config_order = {
        name: i for i, name in enumerate(ROBUSTNESS_CONFIGS)
    }
    dataset_order = {"Chikusei": 0, "ICVL": 1}

    raw["_dataset_order"] = raw["dataset"].map(dataset_order)
    raw["_config_order"] = raw["experiment"].map(config_order)
    raw = raw.sort_values(
        ["_dataset_order", "_config_order", "seed"]
    ).drop(columns=["_dataset_order", "_config_order"])

    raw.to_csv(
        analysis_root / "00_robustness_all_three_seeds.csv",
        index=False,
    )

    audit_rows = []
    for dataset_name in ("Chikusei", "ICVL"):
        for exp_name in ROBUSTNESS_CONFIGS:
            sub = raw[
                (raw["dataset"] == dataset_name)
                & (raw["experiment"] == exp_name)
            ]
            seeds_found = sorted(
                int(x) for x in sub["seed"].dropna().tolist()
            )
            audit_rows.append(
                {
                    "dataset": dataset_name,
                    "experiment": exp_name,
                    "n_completed": len(seeds_found),
                    "seeds_found": ";".join(map(str, seeds_found)),
                    "expected_seeds": "0;1;2",
                    "complete_three_seed_set": seeds_found == [0, 1, 2],
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        analysis_root / "01_robustness_completion_audit.csv",
        index=False,
    )

    summary = (
        raw.groupby(
            ["dataset", "experiment"],
            as_index=False,
            sort=False,
        )
        .agg(
            n=("seed", "count"),
            SAM_mean=("SAM", "mean"),
            SAM_std=("SAM", "std"),
            SAM_min=("SAM", "min"),
            SAM_max=("SAM", "max"),
            PSNR_mean=("PSNR", "mean"),
            PSNR_std=("PSNR", "std"),
            ERGAS_mean=("ERGAS", "mean"),
            ERGAS_std=("ERGAS", "std"),
        )
    )

    summary["_dataset_order"] = summary["dataset"].map(dataset_order)
    summary["_config_order"] = summary["experiment"].map(config_order)
    summary = summary.sort_values(
        ["_dataset_order", "_config_order"]
    ).drop(columns=["_dataset_order", "_config_order"])

    summary.to_csv(
        analysis_root / "02_robustness_mean_std.csv",
        index=False,
    )

    sam_table = summary[
        [
            "dataset",
            "experiment",
            "n",
            "SAM_mean",
            "SAM_std",
            "SAM_min",
            "SAM_max",
        ]
    ].copy()

    sam_table["SAM_mean_plus_minus_SD"] = sam_table.apply(
        lambda r: (
            f"{r['SAM_mean']:.3f} ± {r['SAM_std']:.3f}"
            if int(r["n"]) >= 2 and np.isfinite(r["SAM_std"])
            else f"{r['SAM_mean']:.3f}"
        ),
        axis=1,
    )

    sam_table.to_csv(
        analysis_root / "03_manuscript_SAM_mean_std.csv",
        index=False,
    )

    rank_rows = []
    for (dataset_name, seed), sub in raw.groupby(
        ["dataset", "seed"]
    ):
        ss = sub.sort_values("SAM").copy()
        ss["SAM_rank"] = np.arange(1, len(ss) + 1)
        for _, row in ss.iterrows():
            rank_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "experiment": row["experiment"],
                    "SAM": float(row["SAM"]),
                    "SAM_rank": int(row["SAM_rank"]),
                }
            )

    pd.DataFrame(rank_rows).to_csv(
        analysis_root / "04_seedwise_SAM_ranking.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("THREE-SEED ROBUSTNESS ANALYSIS")
    print("=" * 100)
    print(f"Combined completed rows: {len(raw)} / 30 expected")
    print(
        "All 10 dataset/config groups complete: "
        f"{bool(audit['complete_three_seed_set'].all())}"
    )
    print("\nSAM mean ± SD:")
    print(
        sam_table[
            ["dataset", "experiment", "n", "SAM_mean_plus_minus_SD"]
        ].to_string(index=False)
    )
    print(f"\nSaved analysis to: {analysis_root.resolve()}")


# =============================================================================
# RUN
# =============================================================================
print("=" * 90)
print("SPARSEHSR MULTI-SEED ROBUSTNESS: 5 CONFIGS x SEEDS 1,2 x 2 DATASETS")
print("=" * 90)
print(
    f"Checkpoint selection: minimum original composite loss, checked every epoch\n"
    f"Early stopping: centered abundance Gram-SAM, min decrease {SAM_MIN_DECREASE}, "
    f"patience {SAM_PATIENCE_EPOCHS} epochs\n"
    f"Metrics/visualization: every {EVAL_EVERY} epochs; figures shown with plt.show() only"
)
print(f"New output root : {OUTPUT_ROOT.resolve()}")
print(f"Seed-0 read root: {BASELINE_SEED0_ROOT.resolve()}")
print(f"New seeds       : {SEEDS_TO_RUN}")
print(f"Robust configs  : {ROBUSTNESS_CONFIGS}")

prepared_datasets: List[PreparedDataset] = []

if RUN_CHIKUSEI:
    prepared_datasets.append(prepare_chikusei())
if RUN_ICVL:
    prepared_datasets.append(prepare_icvl())

if RUN_TRAINING:
    for dataset in prepared_datasets:
        train_dataset(dataset)

if RUN_ANALYSIS:
    run_analysis(prepared_datasets)  












