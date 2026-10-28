





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HySure COMPLETE FAIR BASELINE for the full 16-configuration HSI super-resolution study.

Purpose
-------
Run HySure as the PRIMARY external baseline on exactly the same 40-band
full-range targets, LR-HSI degradation, normalization, spatial geometry, and all
16 HR-guide configurations used by the main study, then generate manuscript-ready
within-HySure and cross-method analysis tables/figures.

This is a PyTorch/GPU port of the update equations in the authors' official
HySure MATLAB implementation:
    https://github.com/alfaiate/HySure
    src/data_fusion.m

Reference:
    M. Simoes, J. Bioucas-Dias, L. B. Almeida, J. Chanussot,
    "A Convex Formulation for Hyperspectral Image Superresolution via
    Subspace-Based Regularization," IEEE TGRS, 53(6):3373-3388, 2015.

FAIRNESS PROTOCOL
-----------------
1) SAME target cubes as the final main study:
      Chikusei: 40 bands, 363-1018 nm, 1000x1000
      ICVL:     40 bands, 400-1000 nm, aligned 800x800 crop
2) SAME mutual-max normalization using the aligned 40-band cube and the newly
   prepared 31-band visible cube.
3) SAME LR-HSI generation as the main study:
      8x downsampling; 8x8 Gaussian kernel, sigma=0.5; scipy valid convolution;
      then [::8, ::8].
4) SAME HR inputs as the main study: the 14 narrowband configurations plus
   RGB-only and RGB+801 nm. Narrowband responses use exact 0/1 selection matrices.
   For RGB, the exact HR RGB produced from the rebuilt 31-band visible cube is used;
   its 40->RGB response is estimated only from observed LR-HSI and spatially
   degraded HR-RGB (nonnegative least squares), never from HR-HSI ground truth.
5) HySure is given the EXACT spectral selection matrix R and an EXACT spatial
   blur/downsampling operator B reproducing the simulated LR-HSI. This is a
   favorable/oracle-calibrated condition for HySure and therefore a strong
   baseline rather than an artificially handicapped one.
6) No HR-HSI GT is used by HySure for fitting, stopping, parameter selection, or
   model selection. GT is used only once after reconstruction for SAM/PSNR/
   ERGAS/RMSE evaluation.
7) HySure objective parameters are kept fixed across compared configurations:
      p=10, SVD basis, lambda_phi=5e-4, lambda_m=1, mu=0.05.
   FINAL version does not stop at an arbitrary small iteration count. It runs up
   to 20,000 iterations and stops only after repeated optimization-only convergence
   checks. Non-converged runs are explicitly marked and excluded from final tables.
8) Metrics are recomputed with the SAME definitions used by the final main
   study, not with a different HySure-specific metric implementation.

DEFAULT RUN SET
---------------
All 16 configurations from the final study are run on both Chikusei and ICVL:
2 RGB-related configurations plus all 14 narrowband configurations.

The script also writes the focused analyses needed for the manuscript:
  A. VIS+VIS vs NIR+NIR vs VIS+NIR two-band control
  B. visible-anchor scan with ~801 nm fixed
  C. NIR-anchor scan with ~549 nm fixed
  D. 1 -> 2 -> 3 -> 4 distributed-band progression
  E. full pure-narrowband ranking
  F. visible-vs-NIR regional RMSE
  G. cross-method tables vs main method and any completed fair EDIP-Net runs

The implementation uses torch.fft so the deterministic convex iterations can
run efficiently on the GPU; this changes the computing backend, not HySure's
objective or SALSA update equations. Double precision is used by default to stay
close to the numerical behavior of the original MATLAB implementation.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage, signal
from scipy.io import loadmat
from scipy.optimize import nnls


# =============================================================================
# USER SETTINGS
# =============================================================================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 0
SCALE = 8
VISIBLE_MAX_NM = 700.0

# HySure objective settings. p=10 is a typical low-dimensional subspace size in
# the HySure paper; the Pavia demo values are retained for lambda_phi/lambda_m/mu.
# IMPORTANT FIX: 200 iterations was demonstrably not converged on our scenes, so
# iteration count is no longer used as the stopping rule.
HYSURE_P = 10
HYSURE_BASIS = "SVD"
HYSURE_LAMBDA_PHI = 5e-4
HYSURE_LAMBDA_M = 1.0
HYSURE_MU = 0.05

HYSURE_MAX_ITERS = 20_000
HYSURE_MIN_ITERS = 1_000
HYSURE_CHECK_EVERY = 25
# These tolerances are applied to changes measured over CHECK_EVERY iterations.
# They are strict enough to rule out the clearly unfinished v2 ICVL trajectories,
# while avoiding an unnecessary requirement for machine-precision stationarity.
HYSURE_RELX_TOL = 5e-4
HYSURE_PRIMAL_TOL = 2e-4
HYSURE_OBJECTIVE_REL_TOL = 1e-4
HYSURE_STABLE_CHECKS = 4
PRINT_EVERY = HYSURE_CHECK_EVERY

# Original MATLAB implementation uses double precision.
TORCH_DTYPE = torch.float64

# Final fresh output root: never mix non-converged validation results with complete-baseline runs.
OUTPUT_ROOT = Path("hysure_COMPLETE_BASELINE_FINAL_all16")
ANALYSIS_ROOT = OUTPUT_ROOT / "analysis"
MAIN_STUDY_ROOT = Path("fullBandStudy_new31_both_all16_compositeLoss_GramSAMstop_seed0")
EDIP_ROOT = Path("edipnet_fair_narrowband_baseline")

RUN_CHIKUSEI = True
RUN_ICVL = True
RUN_HYSURE = True
RUN_ANALYSIS = True
SKIP_COMPLETED = True
FORCE_RERUN = False
MAKE_PLOTS = True
SAVE_RECONSTRUCTION = False   # Full cubes are large; metrics/profile are saved regardless.
WRITE_LIVE_SUMMARY = True     # Refresh after every completed configuration.
ANALYSIS_AFTER_EACH_DATASET = True
RUN_ONLY_FIRST_N = None       # e.g. 3 for a smoke run; None = all 16.

# Exact products used by the final main study.
CHIKUSEI_40_PATH = Path("Chikusei/Chikusei_1000_40bands.h5")
CHIKUSEI_31_PATH = Path("Chikusei/Chikusei_1000_visible31_new.h5")
CHIKUSEI_KEY = "chikusei"
ICVL_40_PATH = Path("ICVL_raw/ICVL_1020_1223_40band.mat")
ICVL_31_PATH = Path("ICVL_raw/ICVL_1020_1223_visible31_new.mat")

CHIKUSEI_WAVELENGTHS_NM = np.array(
    [363.00, 378.47, 399.10, 414.57, 430.05, 445.52, 460.99, 481.62, 497.09,
     512.57, 533.20, 548.67, 564.14, 579.61, 600.24, 615.72, 631.19, 651.82,
     667.29, 682.76, 698.24, 713.71, 734.34, 749.81, 765.28, 780.76, 801.39,
     816.86, 832.33, 847.80, 868.43, 883.91, 899.38, 914.85, 935.48, 950.95,
     966.43, 981.90, 1002.53, 1018.00], dtype=np.float32,
)

# Exact 31-sample RGB response used by the main study. Rows are B/G/R.
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

# All 14 pure-narrowband experiments from the final 16-run main design.
CONFIGS: Dict[str, Sequence[float]] = {
    "single_vis_461":                   [461.0],
    "single_vis_549":                   [549.0],
    "single_vis_652":                   [652.0],
    "single_nir_801":                   [801.39],
    "single_nir_935":                   [935.0],
    "two_vis_461_549":                  [461.0, 549.0],
    "two_nir_801_935":                  [801.39, 935.0],
    "spanning_461_801":                 [461.0, 801.39],
    "spanning_549_801":                 [549.0, 801.39],
    "spanning_652_801":                 [652.0, 801.39],
    "spanning_549_714":                 [549.0, 714.0],
    "spanning_549_935":                 [549.0, 935.0],
    "spanning_461_549_801":             [461.0, 549.0, 801.39],
    "spanning_461_549_652_801":         [461.0, 549.0, 652.0, 801.39],
}
# Full 16-configuration design used by the main study. The first three pure
# narrowband runs are the decisive equal-count controls, followed by RGB-related,
# band-count, anchor-scan, and remaining single-band runs.
PAIR_REGION_CONFIGS = (
    "two_vis_461_549",
    "two_nir_801_935",
    "spanning_549_801",
)
RGB_CONFIGS = ("rgb_only", "rgb_plus_nir_801")

ALL_16_CONFIGS = (
    "two_vis_461_549",
    "two_nir_801_935",
    "spanning_549_801",
    "rgb_only",
    "rgb_plus_nir_801",
    "single_nir_801",
    "spanning_461_549_801",
    "spanning_461_549_652_801",
    "spanning_461_801",
    "spanning_652_801",
    "spanning_549_714",
    "spanning_549_935",
    "single_vis_461",
    "single_vis_549",
    "single_vis_652",
    "single_nir_935",
)
VALID_CONFIGS = set(CONFIGS) | set(RGB_CONFIGS)
RUN_CONFIGS = ALL_16_CONFIGS

# Optional smaller subset matching the broader EDIP-Net analysis if ever needed.
CORE_NECESSARY_CONFIGS = (
    "single_nir_801",
    "two_vis_461_549",
    "two_nir_801_935",
    "spanning_461_801",
    "spanning_549_801",
    "spanning_652_801",
    "spanning_549_714",
    "spanning_549_935",
    "spanning_461_549_801",
    "spanning_461_549_652_801",
)
VISIBLE_ANCHOR_CONFIGS = (
    "spanning_461_801", "spanning_549_801", "spanning_652_801"
)
NIR_ANCHOR_CONFIGS = (
    "spanning_549_714", "spanning_549_801", "spanning_549_935"
)
BAND_COUNT_CONFIGS = (
    "single_nir_801",
    "spanning_549_801",
    "spanning_461_549_801",
    "spanning_461_549_652_801",
)
SINGLE_CONFIGS = (
    "single_vis_461", "single_vis_549", "single_vis_652",
    "single_nir_801", "single_nir_935",
)


# =============================================================================
# DATA
# =============================================================================
@dataclass
class PreparedDataset:
    name: str
    hsi_hwc: np.ndarray
    visible31_hwc: np.ndarray
    rgb_hwc: np.ndarray
    wavelengths_nm: np.ndarray
    common_max: float
    raw_full_max: float
    raw_visible_max: float
    rgb_response_40: np.ndarray
    rgb_response_lr_fit_residual: float


def reshape_cube(array: np.ndarray, bands: int, expected_hw: Tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[-1] != bands:
            raise ValueError(f"Expected flattened array with {bands} bands, got {array.shape}")
        if expected_hw is None:
            raise ValueError("expected_hw required for flattened cube")
        h, w = expected_hw
        if h * w != array.shape[0]:
            raise ValueError(f"Cannot reshape {array.shape} to {(h,w,bands)}")
        return array.reshape(h, w, bands)
    if array.ndim != 3:
        raise ValueError(f"Expected 2D/3D cube, got {array.shape}")
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
    """Apply a spectral response matrix of shape C_in x C_out to an HxWxC cube."""
    h, w, c = img.shape
    if spectral_matrix.shape[0] != c:
        raise ValueError(f"Spectral matrix {spectral_matrix.shape} incompatible with image {img.shape}")
    out = img.reshape(-1, c).astype(np.float64) @ spectral_matrix.astype(np.float64)
    return out.reshape(h, w, spectral_matrix.shape[1]).astype(np.float32)


def estimate_rgb_response_from_observations(lr_hsi_hwc: np.ndarray, hr_rgb_hwc: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate the 40->RGB response using ONLY observed inputs.

    The exact RGB guide in this study comes from the separately prepared 31-band
    visible product, so there is no exact 0/1 40-band selection matrix for RGB.
    We therefore spatially degrade the observed HR RGB with the same known B/M
    operator, pair it with the observed LR-HSI, and fit one nonnegative response
    row per RGB channel by NNLS. No HR-HSI ground truth enters this fit.
    """
    rgb_lr = generate_lr_hsi_exact(hr_rgb_hwc, SCALE)
    X = lr_hsi_hwc.reshape(-1, lr_hsi_hwc.shape[2]).astype(np.float64)
    Y = rgb_lr.reshape(-1, rgb_lr.shape[2]).astype(np.float64)
    rows = []
    for j in range(Y.shape[1]):
        r, _ = nnls(X, Y[:, j], maxiter=20 * X.shape[1])
        rows.append(r)
    R = np.stack(rows, axis=0).astype(np.float32)
    pred = X @ R.T.astype(np.float64)
    rel = float(np.linalg.norm(pred - Y) / max(np.linalg.norm(Y), 1e-12))
    return R, rel


def prepare_chikusei() -> PreparedDataset:
    with h5py.File(CHIKUSEI_40_PATH, "r") as f:
        full_raw = reshape_cube(np.asarray(f[CHIKUSEI_KEY]), 40, (1000, 1000))
        if "selected_wavelengths_nm" in f:
            wavelengths = np.asarray(f["selected_wavelengths_nm"], dtype=np.float32).ravel()
        else:
            wavelengths = CHIKUSEI_WAVELENGTHS_NM.copy()
    with h5py.File(CHIKUSEI_31_PATH, "r") as f:
        visible_raw = reshape_cube(np.asarray(f[CHIKUSEI_KEY]), 31, (1000, 1000))

    full_max = float(np.max(full_raw))
    visible_max = float(np.max(visible_raw))
    common_max = max(full_max, visible_max)
    full_norm = (full_raw / common_max).astype(np.float32, copy=False)
    visible_norm = (visible_raw / common_max).astype(np.float64, copy=False)
    rgb = generate_msi(visible_norm, RGB_SRF_BGR.T)
    lr_for_response = generate_lr_hsi_exact(full_norm, SCALE)
    rgb_R, rgb_fit = estimate_rgb_response_from_observations(lr_for_response, rgb)
    print(f"[Chikusei RGB response] observed LR fit residual={rgb_fit:.3e}")
    return PreparedDataset(
        "Chikusei", full_norm, visible_norm, rgb, wavelengths, common_max,
        full_max, visible_max, rgb_R, rgb_fit
    )


def prepare_icvl() -> PreparedDataset:
    p40 = loadmat(ICVL_40_PATH)
    p31 = loadmat(ICVL_31_PATH)
    full_original = np.asarray(p40["rad"], dtype=np.float32)
    visible_original = np.asarray(p31["rad"], dtype=np.float32)
    wavelengths = np.asarray(p40["wavelengths"], dtype=np.float32).ravel()

    # Same verified geometry for both 40-band and rebuilt 31-band products.
    full_aligned = ndimage.rotate(full_original, 90)
    full_aligned = full_aligned[:, ::-1, :]
    full_raw = full_aligned[392:1192, 100:900, :]

    visible_aligned = ndimage.rotate(visible_original, 90)
    visible_aligned = visible_aligned[:, ::-1, :]
    visible_raw = visible_aligned[392:1192, 100:900, :]

    if full_raw.shape != (800, 800, 40):
        raise ValueError(f"Unexpected ICVL 40-band crop: {full_raw.shape}")
    if visible_raw.shape != (800, 800, 31):
        raise ValueError(f"Unexpected ICVL 31-band crop: {visible_raw.shape}")

    full_max = float(np.max(full_raw))
    visible_max = float(np.max(visible_raw))
    common_max = max(full_max, visible_max)
    full_norm = (full_raw / common_max).astype(np.float32, copy=False)
    visible_norm = (visible_raw / common_max).astype(np.float64, copy=False)
    rgb = generate_msi(visible_norm, RGB_SRF_BGR.T)
    lr_for_response = generate_lr_hsi_exact(full_norm, SCALE)
    rgb_R, rgb_fit = estimate_rgb_response_from_observations(lr_for_response, rgb)
    print(f"[ICVL RGB response] observed LR fit residual={rgb_fit:.3e}")
    return PreparedDataset(
        "ICVL", full_norm, visible_norm, rgb, wavelengths, common_max,
        full_max, visible_max, rgb_R, rgb_fit
    )


def matlab_style_gaussian_2d(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    k = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma))
    k[k < np.finfo(k.dtype).eps * k.max()] = 0
    k /= k.sum()
    return k


def generate_lr_hsi_exact(img: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Exact main-study degradation: valid Gaussian convolution + stride."""
    kernel = matlab_style_gaussian_2d((scale, scale), sigma=0.5)
    h, w, c = img.shape
    out = np.zeros((h // scale, w // scale, c), dtype=np.float32)
    for b in range(c):
        filtered = signal.convolve2d(img[:, :, b], kernel, mode="valid")
        out[:, :, b] = filtered[::scale, ::scale]
    return out


def nearest_idx(wavelengths: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wavelengths) - float(nm))))


def selected_indices(wavelengths: np.ndarray, targets: Sequence[float]) -> List[int]:
    idx = [nearest_idx(wavelengths, x) for x in targets]
    if len(set(idx)) != len(idx):
        raise ValueError(f"Wavelength targets collapsed to duplicate bands: {targets} -> {idx}")
    return idx


def selection_matrix(n_hsi_bands: int, indices: Sequence[int]) -> np.ndarray:
    """HySure R matrix, shape n_selected x n_hsi_bands."""
    R = np.zeros((len(indices), n_hsi_bands), dtype=np.float32)
    for row, col in enumerate(indices):
        R[row, int(col)] = 1.0
    return R


# =============================================================================
# EXACT FFT BLUR OPERATOR FOR THE MAIN-STUDY VALID-CONVOLUTION SAMPLES
# =============================================================================
def make_exact_circular_blur_kernel(h: int, w: int, scale: int = SCALE) -> np.ndarray:
    """Construct B so circular convolution sampled at [::scale,::scale] exactly
    equals scipy.signal.convolve2d(..., kernel, 'valid')[::scale,::scale].

    For desired y[i,j] = sum_{a,b} x[i+a,j+b] k[a,b], circular convolution
    y = x (*) B is obtained with B[-a mod H, -b mod W] = k[a,b]. Since sampled
    windows end before the image boundary, no wraparound enters the samples.
    """
    k = matlab_style_gaussian_2d((scale, scale), sigma=0.5)
    B = np.zeros((h, w), dtype=np.float32)
    for a in range(scale):
        for b in range(scale):
            B[(-a) % h, (-b) % w] = np.float32(k[a, b])
    return B


def torch_convc(x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Circular convolution of each leading channel/map using FFT."""
    return torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1)) * F, dim=(-2, -1)).real


def verify_forward_model(gt_hwc: np.ndarray, lr_hwc: np.ndarray, B: np.ndarray, device: torch.device) -> float:
    """Numerically prove that HySure's B+mask corresponds to the main LR data."""
    with torch.no_grad():
        x = torch.from_numpy(np.transpose(gt_hwc, (2, 0, 1))).to(device=device, dtype=TORCH_DTYPE)
        FB = torch.fft.fft2(torch.from_numpy(B).to(device=device, dtype=TORCH_DTYPE))
        y = torch_convc(x, FB)[:, ::SCALE, ::SCALE]
        target = torch.from_numpy(np.transpose(lr_hwc, (2, 0, 1))).to(device=device, dtype=TORCH_DTYPE)
        err = float(torch.max(torch.abs(y - target)).item())
    return err


# =============================================================================
# HySure SALSA -- faithful GPU port of official src/data_fusion.m
# =============================================================================
def estimate_svd_basis(lr_hwc: np.ndarray, p: int, device: torch.device) -> torch.Tensor:
    """Top-p left singular vectors of the observed LR-HSI; no GT."""
    c = lr_hwc.shape[2]
    p_eff = min(int(p), c, lr_hwc.shape[0] * lr_hwc.shape[1])
    Y = torch.from_numpy(lr_hwc.reshape(-1, c).T.copy()).to(device=device, dtype=TORCH_DTYPE)
    U, _, _ = torch.linalg.svd(Y, full_matrices=False)
    return U[:, :p_eff].contiguous()


def solve_maps(A: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = rhs for every spatial position."""
    p, h, w = rhs.shape
    flat = rhs.reshape(p, -1)
    out = torch.linalg.solve(A, flat)
    return out.reshape(p, h, w)


def vector_soft_col_iso(nu_h: torch.Tensor, nu_v: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Column-wise isotropic vector soft threshold used by HySure VTV."""
    mag = torch.sqrt(torch.sum(nu_h * nu_h + nu_v * nu_v, dim=0)).clamp_min(1e-12)
    shrink = torch.clamp(1.0 - float(tau) / mag, min=0.0)
    return nu_h * shrink.unsqueeze(0), nu_v * shrink.unsqueeze(0)


@torch.no_grad()
def hysure_fuse(
    lr_hsi_hwc: np.ndarray,
    hr_guide_hwc: np.ndarray,
    R_np: np.ndarray,
    B_np: np.ndarray,
    device: torch.device,
    p: int = HYSURE_P,
    lambda_phi: float = HYSURE_LAMBDA_PHI,
    lambda_m: float = HYSURE_LAMBDA_M,
    mu: float = HYSURE_MU,
    max_iters: int = HYSURE_MAX_ITERS,
    min_iters: int = HYSURE_MIN_ITERS,
    check_every: int = HYSURE_CHECK_EVERY,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """HySure SALSA/ADMM iterations using an SVD subspace.

    FIXES relative to v1:
      * all optimization tensors stay in float64;
      * no fixed 200-iteration termination;
      * convergence is checked only from optimization/observed quantities, never GT;
      * stopping requires stable X, small split-primal residuals, and stable objective
        for several consecutive checks.
    """
    h, w, n_ms = hr_guide_hwc.shape
    c = lr_hsi_hwc.shape[2]
    p_eff = min(int(p), c)

    E = estimate_svd_basis(lr_hsi_hwc, p_eff, device)       # C x p
    R = torch.from_numpy(R_np).to(device=device, dtype=TORCH_DTYPE)
    RE = R @ E                                               # n x p

    Yh = torch.from_numpy(np.transpose(lr_hsi_hwc, (2, 0, 1)).copy()).to(device=device, dtype=TORCH_DTYPE)
    Ym = torch.from_numpy(np.transpose(hr_guide_hwc, (2, 0, 1)).copy()).to(device=device, dtype=TORCH_DTYPE)

    dh = torch.zeros((h, w), device=device, dtype=TORCH_DTYPE)
    dh[0, 0] = 1.0; dh[0, -1] = -1.0
    dv = torch.zeros((h, w), device=device, dtype=TORCH_DTYPE)
    dv[0, 0] = 1.0; dv[-1, 0] = -1.0

    B = torch.from_numpy(B_np).to(device=device, dtype=TORCH_DTYPE)
    FB = torch.fft.fft2(B)
    FBC = torch.conj(FB)
    FDH = torch.fft.fft2(dh)
    FDHC = torch.conj(FDH)
    FDV = torch.fft.fft2(dv)
    FDVC = torch.conj(FDV)
    denom = torch.abs(FB) ** 2 + torch.abs(FDH) ** 2 + torch.abs(FDV) ** 2 + 1.0
    IBD_B = FBC / denom
    IBD_II = 1.0 / denom
    IBD_DH = FDHC / denom
    IBD_DV = FDVC / denom

    mask = torch.zeros((h, w), device=device, dtype=TORCH_DTYPE)
    mask[::SCALE, ::SCALE] = 1.0

    Yh_up = torch.zeros((c, h, w), device=device, dtype=TORCH_DTYPE)
    Yh_up[:, ::SCALE, ::SCALE] = Yh

    eye = torch.eye(p_eff, device=device, dtype=TORCH_DTYPE)
    IE = E.T @ E + float(mu) * eye
    yyh = torch.einsum("cp,chw->phw", E, Yh_up)
    IRE = float(lambda_m) * (RE.T @ RE) + float(mu) * eye
    yym = torch.einsum("np,nhw->phw", RE, Ym)

    # FIX: v1 initialized these as float32 even though the solver is otherwise float64.
    X = torch.zeros((p_eff, h, w), device=device, dtype=TORCH_DTYPE)
    V1 = torch.zeros_like(X); D1 = torch.zeros_like(X)
    V2 = torch.zeros_like(X); D2 = torch.zeros_like(X)
    V3 = torch.zeros_like(X); D3 = torch.zeros_like(X)
    V4 = torch.zeros_like(X); D4 = torch.zeros_like(X)

    history: List[Dict[str, float]] = []
    last_check_X = None
    last_objective = None
    stable_count = 0
    converged = False
    stopping_reason = "max_iters"
    t0 = time.time()

    def relnorm(a: torch.Tensor, b: torch.Tensor) -> float:
        den = torch.maximum(torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)).clamp_min(1e-12)
        return float((torch.linalg.vector_norm(a - b) / den).item())

    iterations_run = 0
    for i in range(1, int(max_iters) + 1):
        iterations_run = i
        # X update.
        X = (
            torch_convc(V1 + D1, IBD_B)
            + torch_convc(V2 + D2, IBD_II)
            + torch_convc(V3 + D3, IBD_DH)
            + torch_convc(V4 + D4, IBD_DV)
        )

        # V1: LR-HSI data term at the actually sampled locations.
        XB = torch_convc(X, FB)
        NU1 = XB - D1
        rhs1 = yyh + float(mu) * NU1
        V1_data = solve_maps(IE, rhs1)
        V1 = V1_data * mask.unsqueeze(0) + NU1 * (1.0 - mask).unsqueeze(0)

        # V2: HR guide/narrowband data term.
        NU2 = X - D2
        rhs2 = float(lambda_m) * yym + float(mu) * NU2
        V2 = solve_maps(IRE, rhs2)

        # V3/V4: isotropic vector-TV proximal update.
        XDh = torch_convc(X, FDH)
        XDv = torch_convc(X, FDV)
        NU3 = XDh - D3
        NU4 = XDv - D4
        V3, V4 = vector_soft_col_iso(NU3, NU4, float(lambda_phi) / float(mu))

        # Scaled dual updates (same sign convention as v1/official port).
        D1 = -NU1 + V1
        D2 = -NU2 + V2
        D3 = -NU3 + V3
        D4 = -NU4 + V4

        if i == 1 or i % int(check_every) == 0 or i == int(max_iters):
            # Observed-domain predictions (no GT).
            pred_lr = torch.einsum("cp,phw->chw", E, XB[:, ::SCALE, ::SCALE])
            pred_hr = torch.einsum("np,phw->nhw", RE, X)
            rel_lr = float((torch.linalg.vector_norm(pred_lr - Yh) /
                            torch.linalg.vector_norm(Yh).clamp_min(1e-12)).item())
            rel_hr = float((torch.linalg.vector_norm(pred_hr - Ym) /
                            torch.linalg.vector_norm(Ym).clamp_min(1e-12)).item())

            # ADMM split-primal consistency. These must approach zero if SALSA converges.
            pr1 = relnorm(XB, V1)
            pr2 = relnorm(X, V2)
            pr3 = relnorm(XDh, V3)
            pr4 = relnorm(XDv, V4)
            max_primal = max(pr1, pr2, pr3, pr4)

            if last_check_X is None:
                rel_change = float("nan")
            else:
                rel_change = float((torch.linalg.vector_norm(X - last_check_X) /
                                    torch.linalg.vector_norm(last_check_X).clamp_min(1e-12)).item())
            last_check_X = X.clone()

            # True HySure objective (up to the paper's conventional 1/2 constants).
            data_h = 0.5 * torch.sum((pred_lr - Yh) ** 2)
            data_m = 0.5 * float(lambda_m) * torch.sum((pred_hr - Ym) ** 2)
            tv_mag = torch.sqrt(torch.sum(XDh * XDh + XDv * XDv, dim=0)).sum()
            objective = float((data_h + data_m + float(lambda_phi) * tv_mag).item())
            if last_objective is None:
                rel_objective_change = float("nan")
            else:
                rel_objective_change = abs(objective - last_objective) / max(abs(last_objective), 1e-12)
            last_objective = objective

            if (
                i >= int(min_iters)
                and np.isfinite(rel_change)
                and np.isfinite(rel_objective_change)
                and rel_change < HYSURE_RELX_TOL
                and max_primal < HYSURE_PRIMAL_TOL
                and rel_objective_change < HYSURE_OBJECTIVE_REL_TOL
            ):
                stable_count += 1
            else:
                stable_count = 0

            row = {
                "iteration": i,
                "relative_lr_hsi_residual": rel_lr,
                "relative_hr_guide_residual": rel_hr,
                "relative_X_change_since_last_check": rel_change,
                "objective": objective,
                "relative_objective_change": rel_objective_change,
                "primal_B": pr1,
                "primal_identity": pr2,
                "primal_Dh": pr3,
                "primal_Dv": pr4,
                "max_primal_residual": max_primal,
                "stable_checks": stable_count,
                "seconds": time.time() - t0,
            }
            history.append(row)
            print(
                f"[HySure] iter={i:04d}/{max_iters} relLR={rel_lr:.3e} relHR={rel_hr:.3e} "
                f"relX={rel_change:.3e} prim={max_primal:.3e} objΔ={rel_objective_change:.3e} "
                f"stable={stable_count}/{HYSURE_STABLE_CHECKS}"
            )

            if stable_count >= int(HYSURE_STABLE_CHECKS):
                converged = True
                stopping_reason = "unsupervised_ADMM_convergence"
                print(f"[HySure] converged at iteration {i} using optimization-only criteria.")
                break

    # Reconstruct from the actually retained X.
    Z_chw = torch.einsum("cp,phw->chw", E, X)
    reconstruction = np.transpose(Z_chw.detach().cpu().numpy(), (1, 2, 0)).astype(np.float32, copy=False)

    last = history[-1] if history else {}
    if not converged:
        print(
            "[HySure warning] maximum iterations reached before all convergence criteria were met. "
            "Do not use this run as final manuscript evidence until the history is inspected."
        )

    diagnostics = {
        "p_effective": int(p_eff),
        "iterations_run": int(iterations_run),
        "max_iterations": int(max_iters),
        "min_iterations": int(min_iters),
        "check_every": int(check_every),
        "lambda_phi": float(lambda_phi),
        "lambda_m": float(lambda_m),
        "mu": float(mu),
        "basis": HYSURE_BASIS,
        "runtime_seconds": time.time() - t0,
        "history": history,
        "converged": bool(converged),
        "stopping_reason": stopping_reason,
        "final_relative_X_change": last.get("relative_X_change_since_last_check"),
        "final_relative_objective_change": last.get("relative_objective_change"),
        "final_max_primal_residual": last.get("max_primal_residual"),
        "torch_dtype": str(TORCH_DTYPE),
    }

    del E, R, RE, Yh, Ym, FB, FBC, FDH, FDHC, FDV, FDVC, X, V1, D1, V2, D2, V3, D3, V4, D4
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reconstruction, diagnostics


# =============================================================================
# METRICS -- SAME DEFINITIONS AS FINAL MAIN STUDY
# =============================================================================
def main_study_metrics(gt: np.ndarray, pred: np.ndarray, wavelengths: np.ndarray):
    g = gt.reshape(-1, gt.shape[2]).astype(np.float64)
    p = pred.reshape(-1, pred.shape[2]).astype(np.float64)

    numerator = np.sum(g * p, axis=1)
    denom = np.sqrt(np.sum(g ** 2, axis=1)) * np.sqrt(np.sum(p ** 2, axis=1))
    cosv = np.clip(numerator / np.maximum(denom, 1e-30), -1.0 + 1e-8, 1.0 - 1e-8)
    sam = float(np.mean(np.degrees(np.arccos(cosv))))

    msr = np.mean((g - p) ** 2, axis=1)
    max2 = float(np.max(g ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        psnr_all = 10.0 * np.log10(max2 / np.maximum(msr, 1e-30))
    psnr = float(np.mean(psnr_all))

    means = np.mean(g, axis=0)
    mse_band = np.mean((g - p) ** 2, axis=0)
    ergas = float(100.0 * np.sqrt(np.mean(mse_band / (means ** 2 + 1e-8))))

    rmse_band = np.sqrt(mse_band)
    wl = np.asarray(wavelengths, dtype=np.float64)
    vis = wl <= VISIBLE_MAX_NM
    nir = wl > VISIBLE_MAX_NM
    rmse_full = float(np.sqrt(np.mean((g - p) ** 2)))
    rmse_vis = float(np.sqrt(np.mean((g[:, vis] - p[:, vis]) ** 2)))
    rmse_nir = float(np.sqrt(np.mean((g[:, nir] - p[:, nir]) ** 2)))

    return {
        "SAM": sam,
        "PSNR": psnr,
        "ERGAS": ergas,
        "RMSE_full": rmse_full,
        "RMSE_visible": rmse_vis,
        "RMSE_NIR": rmse_nir,
    }, rmse_band


def observed_consistency(
    recon_hwc: np.ndarray,
    lr_hwc: np.ndarray,
    guide_hwc: np.ndarray,
    R_np: np.ndarray,
) -> Dict[str, float]:
    pred_lr = generate_lr_hsi_exact(recon_hwc, SCALE)
    pred_guide = generate_msi(recon_hwc, np.asarray(R_np, dtype=np.float64).T)
    rel_lr = float(np.linalg.norm(pred_lr - lr_hwc) / max(np.linalg.norm(lr_hwc), 1e-12))
    rel_hr = float(np.linalg.norm(pred_guide - guide_hwc) / max(np.linalg.norm(guide_hwc), 1e-12))
    return {
        "relative_lr_hsi_residual": rel_lr,
        "relative_hr_guide_residual": rel_hr,
    }


# =============================================================================
# RUNNER
# =============================================================================
def run_one(dataset: PreparedDataset, config_name: str) -> Dict[str, object]:
    device = torch.device(DEVICE)
    gt = dataset.hsi_hwc
    wavelengths = dataset.wavelengths_nm
    lr = generate_lr_hsi_exact(gt, SCALE)

    if config_name == "rgb_only":
        idx = []
        selected_nm = []
        guide = dataset.rgb_hwc.astype(np.float32, copy=True)
        R = dataset.rgb_response_40.astype(np.float32, copy=True)
        guide_labels = ["RGB_B", "RGB_G", "RGB_R"]
        response_mode = "observed_only_NNLS_40_to_RGB"
    elif config_name == "rgb_plus_nir_801":
        idx = selected_indices(wavelengths, [801.39])
        selected_nm = [float(wavelengths[i]) for i in idx]
        narrow = gt[..., idx].astype(np.float32, copy=True)
        guide = np.concatenate([dataset.rgb_hwc.astype(np.float32, copy=False), narrow], axis=2)
        R = np.vstack([dataset.rgb_response_40, selection_matrix(40, idx)]).astype(np.float32)
        guide_labels = ["RGB_B", "RGB_G", "RGB_R", f"{selected_nm[0]:.3f}nm"]
        response_mode = "observed_only_NNLS_RGB_plus_exact_801_selection"
    else:
        idx = selected_indices(wavelengths, CONFIGS[config_name])
        selected_nm = [float(wavelengths[i]) for i in idx]
        guide = gt[..., idx].astype(np.float32, copy=True)
        R = selection_matrix(40, idx)
        guide_labels = [f"{x:.3f}nm" for x in selected_nm]
        response_mode = "known_exact_0_1_band_selection"

    B = make_exact_circular_blur_kernel(gt.shape[0], gt.shape[1], SCALE)

    run_dir = OUTPUT_ROOT / dataset.name / config_name / f"seed_{SEED}"
    result_path = run_dir / "result.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    if FORCE_RERUN and result_path.exists():
        result_path.unlink()
    if SKIP_COMPLETED and result_path.exists() and not FORCE_RERUN:
        try:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if old.get("status") == "complete":
                print(f"[skip] {dataset.name}/{config_name}: already complete")
                return old
        except Exception:
            pass

    print("\n" + "=" * 100)
    print(
        f"HySure FAIR | {dataset.name} | {config_name} | n={guide.shape[2]} | "
        f"guide={guide_labels}"
    )
    print(
        f"normalization: full_max={dataset.raw_full_max:.10g}, "
        f"visible31_max={dataset.raw_visible_max:.10g}, common_max={dataset.common_max:.10g}"
    )
    print("=" * 100)

    forward_err = verify_forward_model(gt, lr, B, device)
    print(f"[protocol] max|HySure_BM(GT) - exact_main_LR| = {forward_err:.3e}")
    if forward_err > 2e-5:
        raise RuntimeError(
            f"HySure forward operator does not reproduce main-study LR-HSI: max error {forward_err}"
        )

    t0 = time.time()
    recon, diag = hysure_fuse(lr, guide, R, B, device)
    runtime = time.time() - t0

    if not np.all(np.isfinite(recon)):
        raise RuntimeError(f"Non-finite HySure reconstruction for {dataset.name}/{config_name}")
    recon_min = float(np.min(recon))
    recon_max = float(np.max(recon))
    recon_negative_fraction = float(np.mean(recon < 0.0))
    recon_above_one_fraction = float(np.mean(recon > 1.0))
    print(
        f"[sanity] reconstruction range=({recon_min:.6g}, {recon_max:.6g}); "
        f"negative={100*recon_negative_fraction:.3f}%; >1={100*recon_above_one_fraction:.3f}%"
    )

    # Ground truth is used for the first time here: final post-hoc evaluation only.
    metrics, rmse_band = main_study_metrics(gt, recon, wavelengths)
    obs = observed_consistency(recon, lr, guide, R)

    pd.DataFrame({
        "band_index": np.arange(40, dtype=int),
        "wavelength_nm": wavelengths,
        "RMSE": rmse_band,
    }).to_csv(run_dir / "per_band_metrics.csv", index=False)
    pd.DataFrame(diag["history"]).to_csv(run_dir / "hysure_history.csv", index=False)

    if SAVE_RECONSTRUCTION:
        np.savez_compressed(run_dir / "reconstruction.npz", reconstruction=recon)

    result_status = "complete" if bool(diag.get("converged")) else "nonconverged"
    result = {
        "status": result_status,
        "method": "HySure-SVD",
        "protocol": "fair_known_B_exact_narrowband_R_observed_only_RGB_R_no_GT_selection",
        "dataset": dataset.name,
        "configuration": config_name,
        "seed": SEED,
        "n_channels": int(guide.shape[2]),
        "band_indices": [int(i) for i in idx],
        "wavelengths_nm": selected_nm,
        "guide_channels": guide_labels,
        "spectral_response_mode": response_mode,
        "rgb_response_lr_fit_residual": (dataset.rgb_response_lr_fit_residual if config_name in RGB_CONFIGS else None),
        "SAM": metrics["SAM"],
        "PSNR": metrics["PSNR"],
        "ERGAS": metrics["ERGAS"],
        "RMSE_full": metrics["RMSE_full"],
        "RMSE_visible": metrics["RMSE_visible"],
        "RMSE_NIR": metrics["RMSE_NIR"],
        "relative_lr_hsi_residual": obs["relative_lr_hsi_residual"],
        "relative_hr_guide_residual": obs["relative_hr_guide_residual"],
        "runtime_seconds": runtime,
        "common_max": dataset.common_max,
        "raw_full_max": dataset.raw_full_max,
        "raw_visible_max": dataset.raw_visible_max,
        "scale_factor": SCALE,
        "forward_operator_max_abs_error_vs_main_LR": forward_err,
        "reconstruction_min": recon_min,
        "reconstruction_max": recon_max,
        "reconstruction_negative_fraction": recon_negative_fraction,
        "reconstruction_above_one_fraction": recon_above_one_fraction,
        "hysure": {
            "basis": HYSURE_BASIS,
            "p": int(diag["p_effective"]),
            "lambda_phi": HYSURE_LAMBDA_PHI,
            "lambda_m": HYSURE_LAMBDA_M,
            "mu": HYSURE_MU,
            "iterations_run": diag.get("iterations_run"),
            "max_iterations": HYSURE_MAX_ITERS,
            "converged": diag.get("converged"),
            "stopping_reason": diag.get("stopping_reason"),
            "final_relative_X_change": diag.get("final_relative_X_change"),
            "final_relative_objective_change": diag.get("final_relative_objective_change"),
            "final_max_primal_residual": diag.get("final_max_primal_residual"),
            "torch_dtype": diag.get("torch_dtype"),
            "spectral_response_R": response_mode,
            "spatial_response_B": "known exact circular representation of main-study sigma=0.5 valid-convolution samples",
            "gt_used_for_reconstruction_or_selection": False,
            "gt_used_for_final_evaluation_only": True,
        },
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"[{result_status}] {dataset.name}/{config_name}: "
        f"SAM={metrics['SAM']:.6f}, PSNR={metrics['PSNR']:.6f}, "
        f"ERGAS={metrics['ERGAS']:.6f}, RMSE={metrics['RMSE_full']:.6f}, "
        f"relLR={obs['relative_lr_hsi_residual']:.3e}, "
        f"relHR={obs['relative_hr_guide_residual']:.3e}, "
        f"runtime={runtime/60:.2f} min"
    )

    del gt, guide, lr, R, B, recon
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_live_summary() -> None:
    """Write partial results immediately so the study can be inspected mid-run."""
    if not WRITE_LIVE_SUMMARY:
        return
    rows = []
    for dataset_name in ("Chikusei", "ICVL"):
        for cfg in RUN_CONFIGS:
            path = OUTPUT_ROOT / dataset_name / cfg / f"seed_{SEED}" / "result.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("status") != "complete":
                continue
            rows.append({
                "method": payload.get("method", "HySure-SVD"),
                "dataset": dataset_name,
                "configuration": cfg,
                "n_channels": payload.get("n_channels"),
                "wavelengths_nm": payload.get("wavelengths_nm"),
                "SAM": payload.get("SAM"),
                "PSNR": payload.get("PSNR"),
                "ERGAS": payload.get("ERGAS"),
                "RMSE_full": payload.get("RMSE_full"),
                "RMSE_visible": payload.get("RMSE_visible"),
                "RMSE_NIR": payload.get("RMSE_NIR"),
                "runtime_seconds": payload.get("runtime_seconds"),
                "iterations_run": payload.get("hysure", {}).get("iterations_run"),
                "converged": payload.get("hysure", {}).get("converged"),
                "final_relative_X_change": payload.get("hysure", {}).get("final_relative_X_change"),
            })
    if rows:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_ROOT / "LIVE_HySure_results.csv", index=False)
        ranked = df.sort_values(["dataset", "SAM"])
        ranked.to_csv(OUTPUT_ROOT / "LIVE_HySure_ranked_by_SAM.csv", index=False)
        print(f"[live summary] {len(df)} completed run(s) -> {OUTPUT_ROOT / 'LIVE_HySure_results.csv'}")


# =============================================================================
# ANALYSIS
# =============================================================================
def result_to_row(payload: Dict[str, object], method: str | None = None) -> Dict[str, object]:
    return {
        "method": method or payload.get("method", "unknown"),
        "dataset": payload.get("dataset"),
        "configuration": payload.get("configuration", payload.get("experiment")),
        "n_channels": payload.get("n_channels"),
        "wavelengths_nm": payload.get("wavelengths_nm"),
        "SAM": payload.get("SAM"),
        "PSNR": payload.get("PSNR"),
        "ERGAS": payload.get("ERGAS"),
        "RMSE_full": payload.get("RMSE_full"),
        "RMSE_visible": payload.get("RMSE_visible"),
        "RMSE_NIR": payload.get("RMSE_NIR"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "converged": payload.get("hysure", {}).get("converged"),
        "iterations_run": payload.get("hysure", {}).get("iterations_run"),
        "spectral_response_mode": payload.get("spectral_response_mode"),
    }


def collect_hysure_results() -> pd.DataFrame:
    rows = []
    for dataset in ("Chikusei", "ICVL"):
        for cfg in RUN_CONFIGS:
            p = OUTPUT_ROOT / dataset / cfg / f"seed_{SEED}" / "result.json"
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("status") == "complete":
                rows.append(result_to_row(payload, "HySure-SVD"))
    return pd.DataFrame(rows)


def collect_main_results() -> pd.DataFrame:
    rows = []
    if not MAIN_STUDY_ROOT.exists():
        return pd.DataFrame(rows)
    for dataset in ("Chikusei", "ICVL"):
        for cfg in RUN_CONFIGS:
            p = MAIN_STUDY_ROOT / dataset / "experiments" / cfg / f"seed_{SEED}" / "result.json"
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("status") == "complete":
                row = result_to_row(payload, "Gram-framework")
                row["configuration"] = cfg
                rows.append(row)
    return pd.DataFrame(rows)


def collect_edip_results() -> pd.DataFrame:
    rows = []
    if not EDIP_ROOT.exists():
        return pd.DataFrame(rows)
    for dataset in ("Chikusei", "ICVL"):
        for cfg in RUN_CONFIGS:
            p = EDIP_ROOT / dataset / cfg / f"seed_{SEED}" / "result_fair.json"
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("status") == "complete":
                rows.append(result_to_row(payload, "EDIP-Net fair"))
    return pd.DataFrame(rows)


def ordered_subset(df: pd.DataFrame, configs: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[df["configuration"].isin(configs)].copy()
    order = {c: i for i, c in enumerate(configs)}
    out["_order"] = out["configuration"].map(order)
    return out.sort_values(["dataset", "method", "_order"]).drop(columns="_order")


def spanning_improvement_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in combined.groupby(["dataset", "method"]):
        lookup = dict(zip(g["configuration"], g["SAM"]))
        span = lookup.get("spanning_549_801")
        if span is None or not np.isfinite(span):
            continue
        for control in ("two_vis_461_549", "two_nir_801_935"):
            val = lookup.get(control)
            if val is None or not np.isfinite(val):
                continue
            rows.append({
                "dataset": dataset,
                "method": method,
                "control": control,
                "control_SAM": float(val),
                "spanning_549_801_SAM": float(span),
                "absolute_SAM_reduction": float(val - span),
                "percent_SAM_reduction": float(100.0 * (val - span) / val),
            })
    return pd.DataFrame(rows)


def write_analysis(make_plots_now: bool = MAKE_PLOTS) -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    hy = collect_hysure_results()
    main = collect_main_results()
    edip = collect_edip_results()

    if hy.empty:
        print("[analysis] no completed HySure results yet")
        return

    hy.to_csv(ANALYSIS_ROOT / "00_HySure_all_runs.csv", index=False)
    hy.sort_values(["dataset", "SAM"]).to_csv(ANALYSIS_ROOT / "01_HySure_ranked_by_SAM.csv", index=False)

    frames = [x for x in (main, edip, hy) if not x.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(ANALYSIS_ROOT / "02_cross_method_all_matching.csv", index=False)

    ordered_subset(combined, PAIR_REGION_CONFIGS).to_csv(
        ANALYSIS_ROOT / "03_two_band_region_comparison.csv", index=False
    )
    spanning_improvement_table(combined).to_csv(
        ANALYSIS_ROOT / "04_two_band_spanning_improvements.csv", index=False
    )
    ordered_subset(combined, VISIBLE_ANCHOR_CONFIGS).to_csv(
        ANALYSIS_ROOT / "05_visible_anchor_fixed_801.csv", index=False
    )
    ordered_subset(combined, NIR_ANCHOR_CONFIGS).to_csv(
        ANALYSIS_ROOT / "06_nir_anchor_fixed_549.csv", index=False
    )
    ordered_subset(combined, BAND_COUNT_CONFIGS).to_csv(
        ANALYSIS_ROOT / "07_band_count_1_to_4.csv", index=False
    )
    ordered_subset(combined, SINGLE_CONFIGS).to_csv(
        ANALYSIS_ROOT / "08_single_band_comparison.csv", index=False
    )
    hy[["dataset", "configuration", "RMSE_full", "RMSE_visible", "RMSE_NIR"]].to_csv(
        ANALYSIS_ROOT / "09_HySure_regional_RMSE.csv", index=False
    )

    manuscript = ordered_subset(combined, RGB_CONFIGS + PAIR_REGION_CONFIGS + BAND_COUNT_CONFIGS)
    manuscript.to_csv(ANALYSIS_ROOT / "10_manuscript_baseline_table.csv", index=False)

    ordered_subset(combined, RGB_CONFIGS + ("spanning_549_801",)).to_csv(
        ANALYSIS_ROOT / "11_RGB_comparison.csv", index=False
    )

    # Explicit convergence audit: only converged HySure runs enter the final tables.
    convergence_rows = []
    for dataset_name in ("Chikusei", "ICVL"):
        for cfg in RUN_CONFIGS:
            rp = OUTPUT_ROOT / dataset_name / cfg / f"seed_{SEED}" / "result.json"
            if not rp.exists():
                continue
            try:
                payload = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                continue
            convergence_rows.append({
                "dataset": dataset_name,
                "configuration": cfg,
                "status": payload.get("status"),
                "converged": payload.get("hysure", {}).get("converged"),
                "iterations_run": payload.get("hysure", {}).get("iterations_run"),
                "final_relative_X_change": payload.get("hysure", {}).get("final_relative_X_change"),
                "final_relative_objective_change": payload.get("hysure", {}).get("final_relative_objective_change"),
                "final_max_primal_residual": payload.get("hysure", {}).get("final_max_primal_residual"),
            })
    pd.DataFrame(convergence_rows).to_csv(ANALYSIS_ROOT / "12_convergence_audit.csv", index=False)

    audit = {
        "method": "HySure-SVD",
        "source": "PyTorch port of official HySure data_fusion.m SALSA equations",
        "official_repo": "https://github.com/alfaiate/HySure",
        "datasets": ["Chikusei", "ICVL"],
        "run_configs": list(RUN_CONFIGS),
        "same_main_study_mutual_max_normalization": True,
        "same_main_study_LR_HSI": True,
        "same_main_study_selected_HR_narrowbands": True,
        "narrowband_R": "known exact 0/1 selection",
        "RGB_R": "estimated by NNLS from observed LR-HSI and spatially degraded observed HR-RGB only",
        "known_exact_B_given_to_HySure": True,
        "basis": HYSURE_BASIS,
        "p": HYSURE_P,
        "lambda_phi": HYSURE_LAMBDA_PHI,
        "lambda_m": HYSURE_LAMBDA_M,
        "mu": HYSURE_MU,
        "max_iterations": HYSURE_MAX_ITERS,
        "min_iterations": HYSURE_MIN_ITERS,
        "check_every": HYSURE_CHECK_EVERY,
        "relative_X_tolerance": HYSURE_RELX_TOL,
        "primal_tolerance": HYSURE_PRIMAL_TOL,
        "objective_relative_tolerance": HYSURE_OBJECTIVE_REL_TOL,
        "stable_checks_required": HYSURE_STABLE_CHECKS,
        "torch_dtype": str(TORCH_DTYPE),
        "configuration_order_reason": "decisive two-band tests first, then RGB, band-count, anchor, and single-band analyses",
        "GT_used_for_fitting_or_selection": False,
        "GT_used_for_final_metrics_only": True,
        "main_study_results_found": not main.empty,
        "EDIP_results_found": not edip.empty,
    }
    (ANALYSIS_ROOT / "13_protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if make_plots_now:
        make_plots(combined, hy)

    print(f"\n[analysis complete] {ANALYSIS_ROOT.resolve()}")
    for p in sorted(ANALYSIS_ROOT.iterdir()):
        if p.is_file():
            print("  ", p.name)


def make_plots(combined: pd.DataFrame, hy: pd.DataFrame) -> None:
    labels = {
        "rgb_only": "RGB",
        "rgb_plus_nir_801": "RGB+801",
        "two_vis_461_549": "461+549",
        "two_nir_801_935": "801+935",
        "spanning_549_801": "549+801",
        "spanning_461_801": "461+801",
        "spanning_652_801": "652+801",
        "spanning_549_714": "549+714",
        "spanning_549_935": "549+935",
        "single_nir_801": "801",
        "spanning_461_549_801": "461+549+801",
        "spanning_461_549_652_801": "461+549+652+801",
    }

    # 1. Decisive two-band spectral-region control for each dataset.
    for dataset in ("Chikusei", "ICVL"):
        sub = combined[(combined["dataset"] == dataset) & combined["configuration"].isin(PAIR_REGION_CONFIGS)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = list(dict.fromkeys(sub["method"].tolist()))
        x = np.arange(len(PAIR_REGION_CONFIGS), dtype=float)
        width = 0.8 / max(len(methods), 1)
        for mi, method in enumerate(methods):
            g = sub[sub["method"] == method]
            lookup = dict(zip(g["configuration"], g["SAM"]))
            vals = [lookup.get(c, np.nan) for c in PAIR_REGION_CONFIGS]
            ax.bar(x + (mi - (len(methods)-1)/2)*width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in PAIR_REGION_CONFIGS])
        ax.set_ylabel("SAM (degrees, lower is better)")
        ax.set_title(f"{dataset}: equal-count two-band location test")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ANALYSIS_ROOT / f"fig_two_band_region_{dataset}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2. 1->4 band progression across methods/datasets.
    sub = combined[combined["configuration"].isin(BAND_COUNT_CONFIGS)]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        count_map = {cfg: i+1 for i, cfg in enumerate(BAND_COUNT_CONFIGS)}
        for (dataset, method), g in sub.groupby(["dataset", "method"]):
            g = g.copy(); g["band_count"] = g["configuration"].map(count_map)
            g = g.sort_values("band_count")
            ax.plot(g["band_count"], g["SAM"], marker="o", label=f"{dataset} | {method}")
        ax.set_xticks([1,2,3,4])
        ax.set_xlabel("Number of HR narrowbands")
        ax.set_ylabel("SAM (degrees)")
        ax.set_title("Distributed-band progression")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ANALYSIS_ROOT / "fig_band_count_1_to_4_cross_method.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 3/4. Anchor scans.
    for configs, filename, title in [
        (VISIBLE_ANCHOR_CONFIGS, "fig_visible_anchor_fixed_801_cross_method.png", "Visible anchor with ~801 nm fixed"),
        (NIR_ANCHOR_CONFIGS, "fig_nir_anchor_fixed_549_cross_method.png", "NIR anchor with ~549 nm fixed"),
    ]:
        sub = combined[combined["configuration"].isin(configs)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        order = {c:i for i,c in enumerate(configs)}
        for (dataset, method), g in sub.groupby(["dataset", "method"]):
            g = g.copy(); g["_o"] = g["configuration"].map(order); g = g.sort_values("_o")
            ax.plot(range(len(g)), g["SAM"], marker="o", label=f"{dataset} | {method}")
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([labels[c] for c in configs])
        ax.set_ylabel("SAM (degrees)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ANALYSIS_ROOT / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 5. RGB comparison across methods/datasets.
    sub = combined[combined["configuration"].isin(RGB_CONFIGS + ("spanning_549_801",))]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        configs = RGB_CONFIGS + ("spanning_549_801",)
        x = np.arange(len(configs), dtype=float)
        groups = list(sub.groupby(["dataset", "method"]))
        for (dataset_name, method), g in groups:
            lookup = dict(zip(g["configuration"], g["SAM"]))
            vals = [lookup.get(c, np.nan) for c in configs]
            ax.plot(x, vals, marker="o", label=f"{dataset_name} | {method}")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in configs])
        ax.set_ylabel("SAM (degrees)")
        ax.set_title("RGB-related comparison")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ANALYSIS_ROOT / "fig_RGB_comparison_cross_method.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 6. HySure regional RMSE.
    key = hy[hy["configuration"].isin(("single_nir_801", "two_nir_801_935", "spanning_549_801", "spanning_461_549_652_801"))]
    for dataset in ("Chikusei", "ICVL"):
        sub = key[key["dataset"] == dataset]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width/2, sub["RMSE_visible"], width=width, label="Visible")
        ax.bar(x + width/2, sub["RMSE_NIR"], width=width, label="NIR")
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(c,c) for c in sub["configuration"]], rotation=15)
        ax.set_ylabel("RMSE")
        ax.set_title(f"{dataset}: HySure regional reconstruction error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ANALYSIS_ROOT / f"fig_HySure_regional_RMSE_{dataset}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# SANITY TEST (small synthetic problem) -- no dataset files required
# =============================================================================
def _self_test_forward_operator() -> None:
    rng = np.random.default_rng(0)
    h = w = 16
    c = 4
    s = 4
    x = rng.random((h, w, c), dtype=np.float32)
    k = matlab_style_gaussian_2d((s, s), 0.5)
    lr = np.zeros((h//s, w//s, c), dtype=np.float32)
    for b in range(c):
        lr[..., b] = signal.convolve2d(x[..., b], k, mode="valid")[::s, ::s]
    B = np.zeros((h,w), dtype=np.float32)
    for a in range(s):
        for b in range(s):
            B[(-a)%h, (-b)%w] = k[a,b]
    xx = np.transpose(x, (2,0,1))
    circ = np.fft.ifft2(np.fft.fft2(xx, axes=(-2,-1)) * np.fft.fft2(B), axes=(-2,-1)).real
    err = float(np.max(np.abs(np.transpose(circ[:,::s,::s], (1,2,0)) - lr)))
    if err > 1e-6:
        raise RuntimeError(f"Forward-operator self-test failed: {err}")
    print(f"[self-test] exact valid-convolution FFT representation max error = {err:.3e}")


# =============================================================================
# EXECUTE
# =============================================================================
if __name__ == "__main__":
    print("=" * 100)
    print("HySure COMPLETE FAIR FULL-RANGE BASELINE — ALL 16 CONFIGURATIONS")
    print("=" * 100)
    print("device:", DEVICE)
    print("run configs:", RUN_CONFIGS)
    print(
        "HySure settings:",
        f"basis={HYSURE_BASIS}, p={HYSURE_P}, lambda_phi={HYSURE_LAMBDA_PHI}, "
        f"lambda_m={HYSURE_LAMBDA_M}, mu={HYSURE_MU}, max_iters={HYSURE_MAX_ITERS}, "
        f"min_iters={HYSURE_MIN_ITERS}, check_every={HYSURE_CHECK_EVERY}"
    )
    print("output root:", OUTPUT_ROOT.resolve())
    _self_test_forward_operator()

    # Validate configuration list before any expensive work.
    unknown = [cfg for cfg in RUN_CONFIGS if cfg not in VALID_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown RUN_CONFIGS entries: {unknown}")
    if len(set(RUN_CONFIGS)) != len(RUN_CONFIGS):
        raise ValueError("RUN_CONFIGS contains duplicate configurations")

    selected_run_configs = RUN_CONFIGS if RUN_ONLY_FIRST_N is None else RUN_CONFIGS[:int(RUN_ONLY_FIRST_N)]
    print("effective configs this launch:", selected_run_configs)
    print("torch dtype:", TORCH_DTYPE)

    # Make deterministic bookkeeping explicit. HySure-SVD itself has no random initialization.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    datasets: List[PreparedDataset] = []
    if RUN_CHIKUSEI:
        datasets.append(prepare_chikusei())
    if RUN_ICVL:
        datasets.append(prepare_icvl())

    if RUN_HYSURE:
        for dataset in datasets:
            print(f"\n[{dataset.name}] HSI shape={dataset.hsi_hwc.shape}; common_max={dataset.common_max:.10g}")
            for cfg in selected_run_configs:
                run_one(dataset, cfg)
                write_live_summary()
            if ANALYSIS_AFTER_EACH_DATASET and RUN_ANALYSIS:
                write_analysis(make_plots_now=False)
            # Release one full data cube before loading/continuing the next dataset.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if RUN_ANALYSIS:
        write_live_summary()
        write_analysis(make_plots_now=MAKE_PLOTS)



















