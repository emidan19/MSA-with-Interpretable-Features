#!/usr/bin/env python3
"""Run an unsupervised downstream music-structure baseline.

The script consumes the beat-synchronous features produced by
`extract_msa_features.py`, predicts structural boundaries through SSM fusion
and checkerboard novelty, clusters predicted segments, evaluates against AIST
RWC CHORUS annotations when available, and writes visual reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_CACHE_ROOT = os.path.join(os.getcwd(), ".cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))
for _cache_dir in (os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"]):
    os.makedirs(_cache_dir, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, windows
from sklearn.cluster import AgglomerativeClustering


EXPERIMENTS: dict[str, dict[str, float]] = {
    "E0_harmonic": {"chroma": 0.5, "cens": 0.5},
    "E1_mfcc": {"mfcc": 1.0},
    "E2_stm": {"stm": 1.0},
    "E3_content": {"mfcc": 0.34, "chroma": 0.33, "cens": 0.33},
    "E4_stm_mfcc": {"stm": 0.5, "mfcc": 0.5},
    "E5_stm_harmonic": {"stm": 0.5, "chroma": 0.25, "cens": 0.25},
    "E6_stm_content": {"stm": 0.25, "mfcc": 0.25, "chroma": 0.25, "cens": 0.25},
    "E7_stm_content_f0": {"stm": 0.22, "mfcc": 0.22, "chroma": 0.22, "cens": 0.22, "f0": 0.12},
    "E8_adaptive_stm_content": {"stm": 1.0, "mfcc": 1.0, "chroma": 1.0, "cens": 1.0},
    "E9_adaptive_stm_content_f0": {"stm": 1.0, "mfcc": 1.0, "chroma": 1.0, "cens": 1.0, "f0": 1.0},
    "E10_mfcc_multiscale": {"mfcc": 1.0},
    "E11_anchor_mfcc_multiscale": {"mfcc": 0.55, "stm": 0.15, "chroma": 0.15, "cens": 0.15},
    "E12_adaptive_multiscale_stm_content": {"stm": 1.0, "mfcc": 1.0, "chroma": 1.0, "cens": 1.0},
    "E13_anchor_dense": {"mfcc": 0.60, "stm": 0.12, "chroma": 0.14, "cens": 0.14},
    "E14_mfcc_multiscale_dense": {"mfcc": 1.0},
    "E17_learned_ssm_loo": {"stm": 0.25, "mfcc": 0.25, "chroma": 0.25, "cens": 0.25},
    "E18_product_mfcc_cens": {"mfcc": 0.9, "cens": 0.1},
    "E19_product_stm_mfcc_cens": {"stm": 0.1, "mfcc": 0.8, "cens": 0.1},
    "E20_poly_mfcc_cens": {"mfcc": 0.75, "cens": 0.10, "mfcc*cens": 0.15},
    "E21_poly_stm_mfcc_cens": {
        "stm": 0.08,
        "mfcc": 0.70,
        "cens": 0.08,
        "stm*mfcc": 0.07,
        "mfcc*cens": 0.07,
    },
    "E22_arrangement": {"arrangement": 1.0},
    "E23_vocal_proxy": {"vocal": 1.0},
    "E24_bass_lowend": {"bass": 1.0},
    "E25_tonnetz_hcdf": {"tonnetz": 1.0},
    "E26_density_dynamics": {"density": 1.0},
    "E27_orthogonal_all": {
        "arrangement": 0.25,
        "vocal": 0.20,
        "bass": 0.20,
        "tonnetz": 0.20,
        "density": 0.15,
    },
    "E28_mfcc_cens_arrangement": {"mfcc": 0.70, "cens": 0.10, "arrangement": 0.20},
    "E29_mfcc_cens_bass_tonnetz": {"mfcc": 0.65, "cens": 0.10, "bass": 0.15, "tonnetz": 0.10},
    "E30_adaptive_extended": {
        "mfcc": 1.0,
        "cens": 1.0,
        "arrangement": 1.0,
        "vocal": 1.0,
        "bass": 1.0,
        "tonnetz": 1.0,
        "density": 1.0,
    },
    "E31_learned_extended_loo": {
        "mfcc": 0.20,
        "cens": 0.15,
        "arrangement": 0.15,
        "vocal": 0.10,
        "bass": 0.15,
        "tonnetz": 0.10,
        "density": 0.15,
    },
    "E32_poly_mfcc_cens_arrangement": {
        "mfcc": 0.62,
        "cens": 0.08,
        "arrangement": 0.15,
        "mfcc*cens": 0.08,
        "mfcc*arrangement": 0.07,
    },
    "E33_reliability_gated_loo": {
        "stm": 0.20,
        "mfcc": 0.20,
        "cens": 0.20,
        "arrangement": 0.20,
        "density": 0.20,
    },
    "E34_consensus_peak_voting": {
        "mfcc": 0.35,
        "arrangement": 0.25,
        "stm": 0.20,
        "cens": 0.10,
        "density": 0.10,
    },
    "E35_path_enhanced_loo": {
        "stm": 0.25,
        "mfcc": 0.25,
        "cens": 0.25,
        "arrangement": 0.25,
    },
    "E36_path_consensus_rescore_loo": {
        "stm": 0.20,
        "mfcc": 0.20,
        "cens": 0.20,
        "arrangement": 0.20,
        "density": 0.20,
    },
    "E37_path_product_loo": {
        "stm": 0.25,
        "mfcc": 0.25,
        "cens": 0.25,
        "arrangement": 0.25,
    },
    "E38_path_poly_mfcc_arrangement": {
        "mfcc": 0.50,
        "arrangement": 0.25,
        "mfcc*arrangement": 0.15,
        "mfcc*cens": 0.05,
        "stm*mfcc": 0.05,
    },
    "E39_path_selective_consensus": {
        "mfcc": 0.50,
        "arrangement": 0.50,
    },
}

EXPERIMENT_OPTIONS: dict[str, dict] = {
    "E8_adaptive_stm_content": {"adaptive": True, "late_fusion": True},
    "E9_adaptive_stm_content_f0": {"adaptive": True, "late_fusion": True},
    "E10_mfcc_multiscale": {"multiscale": True},
    "E11_anchor_mfcc_multiscale": {"late_fusion": True, "multiscale": True},
    "E12_adaptive_multiscale_stm_content": {
        "adaptive": True,
        "late_fusion": True,
        "multiscale": True,
    },
    "E13_anchor_dense": {"late_fusion": True, "multiscale": True, "peak_profile": "dense"},
    "E14_mfcc_multiscale_dense": {"multiscale": True, "peak_profile": "dense"},
    "E17_learned_ssm_loo": {"supervised": True},
    "E18_product_mfcc_cens": {"product_fusion": True},
    "E19_product_stm_mfcc_cens": {"product_fusion": True},
    "E20_poly_mfcc_cens": {"polynomial_fusion": True},
    "E21_poly_stm_mfcc_cens": {"polynomial_fusion": True},
    "E30_adaptive_extended": {"adaptive": True, "late_fusion": True},
    "E31_learned_extended_loo": {
        "supervised": True,
        "learned_streams": "mfcc,cens,arrangement,vocal,bass,tonnetz,density",
        "learned_weight_step": 0.25,
    },
    "E32_poly_mfcc_cens_arrangement": {"polynomial_fusion": True},
    "E33_reliability_gated_loo": {
        "supervised": True,
        "reliability_gated": True,
        "learned_streams": "stm,mfcc,cens,arrangement,density",
        "learned_weight_step": 0.25,
    },
    "E34_consensus_peak_voting": {
        "reliability_gated": True,
        "consensus_fusion": True,
        "multiscale": True,
    },
    "E35_path_enhanced_loo": {
        "supervised": True,
        "path_enhanced": True,
        "learned_streams": "stm,mfcc,cens,arrangement",
        "learned_weight_step": 0.25,
    },
    "E36_path_consensus_rescore_loo": {
        "supervised": True,
        "path_enhanced": True,
        "candidate_rescore": True,
        "learned_streams": "stm,mfcc,cens,arrangement,density",
        "learned_weight_step": 0.25,
        "candidate_streams": "stm,mfcc,cens,arrangement,density",
    },
    "E37_path_product_loo": {
        "supervised": True,
        "path_enhanced": True,
        "product_fusion": True,
        "learned_streams": "stm,mfcc,cens,arrangement",
        "learned_weight_step": 0.25,
    },
    "E38_path_poly_mfcc_arrangement": {
        "polynomial_fusion": True,
        "path_enhanced": True,
    },
    "E39_path_selective_consensus": {
        "path_enhanced": True,
        "candidate_rescore": True,
        "candidate_selective": True,
        "candidate_blend": 0.28,
        "candidate_streams": "stm,mfcc,cens,arrangement,density",
    },
}

PEAK_PROFILES = {
    "balanced": {"min_segment_s": 7.5, "target_segment_s": 12.0},
    "dense": {"min_segment_s": 5.5, "target_segment_s": 10.0},
    "conservative": {"min_segment_s": 10.0, "target_segment_s": 15.0},
}

STREAM_TO_SSM = {
    "stm": "ssm_stm",
    "mfcc": "ssm_mfcc",
    "chroma": "ssm_chroma",
    "cens": "ssm_cens",
    "f0": "ssm_f0",
    "arrangement": "ssm_arrangement",
    "vocal": "ssm_vocal",
    "bass": "ssm_bass",
    "tonnetz": "ssm_tonnetz",
    "density": "ssm_density",
}

STREAM_TO_BEAT_FEATURE = {
    "stm": "beat_stm",
    "mfcc": "beat_mfcc",
    "chroma": "beat_chroma",
    "cens": "beat_cens",
    "f0": "beat_f0",
    "arrangement": "beat_arrangement",
    "vocal": "beat_vocal",
    "bass": "beat_bass",
    "tonnetz": "beat_tonnetz",
    "density": "beat_density",
}


@dataclass
class ReferenceSegments:
    track_id: str
    segments: list[tuple[float, float, str]]

    @property
    def internal_boundaries(self) -> np.ndarray:
        if len(self.segments) <= 1:
            return np.zeros(0, dtype=np.float32)
        return np.asarray([start for start, _, _ in self.segments[1:]], dtype=np.float32)

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        return float(max(end for _, end, _ in self.segments))

    @property
    def labels(self) -> list[str]:
        return [label for _, _, label in self.segments]


@dataclass
class BoundaryMetrics:
    precision: float
    recall: float
    f_measure: float
    matches: int
    pred_count: int
    ref_count: int
    median_abs_error_s: float | None


def safe_nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_ssm(matrix: np.ndarray) -> np.ndarray:
    matrix = safe_nan_to_num(matrix).astype(np.float32, copy=False)
    if matrix.size == 0:
        return matrix
    matrix = 0.5 * (matrix + matrix.T)
    lo, hi = np.percentile(matrix, [5, 95])
    if hi <= lo:
        out = np.zeros_like(matrix)
    else:
        out = np.clip((matrix - lo) / (hi - lo), 0.0, 1.0)
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def f0_gate_enabled(data: np.lib.npyio.NpzFile) -> bool:
    beat_f0 = np.asarray(data["beat_f0"])
    confidence = np.asarray(data["f0_confidence"])
    voiced_ratio = float(np.mean(beat_f0[2] > 0.5)) if beat_f0.size else 0.0
    mean_confidence = float(np.mean(confidence)) if confidence.size else 0.0
    return mean_confidence >= 0.12 and voiced_ratio >= 0.35


def gated_weights(weights: dict[str, float], data: np.lib.npyio.NpzFile) -> dict[str, float]:
    out = dict(weights)
    if "f0" in out and not f0_gate_enabled(data):
        out.pop("f0", None)
    total = sum(max(v, 0.0) for v in out.values())
    if total <= 0:
        return out
    return {key: value / total for key, value in out.items()}


def stream_novelty_salience(
    data: np.lib.npyio.NpzFile,
    stream: str,
    multiscale: bool = False,
    path_enhanced: bool = False,
) -> float:
    matrix = stream_matrix(data, stream, path_enhanced=path_enhanced)
    novelty = multiscale_novelty_curve(matrix) if multiscale else novelty_curve(matrix)
    if novelty.size == 0:
        return 0.0
    spread = float(np.percentile(novelty, 95) - np.percentile(novelty, 50))
    variability = float(np.std(novelty))
    peakiness = float(np.max(novelty))
    return max(1e-4, spread + 0.5 * variability + 0.15 * peakiness)


def adaptive_weights(
    weights: dict[str, float],
    data: np.lib.npyio.NpzFile,
    multiscale: bool = False,
    path_enhanced: bool = False,
) -> dict[str, float]:
    gated = gated_weights(weights, data)
    scored = {
        stream: base_weight
        * stream_novelty_salience(data, stream, multiscale=multiscale, path_enhanced=path_enhanced)
        for stream, base_weight in gated.items()
    }
    total = sum(scored.values())
    if total <= 0:
        return gated
    normalized = {stream: value / total for stream, value in scored.items()}
    return normalized


def path_enhance_ssm(matrix: np.ndarray, radius: int = 3, blend: float = 0.45) -> np.ndarray:
    """Smooth along diagonal paths to emphasize repeated temporal trajectories."""
    matrix = normalize_ssm(matrix)
    if matrix.size == 0 or radius <= 0:
        return matrix
    acc = np.array(matrix, dtype=np.float32, copy=True)
    count = np.ones_like(matrix, dtype=np.float32)
    for offset in range(1, radius + 1):
        acc[:-offset, :-offset] += matrix[offset:, offset:]
        count[:-offset, :-offset] += 1.0
        acc[offset:, offset:] += matrix[:-offset, :-offset]
        count[offset:, offset:] += 1.0
    enhanced = acc / np.maximum(count, 1.0)
    return normalize_ssm((1.0 - blend) * matrix + blend * enhanced)


def ssm_reliability_features(matrix: np.ndarray) -> dict[str, float]:
    matrix = normalize_ssm(matrix)
    if matrix.size == 0 or matrix.shape[0] < 8:
        return {"contrast": 0.0, "local_gap": 0.0, "recurrence": 0.0}
    n = matrix.shape[0]
    ii, jj = np.indices((n, n))
    dist = np.abs(ii - jj)
    off = dist > 0
    near = (dist > 0) & (dist <= 8)
    far = dist >= min(32, max(9, n // 8))
    off_vals = matrix[off]
    contrast = float(np.std(off_vals)) if off_vals.size else 0.0
    local_gap = float(np.mean(matrix[near]) - np.mean(matrix[far])) if np.any(near) and np.any(far) else 0.0
    recurrence = float(np.mean(off_vals >= 0.82)) if off_vals.size else 0.0
    return {"contrast": contrast, "local_gap": local_gap, "recurrence": recurrence}


def reliability_gated_weights(
    weights: dict[str, float],
    data: np.lib.npyio.NpzFile,
    multiscale: bool = False,
    path_enhanced: bool = False,
) -> dict[str, float]:
    gated = gated_weights(weights, data)
    scored: dict[str, float] = {}
    for stream, base_weight in gated.items():
        key = STREAM_TO_SSM.get(stream)
        if not key or key not in data:
            continue
        matrix = normalize_ssm(np.asarray(data[key]))
        if path_enhanced:
            matrix = path_enhance_ssm(matrix)
        diag = ssm_reliability_features(matrix)
        novelty_salience = stream_novelty_salience(
            data,
            stream,
            multiscale=multiscale,
            path_enhanced=path_enhanced,
        )
        local_bonus = 1.0 + 1.5 * max(0.0, diag["local_gap"])
        contrast_bonus = 0.65 + diag["contrast"]
        recurrence_bonus = 1.0 + 0.5 * min(0.25, diag["recurrence"])
        scored[stream] = base_weight * novelty_salience * local_bonus * contrast_bonus * recurrence_bonus
    total = sum(scored.values())
    if total <= 0:
        return gated
    return {stream: value / total for stream, value in scored.items() if value > 0}


def stream_matrix(data: np.lib.npyio.NpzFile, stream: str, path_enhanced: bool = False) -> np.ndarray:
    matrix = normalize_ssm(np.asarray(data[STREAM_TO_SSM[stream]]))
    return path_enhance_ssm(matrix) if path_enhanced else matrix


def fuse_ssms(data: np.lib.npyio.NpzFile, weights: dict[str, float], path_enhanced: bool = False) -> np.ndarray:
    matrices = []
    used_weights = []
    for stream, weight in weights.items():
        key = STREAM_TO_SSM[stream]
        if key not in data:
            continue
        matrix = stream_matrix(data, stream, path_enhanced=path_enhanced)
        if matrix.size:
            matrices.append(matrix)
            used_weights.append(float(weight))
    if not matrices:
        raise ValueError("No SSMs available for fusion")
    used_weights = np.asarray(used_weights, dtype=np.float32)
    used_weights = used_weights / np.sum(used_weights)
    fused = np.zeros_like(matrices[0], dtype=np.float32)
    for weight, matrix in zip(used_weights, matrices):
        fused += weight * matrix
    return normalize_ssm(fused)


def fuse_ssms_product(
    data: np.lib.npyio.NpzFile,
    weights: dict[str, float],
    eps: float = 1e-6,
    path_enhanced: bool = False,
) -> np.ndarray:
    matrices = []
    used_weights = []
    for stream, weight in weights.items():
        key = STREAM_TO_SSM[stream]
        if key not in data:
            continue
        matrix = stream_matrix(data, stream, path_enhanced=path_enhanced)
        if matrix.size:
            matrices.append(np.clip(matrix, eps, 1.0))
            used_weights.append(float(weight))
    if not matrices:
        raise ValueError("No SSMs available for product fusion")
    used_weights = np.asarray(used_weights, dtype=np.float32)
    used_weights = used_weights / np.sum(used_weights)
    fused_log = np.zeros_like(matrices[0], dtype=np.float32)
    for weight, matrix in zip(used_weights, matrices):
        fused_log += weight * np.log(matrix)
    return normalize_ssm(np.exp(fused_log).astype(np.float32))


def component_streams(weights: dict[str, float]) -> list[str]:
    streams: list[str] = []
    for component in weights:
        for stream in component.split("*"):
            if stream in STREAM_TO_SSM and stream not in streams:
                streams.append(stream)
    return streams


def component_matrix(data: np.lib.npyio.NpzFile, component: str, path_enhanced: bool = False) -> np.ndarray:
    parts = component.split("*")
    matrices = []
    for stream in parts:
        if stream not in STREAM_TO_SSM:
            raise ValueError(f"Unknown SSM component stream: {stream}")
        matrix = stream_matrix(data, stream, path_enhanced=path_enhanced)
        if matrix.size:
            matrices.append(matrix)
    if not matrices:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.ones_like(matrices[0], dtype=np.float32)
    for matrix in matrices:
        out *= matrix
    return normalize_ssm(out)


def fuse_ssms_polynomial(
    data: np.lib.npyio.NpzFile,
    weights: dict[str, float],
    path_enhanced: bool = False,
) -> np.ndarray:
    matrices = []
    used_weights = []
    for component, weight in weights.items():
        matrix = component_matrix(data, component, path_enhanced=path_enhanced)
        if matrix.size:
            matrices.append(matrix)
            used_weights.append(float(weight))
    if not matrices:
        raise ValueError("No SSMs available for polynomial fusion")
    used_weights = np.asarray(used_weights, dtype=np.float32)
    used_weights = used_weights / np.sum(used_weights)
    fused = np.zeros_like(matrices[0], dtype=np.float32)
    for weight, matrix in zip(used_weights, matrices):
        fused += weight * matrix
    return normalize_ssm(fused)


def checkerboard_kernel(half_width: int) -> np.ndarray:
    size = 2 * half_width
    gaussian = windows.gaussian(size, std=max(1.0, half_width / 2.0))
    kernel = np.outer(gaussian, gaussian)
    signs = np.ones((size, size), dtype=np.float32)
    signs[:half_width, half_width:] = -1.0
    signs[half_width:, :half_width] = -1.0
    kernel = kernel * signs
    kernel = kernel - np.mean(kernel)
    norm = np.sum(np.abs(kernel))
    if norm > 0:
        kernel = kernel / norm
    return kernel.astype(np.float32)


def novelty_curve(ssm: np.ndarray, half_width: int = 16, smoothing_sigma: float = 2.0) -> np.ndarray:
    n = ssm.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    half_width = int(min(max(4, half_width), max(4, n // 3)))
    kernel = checkerboard_kernel(half_width)
    novelty = np.zeros(n, dtype=np.float32)
    for idx in range(half_width, n - half_width):
        local = ssm[idx - half_width : idx + half_width, idx - half_width : idx + half_width]
        novelty[idx] = float(np.sum(local * kernel))
    novelty = safe_nan_to_num(novelty)
    novelty = np.maximum(novelty, 0.0)
    if smoothing_sigma > 0:
        novelty = gaussian_filter1d(novelty, sigma=smoothing_sigma).astype(np.float32)
    peak = float(np.max(novelty)) if novelty.size else 0.0
    if peak > 0:
        novelty = novelty / peak
    return novelty.astype(np.float32)


def multiscale_novelty_curve(
    ssm: np.ndarray,
    half_widths: tuple[int, ...] = (8, 16, 32),
) -> np.ndarray:
    curves = []
    for half_width in half_widths:
        if ssm.shape[0] < half_width * 2 + 4:
            continue
        curves.append(novelty_curve(ssm, half_width=half_width, smoothing_sigma=max(1.0, half_width / 8.0)))
    if not curves:
        return novelty_curve(ssm)
    min_len = min(len(curve) for curve in curves)
    fused = np.mean([curve[:min_len] for curve in curves], axis=0)
    peak = float(np.max(fused)) if fused.size else 0.0
    if peak > 0:
        fused = fused / peak
    return fused.astype(np.float32)


def fuse_novelty_curves(
    data: np.lib.npyio.NpzFile,
    weights: dict[str, float],
    multiscale: bool = False,
    path_enhanced: bool = False,
) -> np.ndarray:
    curves = []
    used_weights = []
    for stream, weight in weights.items():
        matrix = stream_matrix(data, stream, path_enhanced=path_enhanced)
        curve = multiscale_novelty_curve(matrix) if multiscale else novelty_curve(matrix)
        if curve.size:
            curves.append(curve)
            used_weights.append(weight)
    if not curves:
        raise ValueError("No novelty curves available for fusion")
    min_len = min(len(curve) for curve in curves)
    used_weights = np.asarray(used_weights, dtype=np.float32)
    used_weights = used_weights / np.sum(used_weights)
    fused = np.zeros(min_len, dtype=np.float32)
    for weight, curve in zip(used_weights, curves):
        fused += weight * curve[:min_len]
    peak = float(np.max(fused)) if fused.size else 0.0
    if peak > 0:
        fused = fused / peak
    return fused.astype(np.float32)


def pulse_train_from_peaks(
    peaks: np.ndarray,
    length: int,
    sigma: float = 2.0,
) -> np.ndarray:
    pulse = np.zeros(length, dtype=np.float32)
    if length == 0 or peaks.size == 0:
        return pulse
    peaks = np.asarray(peaks, dtype=np.int64)
    peaks = peaks[(peaks >= 0) & (peaks < length)]
    pulse[peaks] = 1.0
    pulse = gaussian_filter1d(pulse, sigma=sigma).astype(np.float32)
    peak = float(np.max(pulse)) if pulse.size else 0.0
    if peak > 0:
        pulse = pulse / peak
    return pulse


def consensus_novelty_curve(
    data: np.lib.npyio.NpzFile,
    weights: dict[str, float],
    multiscale: bool = False,
    path_enhanced: bool = False,
) -> np.ndarray:
    beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
    curves = []
    used_weights = []
    for stream, weight in weights.items():
        if stream not in STREAM_TO_SSM or STREAM_TO_SSM[stream] not in data:
            continue
        matrix = stream_matrix(data, stream, path_enhanced=path_enhanced)
        curve = multiscale_novelty_curve(matrix) if multiscale else novelty_curve(matrix)
        if not curve.size:
            continue
        peaks = pick_boundary_indices(curve, beat_boundaries, **PEAK_PROFILES["balanced"])
        pulses = pulse_train_from_peaks(peaks, len(curve), sigma=2.0)
        consensus_curve = 0.60 * curve + 0.40 * pulses
        curves.append(consensus_curve.astype(np.float32))
        used_weights.append(float(weight))
    if not curves:
        raise ValueError("No novelty curves available for consensus fusion")
    min_len = min(len(curve) for curve in curves)
    used_weights = np.asarray(used_weights, dtype=np.float32)
    used_weights = used_weights / np.sum(used_weights)
    fused = np.zeros(min_len, dtype=np.float32)
    for weight, curve in zip(used_weights, curves):
        fused += weight * curve[:min_len]
    fused = gaussian_filter1d(fused, sigma=1.2).astype(np.float32)
    peak = float(np.max(fused)) if fused.size else 0.0
    if peak > 0:
        fused = fused / peak
    return fused.astype(np.float32)


def parse_stream_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [stream.strip() for stream in value.split(",") if stream.strip()]
    return [str(stream).strip() for stream in value if str(stream).strip()]


def candidate_rescored_novelty_curve(
    data: np.lib.npyio.NpzFile,
    fused_novelty: np.ndarray,
    candidate_streams: list[str],
    multiscale: bool = False,
    path_enhanced: bool = False,
    blend: float | None = None,
    selective: bool = False,
) -> np.ndarray:
    """Use per-feature peak consensus as support, then rank with the fused SSM novelty."""
    beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
    available = [
        stream
        for stream in candidate_streams
        if stream in STREAM_TO_SSM and STREAM_TO_SSM[stream] in data
    ]
    if not available:
        return fused_novelty

    base_weights = {stream: 1.0 for stream in available}
    support_weights = reliability_gated_weights(
        base_weights,
        data,
        multiscale=multiscale,
        path_enhanced=path_enhanced,
    )
    curves = []
    pulses = []
    weights = []
    for stream, weight in support_weights.items():
        matrix = stream_matrix(data, stream, path_enhanced=path_enhanced)
        curve = multiscale_novelty_curve(matrix) if multiscale else novelty_curve(matrix)
        if not curve.size:
            continue
        peaks = pick_boundary_indices(curve, beat_boundaries, **PEAK_PROFILES["dense"])
        pulse = pulse_train_from_peaks(peaks, len(curve), sigma=1.4)
        curves.append(curve.astype(np.float32))
        pulses.append(pulse.astype(np.float32))
        weights.append(float(weight))
    if not curves:
        return fused_novelty

    min_len = min([len(fused_novelty), *[len(curve) for curve in curves]])
    if min_len <= 0:
        return fused_novelty
    weights_arr = np.asarray(weights, dtype=np.float32)
    weights_arr = weights_arr / np.maximum(np.sum(weights_arr), 1e-8)
    support_curve = np.zeros(min_len, dtype=np.float32)
    support_pulse = np.zeros(min_len, dtype=np.float32)
    for weight, curve, pulse in zip(weights_arr, curves, pulses):
        support_curve += weight * curve[:min_len]
        support_pulse += weight * pulse[:min_len]
    support_curve = gaussian_filter1d(support_curve, sigma=1.2).astype(np.float32)
    support_pulse = gaussian_filter1d(support_pulse, sigma=1.0).astype(np.float32)
    support = 0.65 * support_curve + 0.35 * support_pulse
    fused = fused_novelty[:min_len]
    if blend is None:
        rescored = 0.58 * fused + 0.27 * support_curve + 0.15 * support_pulse
    else:
        gate = 1.0
        if selective:
            if np.std(fused) > 1e-8 and np.std(support) > 1e-8:
                corr = float(np.corrcoef(fused, support)[0, 1])
            else:
                corr = 0.0
            gate = float(np.clip((corr - 0.10) / 0.35, 0.0, 1.0))
        effective_blend = float(np.clip(blend * gate, 0.0, 0.75))
        rescored = (1.0 - effective_blend) * fused + effective_blend * support
    peak = float(np.max(rescored)) if rescored.size else 0.0
    if peak > 0:
        rescored = rescored / peak
    return rescored.astype(np.float32)


def pick_boundary_indices(
    novelty: np.ndarray,
    beat_boundaries: np.ndarray,
    min_segment_s: float = 7.5,
    target_segment_s: float = 12.0,
) -> np.ndarray:
    if novelty.size < 8 or beat_boundaries.size < 9:
        return np.zeros(0, dtype=np.int64)

    duration = float(beat_boundaries[-1])
    beat_durations = np.diff(beat_boundaries)
    median_beat_s = float(np.median(beat_durations[beat_durations > 0]))
    min_distance = max(4, int(round(min_segment_s / max(median_beat_s, 1e-3))))
    max_boundaries = max(1, int(duration / min_segment_s) - 1)
    target_count = int(np.clip(round(duration / target_segment_s) - 1, 2, max_boundaries))

    height = float(np.quantile(novelty, 0.62))
    prominence = max(0.03, float(np.std(novelty)) * 0.18)
    peaks, _ = find_peaks(novelty, height=height, prominence=prominence, distance=min_distance)
    if peaks.size < target_count:
        peaks, _ = find_peaks(novelty, height=float(np.quantile(novelty, 0.45)), distance=min_distance)
    if peaks.size == 0:
        candidates = np.argsort(novelty)[::-1]
    else:
        candidates = peaks[np.argsort(novelty[peaks])[::-1]]

    selected: list[int] = []
    min_time = max(4.0, min_segment_s * 0.6)
    max_time = max(min_time, duration - min_time)
    for idx in candidates:
        if idx <= 0 or idx >= len(beat_boundaries) - 1:
            continue
        time_s = float(beat_boundaries[idx])
        if time_s <= min_time or time_s >= max_time:
            continue
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(int(idx))
        if len(selected) >= target_count:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def boundaries_from_indices(indices: np.ndarray, beat_boundaries: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.zeros(0, dtype=np.float32)
    indices = np.clip(indices, 1, len(beat_boundaries) - 2)
    return np.asarray(beat_boundaries[indices], dtype=np.float32)


def match_boundaries(predicted: np.ndarray, reference: np.ndarray, tolerance_s: float) -> list[float]:
    predicted = np.asarray(sorted(predicted), dtype=np.float32)
    reference = np.asarray(sorted(reference), dtype=np.float32)
    used_ref: set[int] = set()
    errors: list[float] = []
    for pred in predicted:
        candidates = [
            (abs(float(pred - ref)), ref_idx)
            for ref_idx, ref in enumerate(reference)
            if ref_idx not in used_ref and abs(float(pred - ref)) <= tolerance_s
        ]
        if not candidates:
            continue
        error, ref_idx = min(candidates)
        used_ref.add(ref_idx)
        errors.append(float(error))
    return errors


def boundary_metrics(predicted: np.ndarray, reference: np.ndarray, tolerance_s: float) -> BoundaryMetrics:
    errors = match_boundaries(predicted, reference, tolerance_s)
    matches = len(errors)
    pred_count = int(len(predicted))
    ref_count = int(len(reference))
    precision = matches / pred_count if pred_count else 0.0
    recall = matches / ref_count if ref_count else 0.0
    f_measure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    median_error = float(np.median(errors)) if errors else None
    return BoundaryMetrics(precision, recall, f_measure, matches, pred_count, ref_count, median_error)


def parse_chorus_file(path: Path) -> ReferenceSegments:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"(\d+)\s+(\d+)\s+\"([^\"]+)\"", text)
    segments = [(int(start) / 100.0, int(end) / 100.0, label) for start, end, label in rows]
    track_id = re.search(r"RM-P(\d{3})", path.name)
    return ReferenceSegments(track_id=f"RWC_P{track_id.group(1)}" if track_id else path.stem, segments=segments)


def reference_path_for_track(annotation_dir: Path, track_id: str) -> Path | None:
    match = re.search(r"RWC_P(\d{3})", track_id)
    if not match:
        return None
    path = annotation_dir / "AIST_RWC-MDB-P-2001_CHORUS" / f"RM-P{match.group(1)}.CHORUS.TXT"
    return path if path.exists() else None


def segment_embedding(
    data: np.lib.npyio.NpzFile,
    streams: Iterable[str],
    boundaries_s: np.ndarray,
) -> np.ndarray:
    beat_boundaries = np.asarray(data["beat_boundaries"])
    rows = []
    for start, end in zip(boundaries_s[:-1], boundaries_s[1:]):
        pieces = []
        mask = (beat_boundaries[:-1] >= start) & (beat_boundaries[1:] <= end)
        if not np.any(mask):
            center = (start + end) / 2.0
            idx = int(np.clip(np.searchsorted(beat_boundaries, center), 0, len(beat_boundaries) - 2))
            mask[idx] = True
        for stream in streams:
            key = STREAM_TO_BEAT_FEATURE[stream]
            features = safe_nan_to_num(np.asarray(data[key]))
            local = features[:, mask]
            pieces.append(np.mean(local, axis=1))
            pieces.append(np.std(local, axis=1))
        rows.append(np.concatenate(pieces))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    emb = np.vstack(rows).astype(np.float32)
    mean = np.mean(emb, axis=0, keepdims=True)
    std = np.std(emb, axis=0, keepdims=True)
    return (emb - mean) / np.maximum(std, 1e-8)


def label_segments(embeddings: np.ndarray) -> list[str]:
    n_segments = embeddings.shape[0]
    if n_segments == 0:
        return []
    if n_segments == 1:
        return ["A"]
    n_clusters = int(np.clip(round(n_segments * 0.45), 2, n_segments))
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
    clusters = model.fit_predict(embeddings)
    order: dict[int, str] = {}
    labels: list[str] = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for cluster in clusters:
        cluster = int(cluster)
        if cluster not in order:
            order[cluster] = alphabet[len(order) % len(alphabet)]
        labels.append(order[cluster])
    return labels


def label_repetition_rate(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return 1.0 - (len(set(labels)) / len(labels))


def run_experiment(
    data: np.lib.npyio.NpzFile,
    exp_id: str,
    base_weights: dict[str, float],
) -> dict:
    options = EXPERIMENT_OPTIONS.get(exp_id, {})
    multiscale = bool(options.get("multiscale", False))
    path_enhanced = bool(options.get("path_enhanced", False))
    if options.get("polynomial_fusion", False):
        weights = base_weights
        fused = fuse_ssms_polynomial(data, weights, path_enhanced=path_enhanced)
        streams = component_streams(weights)
    else:
        if options.get("adaptive", False):
            weights = adaptive_weights(base_weights, data, multiscale=multiscale, path_enhanced=path_enhanced)
        elif options.get("reliability_gated", False):
            weights = reliability_gated_weights(
                base_weights,
                data,
                multiscale=multiscale,
                path_enhanced=path_enhanced,
            )
        else:
            weights = gated_weights(base_weights, data)
        fused = (
            fuse_ssms_product(data, weights, path_enhanced=path_enhanced)
            if options.get("product_fusion", False)
            else fuse_ssms(data, weights, path_enhanced=path_enhanced)
        )
        streams = list(weights)
    if options.get("consensus_fusion", False):
        novelty = consensus_novelty_curve(
            data,
            weights,
            multiscale=multiscale,
            path_enhanced=path_enhanced,
        )
    elif options.get("late_fusion", False):
        novelty = fuse_novelty_curves(data, weights, multiscale=multiscale, path_enhanced=path_enhanced)
    elif multiscale:
        novelty = multiscale_novelty_curve(fused)
    else:
        novelty = novelty_curve(fused)
    if options.get("candidate_rescore", False):
        novelty = candidate_rescored_novelty_curve(
            data,
            novelty,
            parse_stream_list(options.get("candidate_streams", list(weights))),
            multiscale=True,
            path_enhanced=path_enhanced,
            blend=options.get("candidate_blend"),
            selective=bool(options.get("candidate_selective", False)),
        )
    beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
    peak_profile = PEAK_PROFILES[options.get("peak_profile", "balanced")]
    pred_indices = pick_boundary_indices(novelty, beat_boundaries, **peak_profile)
    pred_boundaries = boundaries_from_indices(pred_indices, beat_boundaries)
    all_boundaries = np.r_[0.0, pred_boundaries, float(beat_boundaries[-1])]
    embeddings = segment_embedding(data, streams, all_boundaries)
    labels = label_segments(embeddings)
    return {
        "exp_id": exp_id,
        "weights": weights,
        "fused_ssm": fused,
        "novelty": novelty,
        "pred_indices": pred_indices,
        "pred_boundaries": pred_boundaries,
        "pred_labels": labels,
        "repetition_rate": label_repetition_rate(labels),
    }


def time_to_beat_index(times: np.ndarray, beat_boundaries: np.ndarray) -> np.ndarray:
    return np.searchsorted(beat_boundaries, times, side="left").astype(int)


def plot_track_infographic(
    out_path: Path,
    track_id: str,
    data: np.lib.npyio.NpzFile,
    result: dict,
    reference: ReferenceSegments | None,
    metrics_3s: BoundaryMetrics | None,
) -> None:
    beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
    duration = float(beat_boundaries[-1])
    pred_boundaries = np.asarray(result["pred_boundaries"], dtype=np.float32)
    novelty = np.asarray(result["novelty"], dtype=np.float32)
    fused = np.asarray(result["fused_ssm"], dtype=np.float32)
    beat_times = beat_boundaries[:-1]

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.75, 2.2, 1.25, 1.0])
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_ssm = fig.add_subplot(gs[1, :2])
    ax_weights = fig.add_subplot(gs[1, 2])
    ax_novelty = fig.add_subplot(gs[2, :])
    ax_segments = fig.add_subplot(gs[3, :])

    fig.suptitle(f"{track_id}: estructura predicha con {result['exp_id']}", fontsize=16, fontweight="bold")

    if reference and reference.segments:
        unique_labels = {label: idx for idx, label in enumerate(dict.fromkeys(reference.labels))}
        cmap = plt.get_cmap("tab20")
        for start, end, label in reference.segments:
            color = cmap(unique_labels[label] % 20)
            ax_timeline.axvspan(start, end, color=color, alpha=0.75)
            if end - start >= 6:
                ax_timeline.text((start + end) / 2, 0.5, label, ha="center", va="center", fontsize=8)
        for boundary in reference.internal_boundaries:
            ax_timeline.axvline(boundary, color="white", linewidth=0.8, alpha=0.85)
    ax_timeline.vlines(pred_boundaries, 0, 1, color="black", linewidth=1.2, alpha=0.85)
    ax_timeline.set_xlim(0, duration)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_title("Referencia AIST por secciones; lineas negras = fronteras predichas")

    im = ax_ssm.imshow(fused, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    for idx in time_to_beat_index(pred_boundaries, beat_boundaries):
        ax_ssm.axvline(idx, color="cyan", linewidth=0.75, alpha=0.7)
        ax_ssm.axhline(idx, color="cyan", linewidth=0.75, alpha=0.7)
    if reference:
        for idx in time_to_beat_index(reference.internal_boundaries, beat_boundaries):
            ax_ssm.axvline(idx, color="white", linewidth=0.5, alpha=0.6)
            ax_ssm.axhline(idx, color="white", linewidth=0.5, alpha=0.6)
    ax_ssm.set_title("SSM fusionada")
    ax_ssm.set_xlabel("beat interval")
    ax_ssm.set_ylabel("beat interval")
    fig.colorbar(im, ax=ax_ssm, fraction=0.046, pad=0.02)

    weights = result["weights"]
    ax_weights.barh(list(weights), list(weights.values()), color="#4C78A8")
    ax_weights.set_xlim(0, max(0.01, max(weights.values()) * 1.15))
    ax_weights.set_title("Pesos de fusion")
    ax_weights.grid(axis="x", alpha=0.25)
    if metrics_3s:
        text = (
            f"F@3s={metrics_3s.f_measure:.2f}\n"
            f"P={metrics_3s.precision:.2f} R={metrics_3s.recall:.2f}\n"
            f"pred={metrics_3s.pred_count} ref={metrics_3s.ref_count}"
        )
        ax_weights.text(0.02, -0.22, text, transform=ax_weights.transAxes, fontsize=10, va="top")

    ax_novelty.plot(beat_times[: len(novelty)], novelty, color="#222222", linewidth=1.5)
    for boundary in pred_boundaries:
        ax_novelty.axvline(boundary, color="#009E73", linewidth=1.0, alpha=0.85)
    if reference:
        for boundary in reference.internal_boundaries:
            ax_novelty.axvline(boundary, color="#D55E00", linewidth=0.8, alpha=0.55)
    ax_novelty.set_xlim(0, duration)
    ax_novelty.set_ylim(0, 1.05)
    ax_novelty.set_title("Novelty curve: verde = prediccion, naranja = referencia")
    ax_novelty.set_xlabel("tiempo (s)")
    ax_novelty.grid(alpha=0.25)

    pred_all = np.r_[0.0, pred_boundaries, duration]
    pred_labels = result["pred_labels"]
    cmap = plt.get_cmap("Set3")
    for idx, (start, end) in enumerate(zip(pred_all[:-1], pred_all[1:])):
        label = pred_labels[idx] if idx < len(pred_labels) else "?"
        ax_segments.axvspan(start, end, color=cmap(idx % 12), alpha=0.85)
        if end - start >= 7:
            ax_segments.text((start + end) / 2, 0.5, label, ha="center", va="center", fontsize=9)
    ax_segments.set_xlim(0, duration)
    ax_segments.set_ylim(0, 1)
    ax_segments.set_yticks([])
    ax_segments.set_title("Segmentos y etiquetas predichas")
    ax_segments.set_xlabel("tiempo (s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_dashboard(out_path: Path, rows: list[dict], experiments: list[str]) -> None:
    df = pd.DataFrame(rows)
    metrics = df[df["has_reference"]]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    grouped = metrics.groupby("exp_id")[["f_0p5", "f_3p0"]].mean().reindex(experiments)
    grouped.plot(kind="bar", ax=axes[0, 0], color=["#E45756", "#4C78A8"])
    axes[0, 0].set_title("F-measure promedio por experimento")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].tick_params(axis="x", rotation=45)

    pivot = metrics.pivot_table(index="track_id", columns="exp_id", values="f_3p0").reindex(columns=experiments)
    im = axes[0, 1].imshow(pivot.fillna(0).values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("F@3s por track y experimento")
    axes[0, 1].set_xticks(np.arange(len(experiments)))
    axes[0, 1].set_xticklabels(experiments, rotation=45, ha="right")
    axes[0, 1].set_yticks(np.arange(len(pivot.index)))
    axes[0, 1].set_yticklabels(pivot.index)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)

    main = metrics[metrics["exp_id"] == "E6_stm_content"]
    if not main.empty:
        axes[1, 0].scatter(main["ref_count"], main["pred_count"], s=50, color="#72B7B2")
        limit = max(main["ref_count"].max(), main["pred_count"].max()) + 2
        axes[1, 0].plot([0, limit], [0, limit], color="black", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlim(0, limit)
        axes[1, 0].set_ylim(0, limit)
        axes[1, 0].set_title("Conteo de fronteras: E6 vs referencia")
        axes[1, 0].set_xlabel("referencia")
        axes[1, 0].set_ylabel("prediccion")
        axes[1, 0].grid(alpha=0.25)
    else:
        axes[1, 0].axis("off")

    if {"E6_stm_content", "E7_stm_content_f0"}.issubset(set(metrics["exp_id"])):
        e6 = metrics[metrics["exp_id"] == "E6_stm_content"].set_index("track_id")
        e7 = metrics[metrics["exp_id"] == "E7_stm_content_f0"].set_index("track_id")
        joined = e6[["f_3p0", "f0_mean_confidence"]].join(
            e7[["f_3p0"]], lsuffix="_e6", rsuffix="_e7"
        )
        joined["delta"] = joined["f_3p0_e7"] - joined["f_3p0_e6"]
        colors = np.where(joined["delta"] >= 0, "#54A24B", "#E45756")
        axes[1, 1].bar(joined.index, joined["delta"], color=colors)
        axes[1, 1].axhline(0, color="black", linewidth=0.8)
        axes[1, 1].set_title("Impacto de agregar MELODIA gated: E7 - E6 en F@3s")
        axes[1, 1].set_ylabel("delta F@3s")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(axis="y", alpha=0.25)
    else:
        axes[1, 1].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_stm_impact(out_path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    metrics = df[df["has_reference"]]
    wide = metrics.pivot(index="track_id", columns="exp_id", values="f_3p0")
    if "E3_content" not in wide or "E6_stm_content" not in wide:
        return

    wide["delta_E6_E3"] = wide["E6_stm_content"] - wide["E3_content"]
    if "E8_adaptive_stm_content" in wide:
        wide["delta_E8_E3"] = wide["E8_adaptive_stm_content"] - wide["E3_content"]
    else:
        wide["delta_E8_E3"] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ordered = wide.sort_values("delta_E6_E3", ascending=False)
    colors = np.where(ordered["delta_E6_E3"] >= 0, "#54A24B", "#E45756")
    axes[0, 0].bar(ordered.index, ordered["delta_E6_E3"], color=colors)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Impacto de STM con fusion igualitaria: E6 - E3")
    axes[0, 0].set_ylabel("delta F@3s")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].scatter(wide["E3_content"], wide["E6_stm_content"], color="#4C78A8", s=50)
    axes[0, 1].plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("E6 STM+contenido vs E3 contenido")
    axes[0, 1].set_xlabel("E3 F@3s")
    axes[0, 1].set_ylabel("E6 F@3s")
    axes[0, 1].grid(alpha=0.25)

    if wide["delta_E8_E3"].notna().any():
        ordered_adaptive = wide.sort_values("delta_E8_E3", ascending=False)
        colors = np.where(ordered_adaptive["delta_E8_E3"] >= 0, "#54A24B", "#E45756")
        axes[1, 0].bar(ordered_adaptive.index, ordered_adaptive["delta_E8_E3"], color=colors)
        axes[1, 0].axhline(0, color="black", linewidth=0.8)
        axes[1, 0].set_title("Impacto de STM adaptativo: E8 - E3")
        axes[1, 0].set_ylabel("delta F@3s")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(axis="y", alpha=0.25)

        axes[1, 1].scatter(wide["E3_content"], wide["E8_adaptive_stm_content"], color="#F58518", s=50)
        axes[1, 1].plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title("E8 adaptativo vs E3 contenido")
        axes[1, 1].set_xlabel("E3 F@3s")
        axes[1, 1].set_ylabel("E8 F@3s")
        axes[1, 1].grid(alpha=0.25)
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    out_path: Path,
    rows: list[dict],
    figures_dir: Path,
    experiments: list[str],
) -> None:
    df = pd.DataFrame(rows)
    metrics = df[df["has_reference"]]
    grouped = metrics.groupby("exp_id").agg(
        f_0p5=("f_0p5", "mean"),
        f_3p0=("f_3p0", "mean"),
        precision_3p0=("precision_3p0", "mean"),
        recall_3p0=("recall_3p0", "mean"),
        pred_count=("pred_count", "mean"),
        ref_count=("ref_count", "mean"),
    )
    grouped = grouped.reindex(experiments)
    best = grouped["f_3p0"].idxmax()

    def metric_text(exp_id: str, label: str) -> str:
        if exp_id not in grouped.index or pd.isna(grouped.loc[exp_id, "f_3p0"]):
            return f"- {label}: not run."
        return f"- {label}: `{grouped.loc[exp_id, 'f_3p0']:.3f}`."
    table_cols = ["exp_id"] + list(grouped.columns)
    table_rows = []
    for exp_id, row in grouped.reset_index().iterrows():
        table_rows.append(
            [
                str(row["exp_id"]),
                *[
                    f"{float(row[col]):.3f}" if pd.notna(row[col]) else ""
                    for col in table_cols[1:]
                ],
            ]
        )
    markdown_table = [
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join(["---"] * len(table_cols)) + " |",
    ]
    markdown_table.extend("| " + " | ".join(row) + " |" for row in table_rows)
    figure_lines = ["## Figures", ""]
    dashboard_path = figures_dir / "dashboard_experiments.png"
    stm_path = figures_dir / "stm_impact.png"
    if dashboard_path.exists():
        figure_lines.extend(["![Dashboard](figures/dashboard_experiments.png)", ""])
    if stm_path.exists():
        figure_lines.extend(["![STM impact](figures/stm_impact.png)", ""])
    example_figures = sorted((figures_dir / "tracks").glob("*.png"))
    ranked_figures = sorted((figures_dir / "ranked_tracks").glob("*.png"))
    for figure in (example_figures[:1] or ranked_figures[:3]):
        figure_lines.extend([f"![{figure.stem}]({figure.relative_to(out_path.parent)})", ""])

    lines = [
        "# Downstream music-structure pilot results",
        "",
        "## Scope",
        "",
        "- Dataset: first 20 RWC-P audio files.",
        "- Reference: AIST `CHORUS` structural annotations from `rwc-annotations-archive`.",
        "- Method: SSM fusion, checkerboard novelty, beat-grid peak picking, segment clustering.",
        "",
        "## Aggregate boundary metrics",
        "",
        "\n".join(markdown_table),
        "",
        "## Main reading",
        "",
        f"- Best average F@3s: `{best}` with `{grouped.loc[best, 'f_3p0']:.3f}`.",
        metric_text("E1_mfcc", "MFCC baseline E1 F@3s"),
        metric_text("E3_content", "Content baseline E3 F@3s"),
        metric_text("E6_stm_content", "STM + content E6 F@3s"),
        metric_text("E8_adaptive_stm_content", "Adaptive late STM + content E8 F@3s"),
        metric_text("E10_mfcc_multiscale", "Multiscale MFCC E10 F@3s"),
        metric_text("E11_anchor_mfcc_multiscale", "MFCC-anchored multiscale STM/content E11 F@3s"),
        metric_text("E12_adaptive_multiscale_stm_content", "Adaptive multiscale STM/content E12 F@3s"),
        metric_text("E13_anchor_dense", "Dense MFCC-anchored STM/content E13 F@3s"),
        metric_text("E22_arrangement", "Arrangement proxy E22 F@3s"),
        metric_text("E24_bass_lowend", "Bass/low-end proxy E24 F@3s"),
        metric_text("E27_orthogonal_all", "Orthogonal-only fusion E27 F@3s"),
        metric_text("E28_mfcc_cens_arrangement", "MFCC+CENS+arrangement E28 F@3s"),
        metric_text("E31_learned_extended_loo", "Extended learned LOO E31 F@3s"),
        "",
        *figure_lines,
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def result_row(
    track_id: str,
    duration: float,
    reference: ReferenceSegments | None,
    exp_result: dict,
    metrics_05: BoundaryMetrics | None,
    metrics_30: BoundaryMetrics | None,
    f0_mean_confidence: float,
    f0_voiced_ratio: float,
) -> dict:
    pred_boundaries = np.asarray(exp_result["pred_boundaries"])
    row = {
        "track_id": track_id,
        "duration_s": round(duration, 3),
        "exp_id": exp_result["exp_id"],
        "weights": json.dumps(exp_result["weights"], sort_keys=True),
        "pred_count": int(len(pred_boundaries)),
        "pred_boundaries_s": json.dumps([round(float(x), 3) for x in pred_boundaries]),
        "pred_labels": " ".join(exp_result["pred_labels"]),
        "pred_repetition_rate": round(float(exp_result["repetition_rate"]), 6),
        "f0_mean_confidence": round(f0_mean_confidence, 6),
        "f0_voiced_ratio": round(f0_voiced_ratio, 6),
        "has_reference": reference is not None,
        "ref_count": int(len(reference.internal_boundaries)) if reference else 0,
    }
    if metrics_05 and metrics_30:
        row.update(
            {
                "precision_0p5": round(metrics_05.precision, 6),
                "recall_0p5": round(metrics_05.recall, 6),
                "f_0p5": round(metrics_05.f_measure, 6),
                "matches_0p5": metrics_05.matches,
                "precision_3p0": round(metrics_30.precision, 6),
                "recall_3p0": round(metrics_30.recall, 6),
                "f_3p0": round(metrics_30.f_measure, 6),
                "matches_3p0": metrics_30.matches,
                "median_abs_error_3p0": (
                    round(metrics_30.median_abs_error_s, 6)
                    if metrics_30.median_abs_error_s is not None
                    else ""
                ),
            }
        )
    else:
        row.update(
            {
                "precision_0p5": "",
                "recall_0p5": "",
                "f_0p5": "",
                "matches_0p5": "",
                "precision_3p0": "",
                "recall_3p0": "",
                "f_3p0": "",
                "matches_3p0": "",
                "median_abs_error_3p0": "",
            }
        )
    return row


def selected_experiments(requested: str | None) -> list[str]:
    if requested is None or requested.strip().lower() in {"", "all"}:
        return list(EXPERIMENTS)
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        known = ", ".join(EXPERIMENTS)
        raise SystemExit(f"Unknown experiments: {unknown}. Known experiments: {known}")
    return names


def discover_feature_paths(args: argparse.Namespace) -> list[Path]:
    feature_paths = sorted(args.features_dir.glob(args.feature_glob))
    if args.track_regex:
        pattern = re.compile(args.track_regex)
        feature_paths = [path for path in feature_paths if pattern.search(path.name)]
    if args.limit:
        feature_paths = feature_paths[: args.limit]
    return feature_paths


def aggregate_metric_rows(rows: list[dict], experiments: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = df[df["has_reference"]]
    if metrics.empty:
        return pd.DataFrame()
    grouped = metrics.groupby("exp_id").agg(
        f_0p5=("f_0p5", "mean"),
        f_3p0=("f_3p0", "mean"),
        precision_3p0=("precision_3p0", "mean"),
        recall_3p0=("recall_3p0", "mean"),
        pred_count=("pred_count", "mean"),
        ref_count=("ref_count", "mean"),
        tracks=("track_id", "nunique"),
    )
    return grouped.reindex(experiments).reset_index()


def best_experiment_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = df[df["has_reference"]].copy()
    if metrics.empty:
        return pd.DataFrame()
    idx = metrics.groupby("track_id")["f_3p0"].idxmax()
    return metrics.loc[idx].sort_values(["f_3p0", "track_id"], ascending=[False, True])


def simplex_weight_grid(streams: list[str], step: float) -> list[dict[str, float]]:
    """Generate non-negative weights that sum to one on a coarse simplex grid."""
    if step <= 0 or step > 1:
        raise ValueError("--learned-weight-step must be in (0, 1].")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--learned-weight-step must divide 1.0, e.g. 0.25, 0.2, 0.1, 0.05.")

    out: list[dict[str, float]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            values = prefix + [remaining]
            out.append({stream: value / units for stream, value in zip(streams, values) if value > 0})
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], units, len(streams))
    return out


def score_supervised_weights(
    feature_paths: list[Path],
    annotation_dir: Path,
    exp_id: str,
    weights: dict[str, float],
) -> tuple[float, float]:
    scores_3s = []
    scores_05s = []
    for feature_path in feature_paths:
        track_id = feature_path.name.replace("_features.npz", "")
        ref_path = reference_path_for_track(annotation_dir, track_id)
        if not ref_path:
            continue
        reference = parse_chorus_file(ref_path)
        with np.load(feature_path, allow_pickle=True) as data:
            result = run_experiment(data, exp_id, weights)
        scores_3s.append(boundary_metrics(result["pred_boundaries"], reference.internal_boundaries, 3.0).f_measure)
        scores_05s.append(boundary_metrics(result["pred_boundaries"], reference.internal_boundaries, 0.5).f_measure)
    if not scores_3s:
        return 0.0, 0.0
    return float(np.mean(scores_3s)), float(np.mean(scores_05s))


def score_supervised_weight_on_track(
    feature_path: Path,
    annotation_dir: Path,
    exp_id: str,
    weights: dict[str, float],
) -> tuple[str, float, float]:
    track_id = feature_path.name.replace("_features.npz", "")
    ref_path = reference_path_for_track(annotation_dir, track_id)
    if not ref_path:
        return track_id, np.nan, np.nan
    reference = parse_chorus_file(ref_path)
    with np.load(feature_path, allow_pickle=True) as data:
        result = run_experiment(data, exp_id, weights)
    f_3p0 = boundary_metrics(result["pred_boundaries"], reference.internal_boundaries, 3.0).f_measure
    f_0p5 = boundary_metrics(result["pred_boundaries"], reference.internal_boundaries, 0.5).f_measure
    return track_id, float(f_3p0), float(f_0p5)


def learn_supervised_ssm_weights(
    feature_paths: list[Path],
    annotation_dir: Path,
    exp_id: str,
    streams: list[str],
    step: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    candidates = simplex_weight_grid(streams, step)
    rows = []
    best_weights = candidates[0]
    best_score = (-1.0, -1.0)
    for weights in candidates:
        f_3p0, f_0p5 = score_supervised_weights(feature_paths, annotation_dir, exp_id, weights)
        rows.append(
            {
                "weights": json.dumps(weights, sort_keys=True),
                "f_3p0": round(f_3p0, 6),
                "f_0p5": round(f_0p5, 6),
                **{f"w_{stream}": round(float(weights.get(stream, 0.0)), 6) for stream in streams},
            }
        )
        score = (f_3p0, f_0p5)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, pd.DataFrame(rows).sort_values(["f_3p0", "f_0p5"], ascending=False)


def learn_loo_supervised_weights(
    feature_paths: list[Path],
    annotation_dir: Path,
    exp_id: str,
    streams: list[str],
    step: float,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    candidates = simplex_weight_grid(streams, step)
    candidate_rows = []
    for candidate_idx, weights in enumerate(candidates):
        weight_json = json.dumps(weights, sort_keys=True)
        for feature_path in feature_paths:
            track_id, f_3p0, f_0p5 = score_supervised_weight_on_track(feature_path, annotation_dir, exp_id, weights)
            candidate_rows.append(
                {
                    "candidate_idx": candidate_idx,
                    "track_id": track_id,
                    "weights": weight_json,
                    "f_3p0": f_3p0,
                    "f_0p5": f_0p5,
                    **{f"w_{stream}": round(float(weights.get(stream, 0.0)), 6) for stream in streams},
                }
            )
    candidate_scores = pd.DataFrame(candidate_rows)
    by_track: dict[str, dict[str, float]] = {}
    summary_rows = []
    for held_out in feature_paths:
        track_id = held_out.name.replace("_features.npz", "")
        train_scores = candidate_scores[candidate_scores["track_id"] != track_id]
        grid = (
            train_scores.groupby(["candidate_idx", "weights"], as_index=False)
            .agg(f_3p0=("f_3p0", "mean"), f_0p5=("f_0p5", "mean"))
            .sort_values(["f_3p0", "f_0p5"], ascending=False)
        )
        if grid.empty:
            weights = EXPERIMENTS[exp_id]
            best = {}
        else:
            best = grid.iloc[0].to_dict()
            weights = json.loads(str(best["weights"]))
        by_track[track_id] = weights
        summary_rows.append(
            {
                "held_out_track_id": track_id,
                **best,
                **{f"w_{stream}": round(float(weights.get(stream, 0.0)), 6) for stream in streams},
            }
        )
    return by_track, pd.DataFrame(summary_rows)


def write_run_manifest(
    out_path: Path,
    args: argparse.Namespace,
    experiments: list[str],
    feature_paths: list[Path],
    learned_weights_by_track: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    manifest = {
        "features_dir": str(args.features_dir),
        "feature_glob": args.feature_glob,
        "track_regex": args.track_regex,
        "annotation_dir": str(args.annotation_dir),
        "out_dir": str(args.out_dir),
        "limit": args.limit,
        "experiments": experiments,
        "plot_experiment": args.plot_experiment,
        "track_figures": args.track_figures,
        "ranked_track_figures": args.ranked_track_figures,
        "feature_count": len(feature_paths),
        "features": [str(path) for path in feature_paths],
        "learned_weight_step": args.learned_weight_step,
        "learned_streams": args.learned_streams,
        "learned_weights_by_track": learned_weights_by_track or {},
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def plot_ranked_track_figures(
    args: argparse.Namespace,
    rows: list[dict],
    experiments: list[str],
    count: int,
    learned_weights_by_track: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    if count <= 0 or args.plot_experiment not in experiments:
        return
    df = pd.DataFrame(rows)
    subset = df[(df["exp_id"] == args.plot_experiment) & (df["has_reference"])]
    if subset.empty:
        return
    selected = pd.concat(
        [
            subset.sort_values("f_3p0", ascending=False).head(count),
            subset.sort_values("f_3p0", ascending=True).head(count),
        ]
    ).drop_duplicates(subset=["track_id"])
    figures_dir = args.out_dir / "figures" / "ranked_tracks"
    for row in selected.itertuples(index=False):
        feature_path = args.features_dir / f"{row.track_id}_features.npz"
        if not feature_path.exists():
            continue
        with np.load(feature_path, allow_pickle=True) as data:
            ref_path = reference_path_for_track(args.annotation_dir, row.track_id)
            reference = parse_chorus_file(ref_path) if ref_path else None
            weights = EXPERIMENTS[args.plot_experiment]
            if EXPERIMENT_OPTIONS.get(args.plot_experiment, {}).get("supervised") and learned_weights_by_track:
                weights = learned_weights_by_track.get(args.plot_experiment, {}).get(row.track_id, weights)
            result = run_experiment(data, args.plot_experiment, weights)
            metrics_30 = (
                boundary_metrics(result["pred_boundaries"], reference.internal_boundaries, 3.0)
                if reference
                else None
            )
            plot_track_infographic(
                figures_dir / f"{row.track_id}_{args.plot_experiment}_structure.png",
                row.track_id,
                data,
                result,
                reference,
                metrics_30,
            )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("feature_outputs/rwc_p_20"))
    parser.add_argument("--feature-glob", default="*_features.npz")
    parser.add_argument("--track-regex", default=None, help="Optional regex filter over feature filenames.")
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/rwc-annotations-archive"))
    parser.add_argument("--out-dir", type=Path, default=Path("structure_outputs/rwc_p_20"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment IDs to run, or 'all'. Defaults to all registered experiments.",
    )
    parser.add_argument("--plot-experiment", default="E6_stm_content")
    parser.add_argument("--track-figures", action="store_true", help="Write one infographic per track.")
    parser.add_argument(
        "--ranked-track-figures",
        type=int,
        default=0,
        help="Write figures for the top N and bottom N tracks for --plot-experiment.",
    )
    parser.add_argument(
        "--learned-streams",
        default="stm,mfcc,chroma,cens",
        help="Comma-separated streams used by supervised SSM weight learning.",
    )
    parser.add_argument(
        "--learned-weight-step",
        type=float,
        default=0.1,
        help="Simplex grid step for supervised SSM weights. 0.1 gives 286 candidates for four streams.",
    )
    parser.add_argument("--skip-dashboard", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    tracks_fig_dir = figures_dir / "tracks"
    predictions_dir = args.out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    experiments = selected_experiments(args.experiments)
    if args.plot_experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown plot experiment: {args.plot_experiment}")

    feature_paths = discover_feature_paths(args)
    if not feature_paths:
        raise SystemExit(f"No feature files found in {args.features_dir}")

    learned_weights_by_track: dict[str, dict[str, dict[str, float]]] = {}
    supervised_experiments = [exp_id for exp_id in experiments if EXPERIMENT_OPTIONS.get(exp_id, {}).get("supervised")]
    for exp_id in supervised_experiments:
        options = EXPERIMENT_OPTIONS.get(exp_id, {})
        streams_spec = options.get("learned_streams", args.learned_streams)
        step = float(options.get("learned_weight_step", args.learned_weight_step))
        learned_streams = [stream.strip() for stream in streams_spec.split(",") if stream.strip()]
        unknown_streams = [stream for stream in learned_streams if stream not in STREAM_TO_SSM]
        if unknown_streams:
            known = ", ".join(STREAM_TO_SSM)
            raise SystemExit(f"Unknown learned streams: {unknown_streams}. Known streams: {known}")
        print(
            f"[{exp_id}] learning leave-one-out SSM weights over {len(feature_paths)} tracks "
            f"with streams={learned_streams} step={step}",
            flush=True,
        )
        by_track, learned_summary = learn_loo_supervised_weights(
            feature_paths,
            args.annotation_dir,
            exp_id,
            learned_streams,
            step,
        )
        learned_weights_by_track[exp_id] = by_track
        summary_name = "learned_ssm_loo_weights.csv" if exp_id == "E17_learned_ssm_loo" else f"learned_{exp_id}_weights.csv"
        learned_summary.to_csv(args.out_dir / summary_name, index=False)

    rows: list[dict] = []
    write_run_manifest(
        args.out_dir / "run_manifest.json",
        args,
        experiments,
        feature_paths,
        learned_weights_by_track=learned_weights_by_track,
    )
    for feature_path in feature_paths:
        track_id = feature_path.name.replace("_features.npz", "")
        print(f"[{track_id}] running {len(experiments)} experiments", flush=True)
        with np.load(feature_path, allow_pickle=True) as data:
            beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
            duration = float(beat_boundaries[-1])
            f0_mean_confidence = float(np.mean(np.asarray(data["f0_confidence"])))
            beat_f0 = np.asarray(data["beat_f0"])
            f0_voiced_ratio = float(np.mean(beat_f0[2] > 0.5)) if beat_f0.size else 0.0
            ref_path = reference_path_for_track(args.annotation_dir, track_id)
            reference = parse_chorus_file(ref_path) if ref_path else None

            track_predictions: list[dict] = []
            track_results: dict[str, dict] = {}
            for exp_id in experiments:
                weights = EXPERIMENTS[exp_id]
                if EXPERIMENT_OPTIONS.get(exp_id, {}).get("supervised"):
                    weights = learned_weights_by_track.get(exp_id, {}).get(track_id, weights)
                exp_result = run_experiment(data, exp_id, weights)
                track_results[exp_id] = exp_result
                metrics_05 = metrics_30 = None
                if reference:
                    metrics_05 = boundary_metrics(
                        exp_result["pred_boundaries"], reference.internal_boundaries, tolerance_s=0.5
                    )
                    metrics_30 = boundary_metrics(
                        exp_result["pred_boundaries"], reference.internal_boundaries, tolerance_s=3.0
                    )
                row = result_row(
                    track_id,
                    duration,
                    reference,
                    exp_result,
                    metrics_05,
                    metrics_30,
                    f0_mean_confidence,
                    f0_voiced_ratio,
                )
                rows.append(row)
                track_predictions.append(row)

            csv_path = predictions_dir / f"{track_id}_predictions.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(track_predictions[0]))
                writer.writeheader()
                writer.writerows(track_predictions)

            if args.track_figures and args.plot_experiment in track_results:
                plot_result = track_results[args.plot_experiment]
                metrics_30 = None
                if reference:
                    metrics_30 = boundary_metrics(plot_result["pred_boundaries"], reference.internal_boundaries, 3.0)
                plot_track_infographic(
                    tracks_fig_dir / f"{track_id}_{args.plot_experiment}_structure.png",
                    track_id,
                    data,
                    plot_result,
                    reference,
                    metrics_30,
                )

    csv_path = args.out_dir / "experiment_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "experiment_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    aggregate_metric_rows(rows, experiments).to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    best_experiment_rows(rows).to_csv(args.out_dir / "best_experiment_by_track.csv", index=False)

    if not args.skip_dashboard:
        plot_dashboard(figures_dir / "dashboard_experiments.png", rows, experiments)
        plot_stm_impact(figures_dir / "stm_impact.png", rows)
    plot_ranked_track_figures(
        args,
        rows,
        experiments,
        args.ranked_track_figures,
        learned_weights_by_track=learned_weights_by_track,
    )
    write_report(args.out_dir / "results_report.md", rows, figures_dir, experiments)
    print(f"Wrote metrics to {csv_path}")
    print(f"Wrote report to {args.out_dir / 'results_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
