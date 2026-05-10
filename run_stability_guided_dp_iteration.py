#!/usr/bin/env python3
"""Stability-guided global decoding for learned SSM music structure detection.

This iteration keeps the strongest branch from the previous run:

- learned MFCC/density SSMs from pair-label metric learning
- global dynamic-programming segmentation

and uses hierarchical stability as an explicit confidence signal, both as a
boundary prior in the DP objective and as a precision-oriented post-filter.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
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
from scipy.signal import find_peaks

import run_literature_guided_iterations as lit
from run_similarity_metric_experiments import BEAT_KEYS
from run_structure_baseline import boundaries_from_indices, normalize_ssm, novelty_curve


OUT_DIR = Path("structure_outputs/rwc_p_100_stability_guided_dp_v1")
PREV_LIT_AGG = Path("structure_outputs/rwc_p_100_literature_guided_v1/aggregate_metrics.csv")
G21_ROWS = Path("structure_outputs/rwc_p_100_boundary_metric_gating_v1/gating_selector_metrics.csv")


@dataclass(frozen=True)
class StabilityDPConfig:
    config_id: str
    target_s: float
    min_s: float
    max_s: float
    hom_w: float
    nov_w: float
    reg_w: float
    stability_w: float


@dataclass(frozen=True)
class ConfidenceFilterConfig:
    config_id: str
    blend: float
    quantile: float
    min_segment_s: float
    snap_radius_beats: int
    rescue_gap_s: float


def norm_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
    curve = np.maximum(curve, 0.0)
    peak = float(np.max(curve)) if curve.size else 0.0
    return (curve / peak).astype(np.float32) if peak > 0 else curve.astype(np.float32)


def learned_metric_matrix(track: lit.TrackInfo, models: dict[int, dict[str, object]]) -> tuple[np.ndarray, dict[str, float]]:
    """Rebuild the best previous ML02 SSM with pre-registered alpha/density weight."""
    alpha = 0.75
    density_weight = 0.65
    with np.load(track.feature_path, allow_pickle=True) as data:
        base = lit.load_base_metric_matrices(data, track.track_id)
        learned = {
            stream: lit.learned_probability_ssm(
                models[track.fold][stream],
                np.asarray(data[BEAT_KEYS[stream]], dtype=np.float32),
            )
            for stream in lit.METRIC_STREAMS
        }
    matrices = {
        stream: normalize_ssm((1.0 - alpha) * base[stream] + alpha * learned[stream])
        for stream in lit.METRIC_STREAMS
    }
    return lit.reliability_fuse(matrices, {"mfcc": 1.0 - density_weight, "density": density_weight})


def build_ml_matrices(
    tracks: list[lit.TrackInfo],
    models: dict[int, dict[str, object]],
    out_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    matrices: dict[str, np.ndarray] = {}
    rows = []
    for idx, track in enumerate(tracks, 1):
        matrix, weights = learned_metric_matrix(track, models)
        matrices[track.track_id] = matrix
        rows.append(
            lit.evaluate_matrix(
                track,
                matrix,
                "ML03_pair_metric_preregistered",
                extra={"weights": json.dumps(weights, sort_keys=True), "alpha": 0.75, "density_weight": 0.65},
            )
        )
        if idx % 10 == 0:
            print(f"  rebuilt learned SSMs: {idx}/{len(tracks)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ML03_pair_metric_preregistered_metrics.csv", index=False)
    return matrices, df


def stability_curves_for_tracks(
    tracks: list[lit.TrackInfo],
    ml_matrices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    curves = {}
    for idx, track in enumerate(tracks, 1):
        stream_mats = lit.method_stream_matrices(track, ml_matrices[track.track_id])
        curves[track.track_id] = lit.stability_curve_for_track(
            track,
            stream_mats,
            dp_indices=[],
            use_dp_pulse=False,
        )
        if idx % 10 == 0:
            print(f"  hierarchical stability curves: {idx}/{len(tracks)}", flush=True)
    return curves


def curve_peaks(curve: np.ndarray, beat_boundaries: np.ndarray, quantile: float, distance_s: float) -> np.ndarray:
    if curve.size < 8:
        return np.zeros(0, dtype=np.int64)
    beat_durations = np.diff(beat_boundaries)
    median_beat_s = float(np.median(beat_durations[beat_durations > 0])) if np.any(beat_durations > 0) else 0.5
    distance = max(3, int(round(distance_s / max(median_beat_s, 1e-3))))
    peaks, _ = find_peaks(
        curve,
        height=float(np.quantile(curve, quantile)),
        prominence=max(0.01, float(np.std(curve)) * 0.08),
        distance=distance,
    )
    return peaks.astype(np.int64)


def confidence_curve(novelty: np.ndarray, stability: np.ndarray, stability_w: float) -> np.ndarray:
    min_len = min(len(novelty), len(stability))
    if min_len <= 0:
        return np.zeros(0, dtype=np.float32)
    confidence = (1.0 - stability_w) * norm_curve(novelty[:min_len]) + stability_w * norm_curve(stability[:min_len])
    return norm_curve(gaussian_filter1d(confidence.astype(np.float32), sigma=1.0))


def guided_candidate_boundaries(
    novelty: np.ndarray,
    stability: np.ndarray,
    beat_boundaries: np.ndarray,
    stability_w: float,
) -> np.ndarray:
    duration = float(beat_boundaries[-1])
    confidence = confidence_curve(novelty, stability, stability_w)
    regular = np.searchsorted(beat_boundaries, np.arange(8.0, duration - 8.0, 8.0)).astype(np.int64)
    candidates = [
        lit.dense_candidate_boundaries(novelty, beat_boundaries),
        curve_peaks(stability, beat_boundaries, quantile=0.42, distance_s=4.5),
        curve_peaks(confidence, beat_boundaries, quantile=0.35, distance_s=3.5),
        regular,
    ]
    merged = np.unique(np.concatenate([arr for arr in candidates if arr.size]))
    return merged[(merged > 0) & (merged < len(beat_boundaries) - 1)]


def stability_dp_configs() -> list[StabilityDPConfig]:
    configs = []
    for target_s in (14.0, 16.0):
        for hom_w in (0.6, 1.0):
            for nov_w in (0.9, 1.2):
                for stability_w in (0.20, 0.40, 0.60):
                    configs.append(
                        StabilityDPConfig(
                            config_id=f"t{target_s:.0f}_h{hom_w:.1f}_n{nov_w:.1f}_r0.20_s{stability_w:.2f}",
                            target_s=target_s,
                            min_s=5.5,
                            max_s=42.0,
                            hom_w=hom_w,
                            nov_w=nov_w,
                            reg_w=0.20,
                            stability_w=stability_w,
                        )
                    )
    return configs


def stability_guided_dp_indices(
    ssm: np.ndarray,
    beat_boundaries: np.ndarray,
    stability: np.ndarray,
    config: StabilityDPConfig,
) -> np.ndarray:
    n = ssm.shape[0]
    if n < 12:
        return np.zeros(0, dtype=np.int64)
    novelty = novelty_curve(ssm)
    stability = norm_curve(stability[: len(novelty)])
    confidence = confidence_curve(novelty, stability, config.stability_w)
    internal = guided_candidate_boundaries(novelty, stability, beat_boundaries, config.stability_w)
    candidates = np.unique(np.r_[0, internal, n]).astype(np.int64)
    m = len(candidates)
    if m < 3:
        return lit.dp_segment_indices(
            ssm,
            beat_boundaries,
            lit.DPConfig(
                "fallback",
                config.target_s,
                config.min_s,
                config.max_s,
                config.hom_w,
                config.nov_w,
                config.reg_w,
            ),
        )

    duration = float(beat_boundaries[-1])
    target_segments = int(np.clip(round(duration / config.target_s), 2, min(34, m - 1)))
    mean_fn = lit.block_mean_integral(normalize_ssm(ssm))
    seg_score = np.full((m, m), -1e9, dtype=np.float32)
    for left in range(m - 1):
        a = int(candidates[left])
        for right in range(left + 1, m):
            b = int(candidates[right])
            length_s = float(beat_boundaries[b] - beat_boundaries[a])
            if length_s < config.min_s or length_s > config.max_s:
                continue
            hom = mean_fn(a, b)
            boundary_bonus = 0.0
            if 0 < a < n and a < len(confidence):
                boundary_bonus += float(confidence[a])
            if 0 < b < n and b < len(confidence):
                boundary_bonus += float(confidence[b])
            reg = -abs(np.log(max(length_s, 1e-3) / config.target_s))
            seg_score[left, right] = config.hom_w * hom + config.nov_w * boundary_bonus + config.reg_w * reg

    dp = np.full((target_segments + 1, m), -1e9, dtype=np.float32)
    back = np.full((target_segments + 1, m), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for k in range(1, target_segments + 1):
        for right in range(1, m):
            values = dp[k - 1, :right] + seg_score[:right, right]
            best_left = int(np.argmax(values))
            if values[best_left] > -1e8:
                dp[k, right] = values[best_left]
                back[k, right] = best_left
    if dp[target_segments, m - 1] <= -1e8:
        return lit.dp_segment_indices(
            ssm,
            beat_boundaries,
            lit.DPConfig("fallback", config.target_s, config.min_s, config.max_s, config.hom_w, config.nov_w, config.reg_w),
        )

    path = [m - 1]
    cur = m - 1
    for k in range(target_segments, 0, -1):
        cur = int(back[k, cur])
        if cur < 0:
            return np.zeros(0, dtype=np.int64)
        path.append(cur)
    path = list(reversed(path))
    return np.asarray(candidates[path[1:-1]], dtype=np.int64)


def fixed_previous_dp_indices(ssm: np.ndarray, beat_boundaries: np.ndarray) -> np.ndarray:
    return lit.dp_segment_indices(
        ssm,
        beat_boundaries,
        lit.DPConfig("t14_h1.0_n1.2_r0.20", 14.0, 5.5, 42.0, 1.0, 1.2, 0.20),
    )


def run_guided_dp_phase(
    tracks: list[lit.TrackInfo],
    ml_matrices: dict[str, np.ndarray],
    stability_curves: dict[str, np.ndarray],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = stability_dp_configs()
    pd.DataFrame([asdict(config) for config in configs]).to_csv(out_dir / "stability_guided_dp_config_grid.csv", index=False)
    candidate_rows = []
    fixed_rows = []
    for idx, track in enumerate(tracks, 1):
        matrix = ml_matrices[track.track_id]
        fixed = fixed_previous_dp_indices(matrix, track.beat_boundaries)
        fixed_rows.append(
            lit.evaluate_indices(
                track,
                fixed,
                "DP_ml_metric_fixed_previous_config",
                extra={"source_config_id": "t14_h1.0_n1.2_r0.20"},
            )
        )
        for config in configs:
            pred = stability_guided_dp_indices(matrix, track.beat_boundaries, stability_curves[track.track_id], config)
            candidate_rows.append(
                lit.evaluate_indices(
                    track,
                    pred,
                    "SGDP_candidate_stability_prior",
                    extra={"config_id": config.config_id},
                )
            )
        if idx % 10 == 0:
            print(f"  stability-guided DP candidates: {idx}/{len(tracks)}", flush=True)
    candidate_df = pd.DataFrame(candidate_rows)
    fixed_df = pd.DataFrame(fixed_rows)
    candidate_df.to_csv(out_dir / "stability_guided_dp_candidate_scores.csv", index=False)
    fixed_df.to_csv(out_dir / "DP_ml_metric_fixed_previous_config_metrics.csv", index=False)

    selected = loo_select(candidate_df, group_cols=["config_id"], exp_id="SGDP01_stability_prior_dp_loo")
    selected.to_csv(out_dir / "SGDP01_stability_prior_dp_loo_metrics.csv", index=False)
    print("Stability-prior DP phase:", flush=True)
    print(
        lit.aggregate(pd.concat([fixed_df, selected], ignore_index=True))[
            ["exp_id", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]
        ].to_string(index=False),
        flush=True,
    )
    return candidate_df, fixed_df


def confidence_filter_configs() -> list[ConfidenceFilterConfig]:
    configs = []
    for blend in (0.45, 0.65, 0.80):
        for quantile in (0.45, 0.55, 0.65):
            for min_segment_s in (8.5, 10.0):
                configs.append(
                    ConfidenceFilterConfig(
                        config_id=f"b{blend:.2f}_q{quantile:.2f}_m{min_segment_s:.1f}_r4_g40",
                        blend=blend,
                        quantile=quantile,
                        min_segment_s=min_segment_s,
                        snap_radius_beats=4,
                        rescue_gap_s=40.0,
                    )
                )
    return configs


def enforce_min_distance(
    indices: np.ndarray,
    scores: np.ndarray,
    beat_boundaries: np.ndarray,
    min_segment_s: float,
) -> np.ndarray:
    if indices.size == 0:
        return indices.astype(np.int64)
    beat_durations = np.diff(beat_boundaries)
    median_beat_s = float(np.median(beat_durations[beat_durations > 0])) if np.any(beat_durations > 0) else 0.5
    min_distance = max(4, int(round(min_segment_s / max(median_beat_s, 1e-3))))
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    for pos in order:
        idx = int(indices[pos])
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)
    return np.asarray(sorted(selected), dtype=np.int64)


def snap_and_filter_indices(
    indices: np.ndarray,
    confidence: np.ndarray,
    beat_boundaries: np.ndarray,
    config: ConfidenceFilterConfig,
) -> np.ndarray:
    n = len(beat_boundaries) - 1
    if indices.size == 0 or confidence.size == 0:
        return indices.astype(np.int64)
    valid_conf = confidence[1 : min(n, len(confidence))]
    threshold = float(np.quantile(valid_conf, config.quantile)) if valid_conf.size else 0.0
    snapped = []
    for idx in np.asarray(indices, dtype=np.int64):
        left = max(1, int(idx) - config.snap_radius_beats)
        right = min(min(n - 1, len(confidence) - 1), int(idx) + config.snap_radius_beats)
        if right < left:
            continue
        local = np.arange(left, right + 1)
        snapped.append(int(local[np.argmax(confidence[local])]))
    if not snapped:
        return np.zeros(0, dtype=np.int64)
    snapped = np.unique(np.asarray(snapped, dtype=np.int64))
    scores = confidence[snapped]
    keep = snapped[scores >= threshold]
    keep_scores = confidence[keep] if keep.size else np.zeros(0, dtype=np.float32)
    keep = enforce_min_distance(keep, keep_scores, beat_boundaries, config.min_segment_s)

    # Rescue very long gaps with one high-confidence boundary, preserving precision.
    changed = True
    rescue_round = 0
    rescue_threshold = float(np.quantile(valid_conf, max(0.35, config.quantile - 0.15))) if valid_conf.size else 0.0
    while changed and rescue_round < 4:
        rescue_round += 1
        changed = False
        previous_keep = keep.copy()
        sequence = np.r_[0, keep, n]
        additions = []
        for left_idx, right_idx in zip(sequence[:-1], sequence[1:]):
            gap_s = float(beat_boundaries[right_idx] - beat_boundaries[left_idx])
            if gap_s <= config.rescue_gap_s:
                continue
            left_t = float(beat_boundaries[left_idx]) + config.min_segment_s
            right_t = float(beat_boundaries[right_idx]) - config.min_segment_s
            if right_t <= left_t:
                continue
            left = int(np.searchsorted(beat_boundaries, left_t))
            right = int(np.searchsorted(beat_boundaries, right_t))
            right = min(right, len(confidence) - 1, n - 1)
            if right <= left:
                continue
            local = np.arange(left, right + 1)
            best = int(local[np.argmax(confidence[local])])
            if confidence[best] >= rescue_threshold:
                additions.append(best)
        if additions:
            merged = np.unique(np.r_[keep, additions]).astype(np.int64)
            keep = enforce_min_distance(merged, confidence[merged], beat_boundaries, config.min_segment_s)
            changed = not np.array_equal(keep, previous_keep)
    return keep.astype(np.int64)


def filter_candidate_rows(
    tracks: list[lit.TrackInfo],
    base_rows: pd.DataFrame,
    ml_matrices: dict[str, np.ndarray],
    stability_curves: dict[str, np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    configs = confidence_filter_configs()
    pd.DataFrame([asdict(config) for config in configs]).to_csv(out_dir / "confidence_filter_config_grid.csv", index=False)
    rows = []
    track_by_id = {track.track_id: track for track in tracks}
    for row_idx, base in enumerate(base_rows.itertuples(index=False), 1):
        track = track_by_id[base.track_id]
        novelty = novelty_curve(ml_matrices[track.track_id])
        stability = stability_curves[track.track_id]
        pred_indices = np.asarray(json.loads(base.pred_indices), dtype=np.int64)
        source_config = getattr(base, "config_id", getattr(base, "source_config_id", "fixed_previous"))
        for config in configs:
            confidence = confidence_curve(novelty, stability, config.blend)
            filtered = snap_and_filter_indices(pred_indices, confidence, track.beat_boundaries, config)
            rows.append(
                lit.evaluate_indices(
                    track,
                    filtered,
                    "SGDP_filter_candidate",
                    extra={
                        "source_exp_id": base.exp_id,
                        "source_config_id": source_config,
                        "filter_config_id": config.config_id,
                        "blend": config.blend,
                        "quantile": config.quantile,
                        "min_segment_s": config.min_segment_s,
                    },
                )
            )
        if row_idx % 100 == 0:
            print(f"  confidence filter source rows: {row_idx}/{len(base_rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "confidence_filter_candidate_scores.csv", index=False)
    return df


def loo_select(
    candidate_df: pd.DataFrame,
    group_cols: list[str],
    exp_id: str,
    precision_margin: float | None = None,
) -> pd.DataFrame:
    selected = []
    for track_id in sorted(candidate_df["track_id"].unique()):
        train = candidate_df[candidate_df["track_id"] != track_id]
        grid = (
            train.groupby(group_cols, as_index=False)
            .agg(
                f_3p0=("f_3p0", "mean"),
                f_0p5=("f_0p5", "mean"),
                precision_3p0=("precision_3p0", "mean"),
                recall_3p0=("recall_3p0", "mean"),
                pred_count=("pred_count", "mean"),
            )
        )
        if precision_margin is not None:
            best_f = float(grid["f_3p0"].max())
            grid = grid[grid["f_3p0"] >= best_f - precision_margin]
            grid = grid.sort_values(["precision_3p0", "f_0p5", "f_3p0"], ascending=False)
        else:
            grid = grid.sort_values(["f_3p0", "f_0p5"], ascending=False)
        best = grid.iloc[0]
        mask = candidate_df["track_id"] == track_id
        for col in group_cols:
            mask &= candidate_df[col] == best[col]
        row = candidate_df[mask].iloc[0].to_dict()
        row["exp_id"] = exp_id
        row["train_f_3p0"] = float(best["f_3p0"])
        row["train_f_0p5"] = float(best["f_0p5"])
        row["train_precision_3p0"] = float(best["precision_3p0"])
        selected.append(row)
    return pd.DataFrame(selected)


def fold_select(
    candidate_df: pd.DataFrame,
    tracks: list[lit.TrackInfo],
    group_cols: list[str],
    exp_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_folds = {track.track_id: track.fold for track in tracks}
    df = candidate_df.copy()
    df["fold"] = df["track_id"].map(track_folds)
    selected = []
    chosen = []
    for fold in sorted(df["fold"].dropna().unique()):
        train = df[df["fold"] != fold]
        grid = (
            train.groupby(group_cols, as_index=False)
            .agg(
                f_3p0=("f_3p0", "mean"),
                f_0p5=("f_0p5", "mean"),
                precision_3p0=("precision_3p0", "mean"),
                recall_3p0=("recall_3p0", "mean"),
                pred_count=("pred_count", "mean"),
            )
            .sort_values(["f_3p0", "f_0p5"], ascending=False)
        )
        best = grid.iloc[0]
        chosen.append(
            {
                "fold": int(fold),
                **{col: best[col] for col in group_cols},
                "train_f_3p0": float(best["f_3p0"]),
                "train_f_0p5": float(best["f_0p5"]),
                "train_precision_3p0": float(best["precision_3p0"]),
            }
        )
        holdout = df[df["fold"] == fold]
        mask = pd.Series(True, index=holdout.index)
        for col in group_cols:
            mask &= holdout[col] == best[col]
        rows = holdout[mask].copy()
        rows["exp_id"] = exp_id
        rows["train_f_3p0"] = float(best["f_3p0"])
        rows["train_f_0p5"] = float(best["f_0p5"])
        rows["train_precision_3p0"] = float(best["precision_3p0"])
        selected.append(rows)
    selected_df = pd.concat(selected, ignore_index=True).drop(columns=["fold"], errors="ignore")
    return selected_df, pd.DataFrame(chosen)


def write_candidate_aggregates(
    out_dir: Path,
    guided_candidates: pd.DataFrame,
    filter_candidates: pd.DataFrame,
) -> None:
    guided = (
        guided_candidates.groupby("config_id", as_index=False)
        .agg(
            f_3p0=("f_3p0", "mean"),
            f_0p5=("f_0p5", "mean"),
            precision_3p0=("precision_3p0", "mean"),
            recall_3p0=("recall_3p0", "mean"),
            pred_count=("pred_count", "mean"),
            tracks=("track_id", "nunique"),
        )
        .sort_values(["f_3p0", "f_0p5"], ascending=False)
    )
    guided.to_csv(out_dir / "stability_guided_dp_candidate_aggregates.csv", index=False)
    filtered = (
        filter_candidates.groupby(["source_config_id", "filter_config_id"], as_index=False)
        .agg(
            f_3p0=("f_3p0", "mean"),
            f_0p5=("f_0p5", "mean"),
            precision_3p0=("precision_3p0", "mean"),
            recall_3p0=("recall_3p0", "mean"),
            pred_count=("pred_count", "mean"),
            tracks=("track_id", "nunique"),
        )
        .sort_values(["f_3p0", "f_0p5"], ascending=False)
    )
    filtered.to_csv(out_dir / "confidence_filter_candidate_aggregates.csv", index=False)
    if not guided.empty:
        best_id = str(guided.iloc[0]["config_id"])
        guided_candidates[guided_candidates["config_id"] == best_id].assign(
            exp_id="SGDP06_stability_prior_dp_best_fixed_diagnostic"
        ).to_csv(out_dir / "SGDP06_stability_prior_dp_best_fixed_diagnostic_metrics.csv", index=False)
    if not filtered.empty:
        best = filtered.iloc[0]
        mask = (filter_candidates["source_config_id"] == best["source_config_id"]) & (
            filter_candidates["filter_config_id"] == best["filter_config_id"]
        )
        filter_candidates[mask].assign(exp_id="SGDP07_confidence_filter_best_fixed_diagnostic").to_csv(
            out_dir / "SGDP07_confidence_filter_best_fixed_diagnostic_metrics.csv",
            index=False,
        )


def aggregate_with_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    agg = lit.aggregate(df)
    agg["phase"] = phase
    return agg


def previous_rows() -> pd.DataFrame:
    previous = lit.previous_comparison_rows()
    rows = []
    if not previous.empty:
        rows.append(previous[["exp_id", "f_0p5", "f_3p0", "precision_3p0", "recall_3p0", "pred_count", "tracks", "phase"]])
    if PREV_LIT_AGG.exists():
        df = pd.read_csv(PREV_LIT_AGG)
        keep = df[df["exp_id"].isin(["DP_ml_metric_global_loo", "H01_hierarchical_stability_loo"])]
        if not keep.empty:
            keep = keep.copy()
            keep["phase"] = "previous_iteration"
            rows.append(keep[["exp_id", "f_0p5", "f_3p0", "precision_3p0", "recall_3p0", "pred_count", "tracks", "phase"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def comparison_vs_g21(out_dir: Path, method_frames: list[pd.DataFrame]) -> None:
    if not G21_ROWS.exists():
        return
    gdf = pd.read_csv(G21_ROWS)
    g21 = gdf[gdf["selector_exp_id"] == "G21_ridge_boundary_conf_pair"][["track_id", "f_3p0"]].rename(
        columns={"f_3p0": "g21_f_3p0"}
    )
    rows = []
    for frame in method_frames:
        for exp_id, sub in frame.groupby("exp_id"):
            comp = sub[["track_id", "f_3p0"]].merge(g21, on="track_id")
            if comp.empty:
                continue
            diff = comp["f_3p0"] - comp["g21_f_3p0"]
            rows.append(
                {
                    "exp_id": exp_id,
                    "mean_diff_vs_g21": float(diff.mean()),
                    "median_diff_vs_g21": float(diff.median()),
                    "wins_vs_g21": int((diff > 1e-9).sum()),
                    "ties_vs_g21": int((diff.abs() <= 1e-9).sum()),
                    "losses_vs_g21": int((diff < -1e-9).sum()),
                    "best_track_gain": float(diff.max()),
                    "worst_track_loss": float(diff.min()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "comparison_vs_g21.csv", index=False)


def plot_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    shown = comparison.sort_values("f_3p0", ascending=True)
    colors = []
    for exp_id in shown["exp_id"]:
        if "ORACLE" in exp_id.upper():
            colors.append("#7C3AED")
        elif exp_id.startswith("G21"):
            colors.append("#059669")
        elif exp_id.startswith("SGDP"):
            colors.append("#C2410C")
        elif exp_id.startswith("DP"):
            colors.append("#D97706")
        elif exp_id.startswith("H"):
            colors.append("#DB2777")
        elif exp_id.startswith("ML"):
            colors.append("#3B6EA8")
        else:
            colors.append("#64748B")
    fig, ax = plt.subplots(figsize=(12.0, 7.2), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, shown["f_3p0"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shown["exp_id"], fontsize=8)
    ax.set_xlabel("F@3s")
    ax.set_title("SSM aprendida + DP global con estabilidad como confianza")
    ax.grid(axis="x", alpha=0.25)
    lo = max(0.50, float(shown["f_3p0"].min()) - 0.02)
    hi = min(0.73, float(shown["f_3p0"].max()) + 0.02)
    ax.set_xlim(lo, hi)
    for i, row in shown.reset_index(drop=True).iterrows():
        ax.text(float(row["f_3p0"]) + 0.002, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    comparison: pd.DataFrame,
    filter_selected: pd.DataFrame,
    guided_selected: pd.DataFrame,
    guided_fold: pd.DataFrame,
    filter_fold: pd.DataFrame,
) -> None:
    lines = [
        "# Iteracion: SSM aprendida + DP global con estabilidad como confianza",
        "",
        "## Resultado agregado",
        "",
        "| Metodo | F@3s | F@0.5s | Precision@3s | Recall@3s | Pred. prom. | Fase |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison.sort_values(["f_3p0", "f_0p5"], ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.exp_id} | {row.f_3p0:.3f} | {row.f_0p5:.3f} | "
            f"{row.precision_3p0:.3f} | {row.recall_3p0:.3f} | {row.pred_count:.2f} | {row.phase} |"
        )
    lines.extend(["", "## Lectura", ""])
    new_mask = comparison["phase"].astype(str).str.contains("stability_guided_dp|confidence_filter")
    best_new = comparison[new_mask].sort_values(["f_3p0", "f_0p5"], ascending=False).iloc[0]
    lines.append(f"- Mejor variante nueva: `{best_new.exp_id}` con F@3s={best_new.f_3p0:.3f}.")
    lines.append("- El prior de estabilidad dentro del DP prueba si la frontera debe ser globalmente coherente y ademas estar soportada por varias features/escalas.")
    lines.append("- El filtro de confianza prueba una lectura mas conservadora: conservar fronteras del DP solo cuando coinciden con estabilidad jerarquica.")
    if not guided_fold.empty:
        agg = lit.aggregate(guided_fold).iloc[0]
        lines.append(f"- Seleccion por fold del DP con prior: F@3s={agg.f_3p0:.3f}, F@0.5s={agg.f_0p5:.3f}.")
    if not filter_fold.empty:
        agg = lit.aggregate(filter_fold).iloc[0]
        lines.append(f"- Seleccion por fold del filtro: F@3s={agg.f_3p0:.3f}, precision@3s={agg.precision_3p0:.3f}.")
    if not guided_selected.empty and "config_id" in guided_selected:
        lines.extend(["", "## Configuracion DP prior frecuente"])
        for key, value in guided_selected["config_id"].value_counts().head(6).items():
            lines.append(f"- {value} tracks: `{key}`.")
    if not filter_selected.empty and "filter_config_id" in filter_selected:
        lines.extend(["", "## Configuracion filtro frecuente"])
        for key, value in filter_selected["filter_config_id"].value_counts().head(8).items():
            lines.append(f"- {value} tracks: `{key}`.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Si el filtro mejora precision/F@0.5 sin perder demasiado F@3s, conviene tratar la estabilidad como una salida de confianza: fronteras fuertes para la decision principal y fronteras debiles como candidatas secundarias.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=650)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = lit.load_tracks(args.limit, args.folds)
    print(f"Loaded {len(tracks)} tracks in {args.folds} folds", flush=True)

    print("Training pair-label metric models", flush=True)
    samples = lit.precompute_pair_samples(tracks, args.samples_per_class)
    models = lit.fold_models(tracks, samples, args.folds)

    print("Rebuilding learned MFCC/density SSMs", flush=True)
    ml_matrices, ml_df = build_ml_matrices(tracks, models, args.out_dir)

    print("Computing hierarchical stability confidence curves", flush=True)
    stability_curves = stability_curves_for_tracks(tracks, ml_matrices)

    print("Running stability-prior global DP", flush=True)
    guided_candidates, fixed_dp_df = run_guided_dp_phase(tracks, ml_matrices, stability_curves, args.out_dir)
    guided_selected = loo_select(guided_candidates, ["config_id"], "SGDP01_stability_prior_dp_loo")
    guided_selected.to_csv(args.out_dir / "SGDP01_stability_prior_dp_loo_metrics.csv", index=False)

    print("Running stability confidence filter over DP candidates", flush=True)
    filter_sources = pd.concat(
        [
            fixed_dp_df.assign(config_id="fixed_previous"),
            guided_candidates,
        ],
        ignore_index=True,
        sort=False,
    )
    filter_candidates = filter_candidate_rows(tracks, filter_sources, ml_matrices, stability_curves, args.out_dir)
    filter_selected = loo_select(
        filter_candidates,
        ["source_config_id", "filter_config_id"],
        "SGDP02_confidence_filter_loo",
    )
    filter_selected.to_csv(args.out_dir / "SGDP02_confidence_filter_loo_metrics.csv", index=False)
    precision_selected = loo_select(
        filter_candidates,
        ["source_config_id", "filter_config_id"],
        "SGDP03_precision_confidence_filter_loo",
        precision_margin=0.006,
    )
    precision_selected.to_csv(args.out_dir / "SGDP03_precision_confidence_filter_loo_metrics.csv", index=False)
    guided_fold, guided_fold_configs = fold_select(
        guided_candidates,
        tracks,
        ["config_id"],
        "SGDP04_stability_prior_dp_foldcv",
    )
    guided_fold.to_csv(args.out_dir / "SGDP04_stability_prior_dp_foldcv_metrics.csv", index=False)
    guided_fold_configs.to_csv(args.out_dir / "SGDP04_stability_prior_dp_foldcv_chosen_configs.csv", index=False)
    filter_fold, filter_fold_configs = fold_select(
        filter_candidates,
        tracks,
        ["source_config_id", "filter_config_id"],
        "SGDP05_confidence_filter_foldcv",
    )
    filter_fold.to_csv(args.out_dir / "SGDP05_confidence_filter_foldcv_metrics.csv", index=False)
    filter_fold_configs.to_csv(args.out_dir / "SGDP05_confidence_filter_foldcv_chosen_configs.csv", index=False)
    write_candidate_aggregates(args.out_dir, guided_candidates, filter_candidates)

    print("Final aggregation:", flush=True)
    frames = [
        aggregate_with_phase(ml_df, "learned_ssm"),
        aggregate_with_phase(fixed_dp_df, "previous_config_recomputed"),
        aggregate_with_phase(guided_selected, "stability_guided_dp"),
        aggregate_with_phase(guided_fold, "stability_guided_dp_foldcv"),
        aggregate_with_phase(filter_selected, "confidence_filter"),
        aggregate_with_phase(precision_selected, "confidence_filter"),
        aggregate_with_phase(filter_fold, "confidence_filter_foldcv"),
    ]
    previous = previous_rows()
    comparison = pd.concat([previous, *frames], ignore_index=True, sort=False)
    comparison = comparison.sort_values(["f_3p0", "f_0p5"], ascending=False).reset_index(drop=True)
    comparison.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    plot_comparison(comparison, args.out_dir / "aggregate_metrics.png")
    comparison_vs_g21(
        args.out_dir,
        [ml_df, fixed_dp_df, guided_selected, guided_fold, filter_selected, precision_selected, filter_fold],
    )
    write_summary(args.out_dir, comparison, filter_selected, guided_selected, guided_fold, filter_fold)

    print(comparison[["exp_id", "phase", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
