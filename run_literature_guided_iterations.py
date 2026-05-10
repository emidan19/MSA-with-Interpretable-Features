#!/usr/bin/env python3
"""Literature-guided MSA experiments: learned similarity, global DP, stability fusion.

This script explores three ideas motivated by the current experiment report and
recent MSA literature:

1. Pairwise supervised metric learning for MFCC/density SSMs.
2. A global segmentation decoder that optimizes a song-level partition.
3. Hierarchical/stability fusion across features, methods, and novelty scales.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_similarity_metric_experiments import (
    BEAT_KEYS,
    SSM_KEYS,
    variant_stream_ssm,
    zscore_dims,
)
from run_similarity_metric_full_training import reliability_factor
from run_structure_baseline import (
    EXPERIMENTS,
    boundaries_from_indices,
    boundary_metrics,
    multiscale_novelty_curve,
    normalize_ssm,
    novelty_curve,
    parse_chorus_file,
    pick_boundary_indices,
    pulse_train_from_peaks,
    reference_path_for_track,
    run_experiment,
)


FEATURE_DIR = Path("feature_outputs/rwc_p_100_orthogonal_v1")
ANNOTATION_DIR = Path("data/rwc-annotations-archive")
OUT_DIR = Path("structure_outputs/rwc_p_100_literature_guided_v1")
E33_WEIGHTS = Path("structure_outputs/rwc_p_100_best_v1/learned_E33_reliability_gated_loo_weights.csv")
DENSITY_ROWS = Path("structure_outputs/rwc_p_100_selective_similarity_v1/R01_density_S02_selected_metrics.csv")
PREVIOUS_AGG = Path("reports/final_gating_report/final_experiment_catalog.csv")

METRIC_STREAMS = ("mfcc", "density")
STABILITY_STREAMS = ("stm", "mfcc", "cens", "arrangement", "density")


@dataclass(frozen=True)
class TrackInfo:
    track_id: str
    feature_path: Path
    fold: int
    beat_boundaries: np.ndarray
    reference_boundaries: np.ndarray
    label_ids: np.ndarray


@dataclass(frozen=True)
class MetricCandidate:
    candidate_id: str
    alpha: float
    density_weight: float


@dataclass(frozen=True)
class DPConfig:
    config_id: str
    target_s: float
    min_s: float
    max_s: float
    hom_w: float
    nov_w: float
    reg_w: float


@dataclass(frozen=True)
class StabilityConfig:
    config_id: str
    base_method: str
    blend: float
    use_dp_pulse: bool
    profile: str


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def reference_for_track(track_id: str):
    ref_path = reference_path_for_track(ANNOTATION_DIR, track_id)
    if not ref_path:
        return None
    return parse_chorus_file(ref_path)


def beat_label_ids(track_id: str, beat_boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = reference_for_track(track_id)
    n = max(0, len(beat_boundaries) - 1)
    label_ids = np.full(n, -1, dtype=np.int32)
    if reference is None:
        return label_ids, np.zeros(0, dtype=np.float32)
    centers = 0.5 * (beat_boundaries[:-1] + beat_boundaries[1:])
    label_map: dict[str, int] = {}
    for seg_idx, (start, end, label) in enumerate(reference.segments):
        norm = normalize_label(label)
        if norm not in label_map:
            label_map[norm] = len(label_map)
        if seg_idx == len(reference.segments) - 1:
            mask = (centers >= start) & (centers <= end)
        else:
            mask = (centers >= start) & (centers < end)
        label_ids[mask] = label_map[norm]
    return label_ids, reference.internal_boundaries.astype(np.float32)


def load_tracks(limit: int, folds: int) -> list[TrackInfo]:
    tracks: list[TrackInfo] = []
    for idx, feature_path in enumerate(sorted(FEATURE_DIR.glob("*_features.npz"))[:limit]):
        track_id = feature_path.name.replace("_features.npz", "")
        reference = reference_for_track(track_id)
        if reference is None:
            continue
        with np.load(feature_path, allow_pickle=True) as data:
            beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
        label_ids, ref_boundaries = beat_label_ids(track_id, beat_boundaries)
        tracks.append(
            TrackInfo(
                track_id=track_id,
                feature_path=feature_path,
                fold=idx % folds,
                beat_boundaries=beat_boundaries,
                reference_boundaries=ref_boundaries,
                label_ids=label_ids,
            )
        )
    return tracks


def pair_features(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    diff = x[i] - x[j]
    return np.hstack([np.abs(diff), diff * diff]).astype(np.float32)


def sample_pair_training_data(
    features: np.ndarray,
    label_ids: np.ndarray,
    rng: np.random.Generator,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = zscore_dims(features).T.astype(np.float32)
    valid = np.flatnonzero(label_ids >= 0)
    if valid.size < 4 or len(np.unique(label_ids[valid])) < 2:
        return np.zeros((0, x.shape[1] * 2), dtype=np.float32), np.zeros(0, dtype=np.int8)

    labels = np.unique(label_ids[valid])
    by_label = {int(label): valid[label_ids[valid] == label] for label in labels}
    pos_labels = [label for label, idxs in by_label.items() if idxs.size >= 2]
    if not pos_labels:
        return np.zeros((0, x.shape[1] * 2), dtype=np.float32), np.zeros(0, dtype=np.int8)

    pos_i = []
    pos_j = []
    neg_i = []
    neg_j = []
    for _ in range(samples_per_class):
        label = int(rng.choice(pos_labels))
        i, j = rng.choice(by_label[label], size=2, replace=False)
        pos_i.append(i)
        pos_j.append(j)

        left_label, right_label = rng.choice(labels, size=2, replace=False)
        i = int(rng.choice(by_label[int(left_label)]))
        j = int(rng.choice(by_label[int(right_label)]))
        neg_i.append(i)
        neg_j.append(j)

    pos_x = pair_features(x, np.asarray(pos_i), np.asarray(pos_j))
    neg_x = pair_features(x, np.asarray(neg_i), np.asarray(neg_j))
    y = np.r_[np.ones(len(pos_x), dtype=np.int8), np.zeros(len(neg_x), dtype=np.int8)]
    return np.vstack([pos_x, neg_x]), y


def precompute_pair_samples(
    tracks: list[TrackInfo],
    samples_per_class: int,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(20260507)
    samples: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {stream: {} for stream in METRIC_STREAMS}
    for idx, track in enumerate(tracks, 1):
        with np.load(track.feature_path, allow_pickle=True) as data:
            for stream in METRIC_STREAMS:
                x, y = sample_pair_training_data(
                    np.asarray(data[BEAT_KEYS[stream]], dtype=np.float32),
                    track.label_ids,
                    rng,
                    samples_per_class=samples_per_class,
                )
                samples[stream][track.track_id] = (x, y)
        if idx % 10 == 0:
            print(f"  pair samples: {idx}/{len(tracks)}", flush=True)
    return samples


def train_pair_model(
    samples: dict[str, tuple[np.ndarray, np.ndarray]],
    train_ids: list[str],
) -> object:
    xs = []
    ys = []
    for track_id in train_ids:
        x, y = samples[track_id]
        if x.size and y.size:
            xs.append(x)
            ys.append(y)
    if not xs:
        raise RuntimeError("No pair samples available for metric training")
    x_train = np.vstack(xs)
    y_train = np.concatenate(ys)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.7,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=20260507,
        ),
    )
    model.fit(x_train, y_train)
    return model


def learned_probability_ssm(model: object, features: np.ndarray, chunk_size: int = 180_000) -> np.ndarray:
    x = zscore_dims(features).T.astype(np.float32)
    n = x.shape[0]
    out = np.eye(n, dtype=np.float32)
    if n < 2:
        return out
    rows, cols = np.triu_indices(n, k=1)
    probs = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(rows), chunk_size):
        end = min(len(rows), start + chunk_size)
        feats = pair_features(x, rows[start:end], cols[start:end])
        probs[start:end] = model.predict_proba(feats)[:, 1].astype(np.float32)
    out[rows, cols] = probs
    out[cols, rows] = probs
    return normalize_ssm(out)


def fold_models(
    tracks: list[TrackInfo],
    samples: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    folds: int,
) -> dict[int, dict[str, object]]:
    models: dict[int, dict[str, object]] = {}
    for fold in range(folds):
        train_ids = [track.track_id for track in tracks if track.fold != fold]
        models[fold] = {}
        for stream in METRIC_STREAMS:
            print(f"  training pair metric: fold={fold} stream={stream}", flush=True)
            models[fold][stream] = train_pair_model(samples[stream], train_ids)
    return models


def e33_weights_by_track() -> dict[str, dict[str, float]]:
    df = pd.read_csv(E33_WEIGHTS)
    out = {}
    for row in df.itertuples(index=False):
        weights = {}
        for stream in STABILITY_STREAMS:
            col = f"w_{stream}"
            if hasattr(row, col):
                value = float(getattr(row, col))
                if value > 0:
                    weights[stream] = value
        out[row.held_out_track_id] = weights
    return out


def density_weights_by_track() -> dict[str, dict[str, float]]:
    df = pd.read_csv(DENSITY_ROWS)
    return {row.track_id: json.loads(row.weights) for row in df.itertuples(index=False)}


def reliability_fuse(matrices: dict[str, np.ndarray], base_weights: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
    scored = {}
    for key, matrix in matrices.items():
        scored[key] = float(base_weights.get(key, 0.0)) * reliability_factor(normalize_ssm(matrix))
    total = sum(value for value in scored.values() if value > 0)
    if total <= 1e-10:
        total_base = sum(max(float(value), 0.0) for value in base_weights.values())
        weights = {key: max(float(base_weights.get(key, 0.0)), 0.0) / max(total_base, 1e-10) for key in matrices}
    else:
        weights = {key: max(value, 0.0) / total for key, value in scored.items()}
    fused = np.zeros_like(next(iter(matrices.values())), dtype=np.float32)
    for key, weight in weights.items():
        fused += float(weight) * normalize_ssm(matrices[key])
    return normalize_ssm(fused), weights


def evaluate_indices(
    track: TrackInfo,
    pred_indices: np.ndarray,
    exp_id: str,
    extra: dict | None = None,
) -> dict:
    predicted = boundaries_from_indices(pred_indices, track.beat_boundaries)
    m05 = boundary_metrics(predicted, track.reference_boundaries, 0.5)
    m30 = boundary_metrics(predicted, track.reference_boundaries, 3.0)
    row = {
        "track_id": track.track_id,
        "exp_id": exp_id,
        "pred_count": int(len(predicted)),
        "ref_count": int(len(track.reference_boundaries)),
        "precision_0p5": m05.precision,
        "recall_0p5": m05.recall,
        "f_0p5": m05.f_measure,
        "precision_3p0": m30.precision,
        "recall_3p0": m30.recall,
        "f_3p0": m30.f_measure,
        "matches_3p0": m30.matches,
        "median_abs_error_3p0": m30.median_abs_error_s,
        "pred_indices": json.dumps([int(idx) for idx in pred_indices]),
        "pred_boundaries_s": json.dumps([float(value) for value in predicted]),
    }
    if extra:
        row.update(extra)
    return row


def evaluate_matrix(track: TrackInfo, matrix: np.ndarray, exp_id: str, extra: dict | None = None) -> dict:
    curve = novelty_curve(matrix)
    pred_indices = pick_boundary_indices(curve, track.beat_boundaries)
    return evaluate_indices(track, pred_indices, exp_id, extra=extra)


def aggregate(rows: pd.DataFrame | list[dict], id_col: str = "exp_id") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(id_col, as_index=False)
        .agg(
            f_0p5=("f_0p5", "mean"),
            f_3p0=("f_3p0", "mean"),
            precision_3p0=("precision_3p0", "mean"),
            recall_3p0=("recall_3p0", "mean"),
            pred_count=("pred_count", "mean"),
            ref_count=("ref_count", "mean"),
            tracks=("track_id", "nunique"),
        )
        .rename(columns={id_col: "exp_id"})
        .sort_values(["f_3p0", "f_0p5"], ascending=False)
    )


def metric_candidates() -> list[MetricCandidate]:
    candidates = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        for density_weight in (0.45, 0.55, 0.65, 0.75):
            candidates.append(
                MetricCandidate(
                    candidate_id=f"alpha{alpha:.2f}_dw{density_weight:.2f}",
                    alpha=float(alpha),
                    density_weight=float(density_weight),
                )
            )
    return candidates


def load_base_metric_matrices(data: np.lib.npyio.NpzFile, track_id: str) -> dict[str, np.ndarray]:
    return {
        "mfcc": normalize_ssm(np.asarray(data[SSM_KEYS["mfcc"]], dtype=np.float32)),
        "density": variant_stream_ssm("S02_stream_specific", "density", data, track_id, None),
    }


def run_metric_learning_phase(
    tracks: list[TrackInfo],
    models: dict[int, dict[str, object]],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict]]:
    candidates = metric_candidates()
    pd.DataFrame([asdict(candidate) for candidate in candidates]).to_csv(out_dir / "metric_learning_candidate_grid.csv", index=False)
    selected_rows = []
    candidate_rows = []
    selected_matrices: dict[str, np.ndarray] = {}
    selected_specs: dict[str, dict] = {}
    fixed_rows = []

    per_track_candidate_rows: dict[str, list[dict]] = {}
    per_track_candidate_mats: dict[tuple[str, str], np.ndarray] = {}
    for track_idx, track in enumerate(tracks, 1):
        with np.load(track.feature_path, allow_pickle=True) as data:
            base = load_base_metric_matrices(data, track.track_id)
            learned = {
                stream: learned_probability_ssm(models[track.fold][stream], np.asarray(data[BEAT_KEYS[stream]], dtype=np.float32))
                for stream in METRIC_STREAMS
            }
        local_rows = []
        for candidate in candidates:
            matrices = {
                stream: normalize_ssm((1.0 - candidate.alpha) * base[stream] + candidate.alpha * learned[stream])
                for stream in METRIC_STREAMS
            }
            fused, weights = reliability_fuse(
                matrices,
                {"mfcc": 1.0 - candidate.density_weight, "density": candidate.density_weight},
            )
            row = evaluate_matrix(
                track,
                fused,
                "ML_candidate_pair_label_metric",
                extra={
                    "candidate_id": candidate.candidate_id,
                    "alpha": candidate.alpha,
                    "density_weight": candidate.density_weight,
                    "weights": json.dumps(weights, sort_keys=True),
                },
            )
            local_rows.append(row)
            per_track_candidate_mats[(track.track_id, candidate.candidate_id)] = fused
            if candidate.alpha == 1.0 and abs(candidate.density_weight - 0.55) < 1e-6:
                fixed = dict(row)
                fixed["exp_id"] = "ML01_pair_label_metric_fixed"
                fixed_rows.append(fixed)
        per_track_candidate_rows[track.track_id] = local_rows
        candidate_rows.extend(local_rows)
        if track_idx % 10 == 0:
            print(f"  metric candidates scored: {track_idx}/{len(tracks)}", flush=True)

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(out_dir / "metric_learning_candidate_scores.csv", index=False)
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(out_dir / "ML01_pair_label_metric_fixed_metrics.csv", index=False)

    for track in tracks:
        train = candidate_df[candidate_df["track_id"] != track.track_id]
        grid = (
            train.groupby("candidate_id", as_index=False)
            .agg(f_3p0=("f_3p0", "mean"), f_0p5=("f_0p5", "mean"), pred_count=("pred_count", "mean"))
            .sort_values(["f_3p0", "f_0p5"], ascending=False)
        )
        best_id = str(grid.iloc[0]["candidate_id"])
        row = candidate_df[(candidate_df["track_id"] == track.track_id) & (candidate_df["candidate_id"] == best_id)].iloc[0].to_dict()
        row["exp_id"] = "ML02_pair_label_metric_blend_loo"
        row["train_f_3p0"] = float(grid.iloc[0]["f_3p0"])
        row["train_f_0p5"] = float(grid.iloc[0]["f_0p5"])
        selected_rows.append(row)
        selected_matrices[track.track_id] = per_track_candidate_mats[(track.track_id, best_id)]
        selected_specs[track.track_id] = {
            "candidate_id": best_id,
            "alpha": float(row["alpha"]),
            "density_weight": float(row["density_weight"]),
        }

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "ML02_pair_label_metric_blend_loo_metrics.csv", index=False)
    print("Metric-learning phase:", flush=True)
    print(aggregate(pd.concat([fixed_df, selected_df], ignore_index=True))[["exp_id", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    return pd.concat([fixed_df, selected_df], ignore_index=True), selected_matrices, selected_specs


def load_e33_matrix(track: TrackInfo, weights_by_track: dict[str, dict[str, float]]) -> np.ndarray:
    with np.load(track.feature_path, allow_pickle=True) as data:
        return np.asarray(run_experiment(data, "E33_reliability_gated_loo", weights_by_track[track.track_id])["fused_ssm"], dtype=np.float32)


def load_density_matrix(track: TrackInfo, weights_by_track: dict[str, dict[str, float]]) -> np.ndarray:
    with np.load(track.feature_path, allow_pickle=True) as data:
        matrices = {
            "mfcc": normalize_ssm(np.asarray(data[SSM_KEYS["mfcc"]], dtype=np.float32)),
            "density": variant_stream_ssm("S02_stream_specific", "density", data, track.track_id, None),
        }
    fused = np.zeros_like(next(iter(matrices.values())), dtype=np.float32)
    for stream, weight in weights_by_track[track.track_id].items():
        if stream in matrices:
            fused += float(weight) * normalize_ssm(matrices[stream])
    return normalize_ssm(fused)


def dense_candidate_boundaries(novelty: np.ndarray, beat_boundaries: np.ndarray) -> np.ndarray:
    if novelty.size < 8:
        return np.zeros(0, dtype=np.int64)
    duration = float(beat_boundaries[-1])
    beat_durations = np.diff(beat_boundaries)
    median_beat_s = float(np.median(beat_durations[beat_durations > 0])) if np.any(beat_durations > 0) else 0.5
    distance = max(3, int(round(3.5 / max(median_beat_s, 1e-3))))
    peaks, _ = find_peaks(novelty, height=float(np.quantile(novelty, 0.35)), prominence=max(0.01, float(np.std(novelty)) * 0.08), distance=distance)
    if peaks.size == 0:
        peaks = np.argsort(novelty)[::-1][: max(8, int(duration / 8.0))]
    max_count = max(18, int(duration / 4.5))
    peaks = peaks[np.argsort(novelty[peaks])[::-1]][:max_count]
    peaks = peaks[(peaks > 0) & (peaks < len(beat_boundaries) - 1)]
    regular = np.searchsorted(beat_boundaries, np.arange(8.0, duration - 8.0, 8.0)).astype(np.int64)
    candidates = np.unique(np.r_[peaks, regular])
    return candidates[(candidates > 0) & (candidates < len(beat_boundaries) - 1)]


def block_mean_integral(ssm: np.ndarray):
    csum = np.pad(ssm, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    diag_prefix = np.r_[0.0, np.cumsum(np.diag(ssm))]

    def mean(a: int, b: int) -> float:
        length = b - a
        if length <= 1:
            return 0.0
        total = csum[b, b] - csum[a, b] - csum[b, a] + csum[a, a]
        diag = diag_prefix[b] - diag_prefix[a]
        denom = length * length - length
        return float((total - diag) / max(denom, 1))

    return mean


def dp_segment_indices(
    ssm: np.ndarray,
    beat_boundaries: np.ndarray,
    config: DPConfig,
) -> np.ndarray:
    n = ssm.shape[0]
    if n < 12:
        return np.zeros(0, dtype=np.int64)
    curve = novelty_curve(ssm)
    internal = dense_candidate_boundaries(curve, beat_boundaries)
    candidates = np.unique(np.r_[0, internal, n])
    m = len(candidates)
    if m < 3:
        return pick_boundary_indices(curve, beat_boundaries)
    duration = float(beat_boundaries[-1])
    target_segments = int(np.clip(round(duration / config.target_s), 2, min(34, m - 1)))
    mean_fn = block_mean_integral(normalize_ssm(ssm))
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
            if 0 < a < n:
                boundary_bonus += float(curve[a])
            if 0 < b < n:
                boundary_bonus += float(curve[b])
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
        return pick_boundary_indices(curve, beat_boundaries)
    path = [m - 1]
    cur = m - 1
    for k in range(target_segments, 0, -1):
        cur = int(back[k, cur])
        if cur < 0:
            return pick_boundary_indices(curve, beat_boundaries)
        path.append(cur)
    path = list(reversed(path))
    boundaries = candidates[path[1:-1]]
    return np.asarray(boundaries, dtype=np.int64)


def dp_configs() -> list[DPConfig]:
    configs = []
    for target_s in (10.0, 12.0, 14.0, 16.0):
        for hom_w in (0.6, 1.0):
            for nov_w in (0.6, 1.2):
                for reg_w in (0.05, 0.20):
                    configs.append(
                        DPConfig(
                            config_id=f"t{target_s:.0f}_h{hom_w:.1f}_n{nov_w:.1f}_r{reg_w:.2f}",
                            target_s=target_s,
                            min_s=5.5,
                            max_s=42.0,
                            hom_w=hom_w,
                            nov_w=nov_w,
                            reg_w=reg_w,
                        )
                    )
    return configs


def run_dp_phase(
    tracks: list[TrackInfo],
    method_matrices: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    configs = dp_configs()
    pd.DataFrame([asdict(config) for config in configs]).to_csv(out_dir / "dp_candidate_grid.csv", index=False)
    candidate_rows = []
    selected_rows = []
    selected_indices: dict[str, dict[str, np.ndarray]] = {method: {} for method in method_matrices}
    for method_id, matrices_by_track in method_matrices.items():
        for track_idx, track in enumerate(tracks, 1):
            matrix = matrices_by_track[track.track_id]
            for config in configs:
                pred_indices = dp_segment_indices(matrix, track.beat_boundaries, config)
                candidate_rows.append(
                    evaluate_indices(
                        track,
                        pred_indices,
                        f"DP_candidate_{method_id}",
                        extra={"method_id": method_id, "config_id": config.config_id},
                    )
                )
            if track_idx % 10 == 0:
                print(f"  DP candidates: method={method_id} {track_idx}/{len(tracks)}", flush=True)
    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(out_dir / "dp_candidate_scores.csv", index=False)

    for method_id in method_matrices:
        local = candidate_df[candidate_df["method_id"] == method_id]
        for track in tracks:
            train = local[local["track_id"] != track.track_id]
            grid = (
                train.groupby("config_id", as_index=False)
                .agg(f_3p0=("f_3p0", "mean"), f_0p5=("f_0p5", "mean"), pred_count=("pred_count", "mean"))
                .sort_values(["f_3p0", "f_0p5"], ascending=False)
            )
            best_config = str(grid.iloc[0]["config_id"])
            row = local[(local["track_id"] == track.track_id) & (local["config_id"] == best_config)].iloc[0].to_dict()
            exp_id = f"DP_{method_id}_global_loo"
            row["exp_id"] = exp_id
            row["train_f_3p0"] = float(grid.iloc[0]["f_3p0"])
            row["train_f_0p5"] = float(grid.iloc[0]["f_0p5"])
            selected_rows.append(row)
            selected_indices[method_id][track.track_id] = np.asarray(json.loads(row["pred_indices"]), dtype=np.int64)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "dp_selected_metrics.csv", index=False)
    print("DP phase:", flush=True)
    print(aggregate(selected_df)[["exp_id", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    return selected_df, method_matrices, selected_indices


def stability_configs(method_ids: Iterable[str]) -> list[StabilityConfig]:
    configs = []
    for base_method in method_ids:
        for blend in (0.30, 0.45, 0.60):
            for use_dp_pulse in (False, True):
                for profile in ("balanced", "conservative"):
                    configs.append(
                        StabilityConfig(
                            config_id=f"{base_method}_b{blend:.2f}_dp{int(use_dp_pulse)}_{profile}",
                            base_method=base_method,
                            blend=blend,
                            use_dp_pulse=use_dp_pulse,
                            profile=profile,
                        )
                    )
    return configs


def method_stream_matrices(track: TrackInfo, ml_matrix: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(track.feature_path, allow_pickle=True) as data:
        return {
            "stm": normalize_ssm(np.asarray(data[SSM_KEYS["stm"]], dtype=np.float32)),
            "mfcc": normalize_ssm(np.asarray(data[SSM_KEYS["mfcc"]], dtype=np.float32)),
            "cens": normalize_ssm(np.asarray(data[SSM_KEYS["cens"]], dtype=np.float32)),
            "arrangement": normalize_ssm(np.asarray(data[SSM_KEYS["arrangement"]], dtype=np.float32)),
            "density": variant_stream_ssm("S02_stream_specific", "density", data, track.track_id, None),
            "ml_metric": normalize_ssm(ml_matrix),
        }


def stability_curve_for_track(
    track: TrackInfo,
    stream_mats: dict[str, np.ndarray],
    dp_indices: list[np.ndarray],
    use_dp_pulse: bool,
) -> np.ndarray:
    curves = []
    pulses = []
    for _name, matrix in stream_mats.items():
        for half_width in (8, 16, 32):
            if matrix.shape[0] < half_width * 2 + 4:
                continue
            curve = novelty_curve(matrix, half_width=half_width, smoothing_sigma=max(1.0, half_width / 8.0))
            curves.append(curve)
            peaks = pick_boundary_indices(curve, track.beat_boundaries, min_segment_s=5.5, target_segment_s=10.0)
            pulses.append(pulse_train_from_peaks(peaks, len(curve), sigma=1.2))
    if use_dp_pulse:
        for indices in dp_indices:
            if indices.size:
                pulses.append(pulse_train_from_peaks(indices, len(track.beat_boundaries) - 1, sigma=1.8))
    min_len = min([len(track.beat_boundaries) - 1, *[len(curve) for curve in curves + pulses]]) if curves or pulses else len(track.beat_boundaries) - 1
    if min_len <= 0:
        return np.zeros(0, dtype=np.float32)
    curve_part = np.mean([curve[:min_len] for curve in curves], axis=0) if curves else np.zeros(min_len, dtype=np.float32)
    pulse_part = np.mean([pulse[:min_len] for pulse in pulses], axis=0) if pulses else np.zeros(min_len, dtype=np.float32)
    stability = 0.45 * curve_part + 0.55 * pulse_part
    peak = float(np.max(stability)) if stability.size else 0.0
    return (stability / peak).astype(np.float32) if peak > 0 else stability.astype(np.float32)


def pick_from_profile(curve: np.ndarray, beat_boundaries: np.ndarray, profile: str) -> np.ndarray:
    if profile == "conservative":
        return pick_boundary_indices(curve, beat_boundaries, min_segment_s=10.0, target_segment_s=15.0)
    return pick_boundary_indices(curve, beat_boundaries)


def run_stability_phase(
    tracks: list[TrackInfo],
    method_matrices: dict[str, dict[str, np.ndarray]],
    dp_indices: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> pd.DataFrame:
    configs = stability_configs(method_matrices.keys())
    pd.DataFrame([asdict(config) for config in configs]).to_csv(out_dir / "stability_candidate_grid.csv", index=False)
    candidate_rows = []
    stability_cache: dict[tuple[str, bool], np.ndarray] = {}
    for track_idx, track in enumerate(tracks, 1):
        stream_mats = method_stream_matrices(track, method_matrices["ml_metric"][track.track_id])
        local_dp = [by_track.get(track.track_id, np.zeros(0, dtype=np.int64)) for by_track in dp_indices.values()]
        for use_dp in (False, True):
            stability_cache[(track.track_id, use_dp)] = stability_curve_for_track(track, stream_mats, local_dp, use_dp)
        for config in configs:
            base_curve = multiscale_novelty_curve(method_matrices[config.base_method][track.track_id])
            stability = stability_cache[(track.track_id, config.use_dp_pulse)]
            min_len = min(len(base_curve), len(stability))
            if min_len <= 0:
                pred_indices = np.zeros(0, dtype=np.int64)
            else:
                curve = (1.0 - config.blend) * base_curve[:min_len] + config.blend * stability[:min_len]
                curve = gaussian_filter1d(curve.astype(np.float32), sigma=1.0)
                peak = float(np.max(curve)) if curve.size else 0.0
                if peak > 0:
                    curve = curve / peak
                pred_indices = pick_from_profile(curve, track.beat_boundaries, config.profile)
            candidate_rows.append(
                evaluate_indices(
                    track,
                    pred_indices,
                    "H_candidate_stability",
                    extra={
                        "config_id": config.config_id,
                        "base_method": config.base_method,
                        "blend": config.blend,
                        "use_dp_pulse": config.use_dp_pulse,
                        "profile": config.profile,
                    },
                )
            )
        if track_idx % 10 == 0:
            print(f"  stability candidates: {track_idx}/{len(tracks)}", flush=True)
    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(out_dir / "stability_candidate_scores.csv", index=False)

    selected_rows = []
    for track in tracks:
        train = candidate_df[candidate_df["track_id"] != track.track_id]
        grid = (
            train.groupby("config_id", as_index=False)
            .agg(f_3p0=("f_3p0", "mean"), f_0p5=("f_0p5", "mean"), pred_count=("pred_count", "mean"))
            .sort_values(["f_3p0", "f_0p5"], ascending=False)
        )
        best_id = str(grid.iloc[0]["config_id"])
        row = candidate_df[(candidate_df["track_id"] == track.track_id) & (candidate_df["config_id"] == best_id)].iloc[0].to_dict()
        row["exp_id"] = "H01_hierarchical_stability_loo"
        row["train_f_3p0"] = float(grid.iloc[0]["f_3p0"])
        row["train_f_0p5"] = float(grid.iloc[0]["f_0p5"])
        selected_rows.append(row)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "stability_selected_metrics.csv", index=False)
    print("Stability phase:", flush=True)
    print(aggregate(selected_df)[["exp_id", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    return selected_df


def previous_comparison_rows() -> pd.DataFrame:
    if not PREVIOUS_AGG.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREVIOUS_AGG)
    wanted = {
        "E33_previous": "baseline_previous",
        "DENSITY_S02_previous": "baseline_previous",
        "G21_ridge_boundary_conf_pair": "baseline_previous",
        "ORACLE_E33_density_recomputed": "oracle_previous",
        "ORACLE_candidate_pool": "oracle_previous",
    }
    rows = []
    for exp_id, family in wanted.items():
        local = df[df["exp_id"] == exp_id]
        if local.empty:
            continue
        row = local.sort_values("f_3p0", ascending=False).iloc[0].to_dict()
        row["phase"] = family
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    shown = comparison.copy().sort_values("f_3p0", ascending=True)
    colors = []
    for exp_id in shown["exp_id"]:
        if "ORACLE" in exp_id.upper():
            colors.append("#7C3AED")
        elif exp_id.startswith("G21"):
            colors.append("#059669")
        elif exp_id.startswith("ML"):
            colors.append("#3B6EA8")
        elif exp_id.startswith("DP"):
            colors.append("#D97706")
        elif exp_id.startswith("H"):
            colors.append("#DB2777")
        else:
            colors.append("#64748B")
    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, shown["f_3p0"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shown["exp_id"], fontsize=8)
    ax.set_xlabel("F@3s")
    ax.set_title("Ideas guiadas por literatura: resultados agregados")
    ax.grid(axis="x", alpha=0.25)
    lo = max(0.50, float(shown["f_3p0"].min()) - 0.02)
    hi = min(0.73, float(shown["f_3p0"].max()) + 0.02)
    ax.set_xlim(lo, hi)
    for i, row in shown.reset_index(drop=True).iterrows():
        ax.text(float(row["f_3p0"]) + 0.002, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary(out_dir: Path, comparison: pd.DataFrame, metric_df: pd.DataFrame, dp_df: pd.DataFrame, stability_df: pd.DataFrame) -> None:
    best_new = comparison[comparison["phase"].isin(["metric_learning", "global_decoder", "hierarchical_stability"])].sort_values(["f_3p0", "f_0p5"], ascending=False).iloc[0]
    g21 = comparison[comparison["exp_id"] == "G21_ridge_boundary_conf_pair"]
    g21_f3 = float(g21.iloc[0]["f_3p0"]) if not g21.empty else np.nan
    lines = [
        "# Iteracion guiada por literatura: metric learning, DP global y estabilidad jerarquica",
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
    lines.extend(
        [
            "",
            "## Lectura por idea",
            "",
            f"- Mejor idea nueva: `{best_new.exp_id}` con F@3s={best_new.f_3p0:.3f} "
            f"({best_new.f_3p0 - g21_f3:+.3f} vs G21 previo)." if not np.isnan(g21_f3) else f"- Mejor idea nueva: `{best_new.exp_id}` con F@3s={best_new.f_3p0:.3f}.",
            "- Idea 1 aprende una similitud de MFCC/density a partir de pares de beats con igual/distinta etiqueta estructural. Es una version pequena del enfoque SSM/metric-learning, sin cambiar el extractor de audio.",
            "- Idea 2 reemplaza el peak-picking local por una particion global con DP: cada segmento debe tener homogeneidad interna, bordes con novelty alta y duracion razonable.",
            "- Idea 3 usa persistencia jerarquica: una frontera gana confianza si aparece en varias features, escalas de novelty y decoders.",
            "",
            "## Candidatos frecuentes",
            "",
        ]
    )
    if "candidate_id" in metric_df:
        counts = metric_df[metric_df["exp_id"] == "ML02_pair_label_metric_blend_loo"]["candidate_id"].value_counts().head(8)
        lines.append("Metric learning LOO:")
        for key, value in counts.items():
            lines.append(f"- {value} tracks: `{key}`.")
    if "config_id" in dp_df:
        lines.append("")
        lines.append("DP global LOO:")
        if "method_id" in dp_df:
            for method_id, sub in dp_df.groupby("method_id"):
                counts = sub["config_id"].value_counts().head(4)
                for key, value in counts.items():
                    lines.append(f"- {method_id}: {value} tracks con `{key}`.")
        else:
            counts = dp_df["config_id"].value_counts().head(8)
            for key, value in counts.items():
                lines.append(f"- {value} tracks: `{key}`.")
    if "config_id" in stability_df:
        counts = stability_df["config_id"].value_counts().head(8)
        lines.append("")
        lines.append("Estabilidad jerarquica LOO:")
        for key, value in counts.items():
            lines.append(f"- {value} tracks: `{key}`.")
    lines.extend(
        [
            "",
            "## Decision operativa",
            "",
            "Si alguna idea nueva supera a G21, conviene hacer una segunda iteracion ampliando solo esa rama. Si no lo supera, la conclusion fuerte es que el margen oracular no se cierra con mas postprocesamiento sobre las SSM actuales: habria que entrenar representaciones/metricas con targets mas ricos o sumar datos anotados como Harmonix/SALAMI.",
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
    tracks = load_tracks(args.limit, args.folds)
    if not tracks:
        raise SystemExit("No tracks with annotations were found.")
    print(f"Loaded {len(tracks)} tracks in {args.folds} folds", flush=True)

    print("Idea 1: pair-label metric learning for MFCC/density", flush=True)
    samples = precompute_pair_samples(tracks, args.samples_per_class)
    models = fold_models(tracks, samples, args.folds)
    metric_df, ml_matrices, metric_specs = run_metric_learning_phase(tracks, models, args.out_dir)

    print("Preparing reference matrices for DP/stability", flush=True)
    e33_w = e33_weights_by_track()
    density_w = density_weights_by_track()
    method_matrices: dict[str, dict[str, np.ndarray]] = {"e33": {}, "density_s02": {}, "ml_metric": {}}
    for idx, track in enumerate(tracks, 1):
        method_matrices["e33"][track.track_id] = load_e33_matrix(track, e33_w)
        method_matrices["density_s02"][track.track_id] = load_density_matrix(track, density_w)
        method_matrices["ml_metric"][track.track_id] = ml_matrices[track.track_id]
        if idx % 10 == 0:
            print(f"  prepared matrices: {idx}/{len(tracks)}", flush=True)

    print("Idea 2: global DP segmentation decoder", flush=True)
    dp_df, method_matrices, dp_indices = run_dp_phase(tracks, method_matrices, args.out_dir)

    print("Idea 3: hierarchical stability after observing ideas 1 and 2", flush=True)
    stability_df = run_stability_phase(tracks, method_matrices, dp_indices, args.out_dir)

    metric_agg = aggregate(metric_df)
    metric_agg["phase"] = "metric_learning"
    dp_agg = aggregate(dp_df)
    dp_agg["phase"] = "global_decoder"
    stability_agg = aggregate(stability_df)
    stability_agg["phase"] = "hierarchical_stability"
    previous = previous_comparison_rows()
    if not previous.empty:
        previous = previous[["exp_id", "f_0p5", "f_3p0", "precision_3p0", "recall_3p0", "pred_count", "tracks", "phase"]]
    comparison = pd.concat([previous, metric_agg, dp_agg, stability_agg], ignore_index=True, sort=False)
    comparison = comparison.sort_values(["f_3p0", "f_0p5"], ascending=False).reset_index(drop=True)
    comparison.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    plot_comparison(comparison, args.out_dir / "aggregate_metrics.png")
    write_summary(args.out_dir, comparison, metric_df, dp_df, stability_df)

    manifest = {
        "features_dir": str(FEATURE_DIR),
        "annotation_dir": str(ANNOTATION_DIR),
        "limit": args.limit,
        "folds": args.folds,
        "samples_per_class": args.samples_per_class,
        "tracks": [track.track_id for track in tracks],
        "metric_specs": metric_specs,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Final comparison:", flush=True)
    print(comparison[["exp_id", "phase", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count", "tracks"]].to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
