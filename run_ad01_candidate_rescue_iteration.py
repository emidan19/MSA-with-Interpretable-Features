#!/usr/bin/env python3
"""AD01-anchored boundary rescue using the strongest exploratory signals.

This iteration keeps AD01_PR01 as the base system and uses the previous
exploratory branches as local evidence rather than complete replacements:

- I02 statistical novelty
- I03 multi-scale SSM prior
- I05 non-linear learned/AD01 gate
- I04 DP margin as an extra confidence/support flag

The candidate selector is trained out-of-fold at boundary level. Final configs
are selected by fold, so the reported score is a conservative continuation of
the previous evaluation style.
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_ad01_learned_fusion_experiment as fus
import run_disruptive_ideas_experiments as dis
import run_literature_guided_iterations as lit
import run_stability_guided_dp_iteration as sgdp
from run_structure_baseline import novelty_curve


OUT_DIR = Path("structure_outputs/rwc_p_100_ad01_candidate_rescue_v1")
DISRUPTIVE_DIR = Path("structure_outputs/rwc_p_100_disruptive_ideas_v1")

SOURCE_FILES = {
    "ad01": "AD01_PR01_sgdp_preregistered_metrics.csv",
    "i01": "I01_state_decoder_foldcv_metrics.csv",
    "i02": "I02_statistical_novelty_foldcv_metrics.csv",
    "i03": "I03_multiscale_ssm_foldcv_metrics.csv",
    "i04": "I04_dp_margin_foldcv_metrics.csv",
    "i05": "I05_nonlinear_gate_foldcv_metrics.csv",
}


@dataclass(frozen=True)
class SelectionConfig:
    config_id: str
    strategy: str
    model_id: str
    min_segment_s: float
    ad01_bonus: float
    target_delta: int = 0
    snap_radius_s: float = 0.0
    rescue_gap_s: float = 0.0
    max_extra: int = 0
    min_quantile: float = 0.0


def parse_indices(value: str) -> np.ndarray:
    try:
        return np.asarray(json.loads(value), dtype=np.int64)
    except Exception:
        return np.zeros(0, dtype=np.int64)


def median_beat_s(beat_boundaries: np.ndarray) -> float:
    diff = np.diff(beat_boundaries)
    valid = diff[diff > 0]
    return float(np.median(valid)) if valid.size else 0.5


def boundary_label(idx: int, beat_boundaries: np.ndarray, reference: np.ndarray, tolerance_s: float) -> int:
    if idx <= 0 or idx >= len(beat_boundaries) - 1 or reference.size == 0:
        return 0
    time_s = float(beat_boundaries[idx])
    return int(np.any(np.abs(reference - time_s) <= tolerance_s))


def nearest_time_distance(indices: np.ndarray, idx: int, beat_boundaries: np.ndarray) -> float:
    if indices.size == 0:
        return 999.0
    idx = int(np.clip(idx, 0, len(beat_boundaries) - 1))
    times = beat_boundaries[np.clip(indices.astype(np.int64), 0, len(beat_boundaries) - 1)]
    return float(np.min(np.abs(times - float(beat_boundaries[idx]))))


def curve_value(curve: np.ndarray, idx: int) -> float:
    if curve.size == 0:
        return 0.0
    return float(curve[int(np.clip(idx, 0, len(curve) - 1))])


def local_curve_max(curve: np.ndarray, idx: int, radius: int) -> float:
    if curve.size == 0:
        return 0.0
    lo = max(0, int(idx) - radius)
    hi = min(len(curve), int(idx) + radius + 1)
    return float(np.max(curve[lo:hi])) if hi > lo else 0.0


def load_source_predictions(base_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for source, filename in SOURCE_FILES.items():
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["source"] = source
        out[source] = df
    return out


def predictions_by_source(source_frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for source, df in source_frames.items():
        for row in df.itertuples(index=False):
            out[(row.track_id, source)] = parse_indices(row.pred_indices)
    return out


def ad01_rows(source_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return source_frames["ad01"].copy()


def build_ad01_context(tracks: list[lit.TrackInfo]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices = {}
    ad_conf = {}
    stat_curves = {}
    multi_curves = {}
    print("Building AD01 and auxiliary confidence curves", flush=True)
    for idx, track in enumerate(tracks, 1):
        matrix, _weights = fus.ad01_matrix(track)
        matrices[track.track_id] = matrix
        stability = sgdp.stability_curves_for_tracks([track], {track.track_id: matrix})[track.track_id]
        conf = sgdp.confidence_curve(novelty_curve(matrix), stability, fus.AD01_CONFIG.stability_w)
        ad_conf[track.track_id] = sgdp.norm_curve(conf)
        stat_curves[track.track_id] = dis.statistical_curve(track)
        multi_curves[track.track_id] = dis.multiscale_ssm_curve(track)
        if idx % 10 == 0:
            print(f"  context tracks: {idx}/{len(tracks)}", flush=True)
    return matrices, ad_conf, stat_curves, multi_curves


def candidate_pool_for_track(track: lit.TrackInfo, pred_lookup: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    n = len(track.beat_boundaries) - 1
    pool: set[int] = set()
    for source in SOURCE_FILES:
        indices = pred_lookup.get((track.track_id, source), np.zeros(0, dtype=np.int64))
        for idx in indices:
            idx = int(idx)
            if 0 < idx < n:
                pool.add(idx)
            if source in {"ad01", "i02", "i03", "i05"}:
                for shift in (-2, -1, 1, 2):
                    shifted = idx + shift
                    if 0 < shifted < n:
                        pool.add(int(shifted))
    return np.asarray(sorted(pool), dtype=np.int64)


def build_candidate_features(
    tracks: list[lit.TrackInfo],
    pred_lookup: dict[tuple[str, str], np.ndarray],
    ad_conf: dict[str, np.ndarray],
    stat_curves: dict[str, np.ndarray],
    multi_curves: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for track_idx, track in enumerate(tracks, 1):
        pool = candidate_pool_for_track(track, pred_lookup)
        beat_tol = max(2, int(round(3.0 / max(median_beat_s(track.beat_boundaries), 1e-3))))
        radius_local = max(2, int(round(1.5 / max(median_beat_s(track.beat_boundaries), 1e-3))))
        source_indices = {source: pred_lookup.get((track.track_id, source), np.zeros(0, dtype=np.int64)) for source in SOURCE_FILES}
        ad_sequence = np.r_[0, source_indices["ad01"], len(track.beat_boundaries) - 1]
        for idx in pool:
            exact = {}
            near = {}
            dist_s = {}
            for source, indices in source_indices.items():
                exact[source] = int(np.any(indices == idx))
                dist_s[source] = nearest_time_distance(indices, int(idx), track.beat_boundaries)
                near[source] = int(dist_s[source] <= 3.0)
            ad_insert = int(np.searchsorted(ad_sequence, idx))
            prev_ad = int(ad_sequence[max(0, ad_insert - 1)])
            next_ad = int(ad_sequence[min(len(ad_sequence) - 1, ad_insert)])
            left_gap = float(track.beat_boundaries[idx] - track.beat_boundaries[prev_ad]) if idx >= prev_ad else 0.0
            right_gap = float(track.beat_boundaries[next_ad] - track.beat_boundaries[idx]) if next_ad >= idx else 0.0
            ad = ad_conf[track.track_id]
            stat = stat_curves[track.track_id]
            multi = multi_curves[track.track_id]
            curve_vals = {
                "ad_conf": curve_value(ad, int(idx)),
                "stat_curve": curve_value(stat, int(idx)),
                "multi_curve": curve_value(multi, int(idx)),
                "ad_conf_local": local_curve_max(ad, int(idx), radius_local),
                "stat_local": local_curve_max(stat, int(idx), radius_local),
                "multi_local": local_curve_max(multi, int(idx), radius_local),
            }
            row = {
                "track_id": track.track_id,
                "idx": int(idx),
                "time_s": float(track.beat_boundaries[idx]),
                "time_norm": float(track.beat_boundaries[idx] / max(track.beat_boundaries[-1], 1e-8)),
                "pool_size": float(len(pool)),
                "label_3p0": boundary_label(int(idx), track.beat_boundaries, track.reference_boundaries, 3.0),
                "label_0p5": boundary_label(int(idx), track.beat_boundaries, track.reference_boundaries, 0.5),
                "support_exact": float(sum(exact.values())),
                "support_near": float(sum(near.values())),
                "support_ideas_exact": float(sum(exact[src] for src in ("i02", "i03", "i05"))),
                "support_ideas_near": float(sum(near[src] for src in ("i02", "i03", "i05"))),
                "left_gap_ad01_s": left_gap,
                "right_gap_ad01_s": right_gap,
                "min_gap_ad01_s": min(left_gap, right_gap),
                "max_gap_ad01_s": max(left_gap, right_gap),
                **curve_vals,
            }
            row["curve_mean"] = float(np.mean([curve_vals["ad_conf"], curve_vals["stat_curve"], curve_vals["multi_curve"]]))
            row["curve_max"] = float(max(curve_vals["ad_conf"], curve_vals["stat_curve"], curve_vals["multi_curve"]))
            row["curve_disagreement"] = float(np.std([curve_vals["ad_conf"], curve_vals["stat_curve"], curve_vals["multi_curve"]]))
            for source in SOURCE_FILES:
                row[f"{source}_exact"] = float(exact[source])
                row[f"{source}_near"] = float(near[source])
                row[f"{source}_dist_s"] = min(float(dist_s[source]), 32.0) / 32.0
            rows.append(row)
        if track_idx % 10 == 0:
            print(f"  candidate features: {track_idx}/{len(tracks)}", flush=True)
    return pd.DataFrame(rows)


def feature_columns(candidates: pd.DataFrame) -> list[str]:
    excluded = {"track_id", "idx", "time_s", "label_3p0", "label_0p5"}
    return [col for col in candidates.columns if col not in excluded]


def model_factories() -> dict[str, object]:
    return {
        "logit": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, class_weight="balanced", solver="liblinear", random_state=20260507),
        ),
        "gb": lambda: GradientBoostingClassifier(
            n_estimators=110,
            max_depth=2,
            learning_rate=0.045,
            min_samples_leaf=18,
            random_state=20260507,
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=360,
            max_depth=7,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            max_features=0.72,
            random_state=20260507,
            n_jobs=-1,
        ),
    }


def train_candidate_models(candidates: pd.DataFrame, tracks: list[lit.TrackInfo]) -> pd.DataFrame:
    print("Training out-of-fold boundary confidence models", flush=True)
    fold_by_track = {track.track_id: track.fold for track in tracks}
    df = candidates.copy()
    df["fold"] = df["track_id"].map(fold_by_track)
    xcols = feature_columns(df)
    prob_rows = []
    for fold in sorted(df["fold"].dropna().unique()):
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold].copy()
        x_train = train[xcols].to_numpy(dtype=float)
        y_train = train["label_3p0"].to_numpy(dtype=int)
        x_test = test[xcols].to_numpy(dtype=float)
        for model_id, factory in model_factories().items():
            model = factory()
            if model_id == "gb":
                pos = max(1, int(y_train.sum()))
                neg = max(1, int(len(y_train) - y_train.sum()))
                weights = np.where(y_train == 1, neg / pos, 1.0)
                weights = np.where(train["label_0p5"].to_numpy(dtype=int) == 1, weights * 1.25, weights)
                model.fit(x_train, y_train, sample_weight=weights)
            else:
                model.fit(x_train, y_train)
            local = test[["track_id", "idx", "time_s", "label_3p0", "label_0p5", "ad01_exact", "support_near", "support_exact"]].copy()
            local["model_id"] = model_id
            local["prob_boundary"] = model.predict_proba(x_test)[:, 1]
            prob_rows.append(local)
        print(f"  trained/tested fold {int(fold)}", flush=True)
    return pd.concat(prob_rows, ignore_index=True)


def min_distance_beats(beat_boundaries: np.ndarray, min_segment_s: float) -> int:
    return max(3, int(round(min_segment_s / max(median_beat_s(beat_boundaries), 1e-3))))


def enforce_min_distance(indices: np.ndarray, scores: np.ndarray, beat_boundaries: np.ndarray, min_segment_s: float) -> np.ndarray:
    if indices.size == 0:
        return indices.astype(np.int64)
    distance = min_distance_beats(beat_boundaries, min_segment_s)
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    for pos in order:
        idx = int(indices[pos])
        if all(abs(idx - previous) >= distance for previous in selected):
            selected.append(idx)
    return np.asarray(sorted(selected), dtype=np.int64)


def rank_select(local: pd.DataFrame, beat_boundaries: np.ndarray, config: SelectionConfig, target_count: int) -> np.ndarray:
    local = local.copy()
    local["score"] = (
        local["prob_boundary"]
        + config.ad01_bonus * local["ad01_exact"]
        + 0.035 * local["support_near"]
        + 0.010 * local["support_exact"]
    )
    min_t = max(4.0, 0.55 * config.min_segment_s)
    max_t = float(beat_boundaries[-1]) - min_t
    local = local[(local["time_s"] >= min_t) & (local["time_s"] <= max_t)]
    if local.empty:
        return np.zeros(0, dtype=np.int64)
    selected = enforce_min_distance(
        local["idx"].to_numpy(dtype=np.int64),
        local["score"].to_numpy(dtype=np.float32),
        beat_boundaries,
        config.min_segment_s,
    )
    if selected.size > target_count:
        selected_scores = local.set_index("idx").loc[selected, "score"].to_numpy(dtype=np.float32)
        order = np.argsort(selected_scores)[::-1][:target_count]
        selected = np.sort(selected[order])
    return selected.astype(np.int64)


def snap_ad01(local: pd.DataFrame, ad01_idx: np.ndarray, beat_boundaries: np.ndarray, config: SelectionConfig) -> np.ndarray:
    local = local.copy()
    local["score"] = local["prob_boundary"] + config.ad01_bonus * local["ad01_exact"] + 0.025 * local["support_near"]
    chosen = []
    for idx in ad01_idx:
        t = float(beat_boundaries[int(idx)])
        near = local[np.abs(local["time_s"] - t) <= config.snap_radius_s]
        if near.empty:
            chosen.append(int(idx))
        else:
            chosen.append(int(near.sort_values(["score", "support_near"], ascending=False).iloc[0]["idx"]))
    chosen = np.unique(np.asarray(chosen, dtype=np.int64))
    if chosen.size < ad01_idx.size:
        extra_count = int(ad01_idx.size - chosen.size)
        ranked = local[~local["idx"].isin(chosen)].sort_values(["score", "support_near"], ascending=False)
        chosen = np.unique(np.r_[chosen, ranked["idx"].head(extra_count).to_numpy(dtype=np.int64)])
    scores = local.set_index("idx").reindex(chosen)["score"].fillna(0.0).to_numpy(dtype=np.float32)
    return enforce_min_distance(chosen, scores, beat_boundaries, config.min_segment_s)


def rescue_long_gaps(
    local: pd.DataFrame,
    base_idx: np.ndarray,
    beat_boundaries: np.ndarray,
    config: SelectionConfig,
) -> np.ndarray:
    local = local.copy()
    local["score"] = local["prob_boundary"] + config.ad01_bonus * local["ad01_exact"] + 0.035 * local["support_near"]
    threshold = float(local["score"].quantile(config.min_quantile)) if len(local) else 1.0
    selected = list(np.asarray(base_idx, dtype=np.int64))
    added = 0
    n = len(beat_boundaries) - 1
    while added < config.max_extra:
        sequence = np.r_[0, np.asarray(sorted(selected), dtype=np.int64), n]
        best = None
        for left, right in zip(sequence[:-1], sequence[1:]):
            gap_s = float(beat_boundaries[right] - beat_boundaries[left])
            if gap_s < config.rescue_gap_s:
                continue
            left_t = float(beat_boundaries[left]) + config.min_segment_s
            right_t = float(beat_boundaries[right]) - config.min_segment_s
            pool = local[(local["time_s"] >= left_t) & (local["time_s"] <= right_t)]
            pool = pool[~pool["idx"].isin(selected)]
            pool = pool[pool["score"] >= threshold]
            if pool.empty:
                continue
            row = pool.sort_values(["score", "support_near"], ascending=False).iloc[0]
            candidate = (float(row["score"]), int(row["idx"]))
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        selected.append(best[1])
        added += 1
    selected = np.unique(np.asarray(selected, dtype=np.int64))
    scores = local.set_index("idx").reindex(selected)["score"].fillna(0.0).to_numpy(dtype=np.float32)
    return enforce_min_distance(selected, scores, beat_boundaries, config.min_segment_s)


def selection_configs() -> list[SelectionConfig]:
    configs = []
    for model_id in ("logit", "gb", "rf"):
        for ad01_bonus in (0.20, 0.35, 0.50):
            for target_delta in (-1, 0, 1):
                configs.append(
                    SelectionConfig(
                        config_id=f"{model_id}_rank_b{ad01_bonus:.2f}_d{target_delta:+d}_m6.5",
                        strategy="rank",
                        model_id=model_id,
                        min_segment_s=6.5,
                        ad01_bonus=ad01_bonus,
                        target_delta=target_delta,
                    )
                )
        for snap_radius_s in (1.0, 2.0, 3.0):
            configs.append(
                SelectionConfig(
                    config_id=f"{model_id}_snap_r{snap_radius_s:.1f}_b0.30_m5.5",
                    strategy="snap",
                    model_id=model_id,
                    min_segment_s=5.5,
                    ad01_bonus=0.30,
                    snap_radius_s=snap_radius_s,
                )
            )
        for snap_radius_s in (2.0, 3.0):
            for rescue_gap_s in (32.0, 40.0):
                for max_extra in (1, 2):
                    for min_quantile in (0.72, 0.82):
                        configs.append(
                            SelectionConfig(
                                config_id=(
                                    f"{model_id}_snapres_r{snap_radius_s:.1f}_g{rescue_gap_s:.0f}"
                                    f"_x{max_extra}_q{min_quantile:.2f}_b0.25"
                                ),
                                strategy="snap_rescue",
                                model_id=model_id,
                                min_segment_s=5.5,
                                ad01_bonus=0.25,
                                snap_radius_s=snap_radius_s,
                                rescue_gap_s=rescue_gap_s,
                                max_extra=max_extra,
                                min_quantile=min_quantile,
                            )
                        )
        for rescue_gap_s in (28.0, 36.0, 44.0):
            for max_extra in (1, 2, 3):
                for min_quantile in (0.75, 0.85, 0.92):
                    configs.append(
                        SelectionConfig(
                            config_id=f"{model_id}_rescue_g{rescue_gap_s:.0f}_x{max_extra}_q{min_quantile:.2f}_b0.20",
                            strategy="rescue",
                            model_id=model_id,
                            min_segment_s=5.5,
                            ad01_bonus=0.20,
                            rescue_gap_s=rescue_gap_s,
                            max_extra=max_extra,
                            min_quantile=min_quantile,
                        )
                    )
    return configs


def predict_for_config(
    local: pd.DataFrame,
    ad01_idx: np.ndarray,
    beat_boundaries: np.ndarray,
    config: SelectionConfig,
) -> np.ndarray:
    target = max(2, int(len(ad01_idx) + config.target_delta))
    if config.strategy == "rank":
        return rank_select(local, beat_boundaries, config, target)
    if config.strategy == "snap":
        return snap_ad01(local, ad01_idx, beat_boundaries, config)
    if config.strategy == "rescue":
        return rescue_long_gaps(local, ad01_idx, beat_boundaries, config)
    if config.strategy == "snap_rescue":
        snapped = snap_ad01(local, ad01_idx, beat_boundaries, config)
        return rescue_long_gaps(local, snapped, beat_boundaries, config)
    raise ValueError(config.strategy)


def run_selection_grid(
    tracks: list[lit.TrackInfo],
    probabilities: pd.DataFrame,
    pred_lookup: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    print("Scoring AD01-anchored selection configs", flush=True)
    configs = selection_configs()
    pd.DataFrame([config.__dict__ for config in configs]).to_csv(out_dir / "selection_config_grid.csv", index=False)
    track_map = {track.track_id: track for track in tracks}
    rows = []
    grouped = {(track_id, model_id): group for (track_id, model_id), group in probabilities.groupby(["track_id", "model_id"])}
    for cfg_idx, config in enumerate(configs, 1):
        for track in tracks:
            local = grouped[(track.track_id, config.model_id)]
            ad01_idx = pred_lookup[(track.track_id, "ad01")]
            pred = predict_for_config(local, ad01_idx, track.beat_boundaries, config)
            rows.append(
                lit.evaluate_indices(
                    track_map[track.track_id],
                    pred,
                    "ADRES_candidate",
                    extra={
                        "config_id": config.config_id,
                        "strategy": config.strategy,
                        "model_id": config.model_id,
                    },
                )
            )
        if cfg_idx % 40 == 0:
            print(f"  configs scored: {cfg_idx}/{len(configs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ad01_candidate_rescue_scores.csv", index=False)
    return df


def candidate_oracle(
    tracks: list[lit.TrackInfo],
    candidates: pd.DataFrame,
    pred_lookup: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for track in tracks:
        local = candidates[candidates["track_id"] == track.track_id].copy()
        local["oracle_score"] = local["label_3p0"] + 0.20 * local["label_0p5"] + 0.001 * local["support_near"]
        target = len(pred_lookup[(track.track_id, "ad01")])
        selected = rank_select(
            local.rename(columns={"oracle_score": "prob_boundary"}).assign(ad01_exact=local["ad01_exact"]),
            track.beat_boundaries,
            SelectionConfig("oracle", "rank", "oracle", 5.5, 0.0),
            target,
        )
        rows.append(lit.evaluate_indices(track, selected, "ORACLE_ad01_candidate_pool"))
    return pd.DataFrame(rows)


def aggregate_with_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    agg = lit.aggregate(df)
    agg["phase"] = phase
    return agg


def write_candidate_aggregates(scores: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        scores.groupby(["config_id", "strategy", "model_id"], as_index=False)
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
    agg.to_csv(out_dir / "ad01_candidate_rescue_aggregates.csv", index=False)
    if not agg.empty:
        best = agg.iloc[0]
        fixed = scores[scores["config_id"] == best["config_id"]].copy()
        fixed["exp_id"] = "ADRES_best_fixed_diagnostic"
        fixed.to_csv(out_dir / "ADRES_best_fixed_diagnostic_metrics.csv", index=False)


def compare_vs_baseline(method_df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: Path) -> None:
    base = baseline_df.set_index("track_id")
    rows = []
    for exp_id, df in method_df.groupby("exp_id"):
        local = df.set_index("track_id")
        diff = local["f_3p0"] - base["f_3p0"]
        rows.append(
            {
                "exp_id": exp_id,
                "mean_diff_vs_ad01": float(diff.mean()),
                "median_diff_vs_ad01": float(diff.median()),
                "wins_vs_ad01": int((diff > 1e-9).sum()),
                "ties_vs_ad01": int((np.abs(diff) <= 1e-9).sum()),
                "losses_vs_ad01": int((diff < -1e-9).sum()),
                "best_track_gain": float(diff.max()),
                "worst_track_loss": float(diff.min()),
            }
        )
    pd.DataFrame(rows).sort_values("mean_diff_vs_ad01", ascending=False).to_csv(out_dir / "comparison_vs_ad01.csv", index=False)


def plot_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    shown = comparison.sort_values("f_3p0", ascending=True)
    colors = []
    for exp_id in shown["exp_id"]:
        if "ORACLE" in exp_id:
            colors.append("#7C3AED")
        elif exp_id.startswith("ADRES"):
            colors.append("#0F766E")
        elif exp_id.startswith("AD01"):
            colors.append("#2563EB")
        elif exp_id.startswith("I0"):
            colors.append("#C2410C")
        elif exp_id.startswith("FUS"):
            colors.append("#059669")
        else:
            colors.append("#94A3B8")
    fig, ax = plt.subplots(figsize=(12.4, 7.8), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, shown["f_3p0"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shown["exp_id"], fontsize=8.5)
    ax.set_xlabel("F@3s")
    ax.set_title("AD01 anclada + rescate/snapping de fronteras")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(max(0.62, float(shown["f_3p0"].min()) - 0.02), min(0.73, float(shown["f_3p0"].max()) + 0.02))
    for i, row in shown.reset_index(drop=True).iterrows():
        ax.text(float(row["f_3p0"]) + 0.002, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8.3)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| Metodo | F@3s | F@0.5s | Precision@3s | Recall@3s | Pred. prom. | Fase |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.exp_id} | {row.f_3p0:.3f} | {row.f_0p5:.3f} | {row.precision_3p0:.3f} | "
            f"{row.recall_3p0:.3f} | {row.pred_count:.2f} | {row.phase} |"
        )
    return "\n".join(lines)


def chosen_table(chosen: pd.DataFrame) -> str:
    if chosen.empty:
        return "_Sin configuraciones seleccionadas._"
    cols = ["fold", "config_id", "train_f_3p0", "train_f_0p5", "train_precision_3p0"]
    cols = [col for col in cols if col in chosen.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in cols[1:]]) + " |",
    ]
    for row in chosen[cols].itertuples(index=False):
        values = []
        for col, value in zip(cols, row):
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    comparison: pd.DataFrame,
    selected: pd.DataFrame,
    chosen: pd.DataFrame,
    candidates: pd.DataFrame,
    out_dir: Path,
) -> None:
    best_real = comparison[~comparison["exp_id"].str.contains("ORACLE")].iloc[0]
    lines = [
        "# AD01 anclada con rescate de fronteras",
        "",
        "## Que se probo",
        "",
        "La prueba parte de AD01_PR01, el mejor sistema actual. En lugar de sustituirlo por I02/I03/I05, se construyo un pool local de candidatas: fronteras de AD01, fronteras propuestas por las ideas exploratorias y pequenos desplazamientos alrededor de esas fronteras. Para cada candidata se calcularon senales de soporte entre metodos, distancia a AD01, confianza AD01, novelty estadistica e informacion multiescala.",
        "",
        "Luego se entrenaron clasificadores de frontera out-of-fold y se evaluaron cuatro acciones conservadoras: ranking con bonus a AD01, snap local de AD01, rescate de huecos largos y snap+rescate.",
        "",
        "## Resultado agregado",
        "",
        markdown_table(comparison),
        "",
        "## Lectura",
        "",
        f"- Mejor resultado no-oracular: `{best_real.exp_id}` con F@3s={best_real.f_3p0:.3f}.",
        f"- Pool candidato: {len(candidates)} fronteras candidatas; positivas@3s={candidates['label_3p0'].mean():.2%}; positivas@0.5s={candidates['label_0p5'].mean():.2%}.",
        "- Si ADRES supera AD01, la mejora viene de usar las ideas nuevas como evidencia local y no como detector completo.",
        "- Si no supera AD01, la lectura es que las senales agregan informacion pero aun no calibran bien el balance precision/recall.",
        "",
        "## Configuracion seleccionada por fold",
        "",
        chosen_table(chosen),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--disruptive-dir", type=Path, default=DISRUPTIVE_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = lit.load_tracks(args.limit, args.folds)
    print(f"Loaded {len(tracks)} tracks", flush=True)

    source_frames = load_source_predictions(args.disruptive_dir)
    pred_lookup = predictions_by_source(source_frames)
    baseline_df = ad01_rows(source_frames)

    _ad_matrices, ad_conf, stat_curves, multi_curves = build_ad01_context(tracks)
    candidates = build_candidate_features(tracks, pred_lookup, ad_conf, stat_curves, multi_curves)
    candidates.to_csv(args.out_dir / "boundary_candidate_features.csv", index=False)
    probabilities = train_candidate_models(candidates, tracks)
    probabilities.to_csv(args.out_dir / "boundary_candidate_probabilities.csv", index=False)

    scores = run_selection_grid(tracks, probabilities, pred_lookup, args.out_dir)
    write_candidate_aggregates(scores, args.out_dir)
    selected, chosen = sgdp.fold_select(scores, tracks, ["config_id"], "ADRES01_ad01_anchored_candidate_rescue_foldcv")
    selected.to_csv(args.out_dir / "ADRES01_ad01_anchored_candidate_rescue_foldcv_metrics.csv", index=False)
    chosen.to_csv(args.out_dir / "ADRES01_ad01_anchored_candidate_rescue_foldcv_chosen_configs.csv", index=False)
    oracle = candidate_oracle(tracks, candidates, pred_lookup)
    oracle.to_csv(args.out_dir / "ORACLE_ad01_candidate_pool_metrics.csv", index=False)

    previous_agg = pd.read_csv(args.disruptive_dir / "aggregate_metrics.csv")
    keep_ids = [
        "AD01_PR01_sgdp_preregistered",
        "FUS01_ad01_learned_sgdp_foldcv",
        "I05_nonlinear_gate_foldcv",
        "I03_multiscale_ssm_foldcv",
        "I02_statistical_novelty_foldcv",
        "ORACLE_candidate_pool",
    ]
    keep = previous_agg[previous_agg["exp_id"].isin(keep_ids)].copy()
    keep["phase"] = keep.get("phase", "previous")
    comparison = pd.concat(
        [
            keep[["exp_id", "f_0p5", "f_3p0", "precision_3p0", "recall_3p0", "pred_count", "tracks", "phase"]],
            aggregate_with_phase(selected, "ad01_candidate_rescue_foldcv"),
            aggregate_with_phase(oracle, "oracle_candidate_pool"),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["f_3p0", "f_0p5"], ascending=False)
    comparison.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    compare_vs_baseline(pd.concat([selected, oracle], ignore_index=True), baseline_df, args.out_dir)
    plot_comparison(comparison, args.out_dir / "aggregate_metrics.png")
    write_summary(comparison, selected, chosen, candidates, args.out_dir)

    print(comparison[["exp_id", "phase", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
