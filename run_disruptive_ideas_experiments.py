#!/usr/bin/env python3
"""Five more disruptive MSA experiments on top of AD01 + stability-guided DP.

Ideas tested, in order:
1. Global decoder with latent repetition/state reward.
2. Local statistical-test novelty as a boundary prior.
3. Multi-scale beat/phrase SSM prior.
4. DP local margin as confidence.
5. Non-linear AD01/learned-SSM gating.
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
from scipy.spatial.distance import cdist

import run_ad01_learned_fusion_experiment as fus
import run_literature_guided_iterations as lit
import run_song_adaptive_ssm_experiments as ad
import run_stability_guided_dp_iteration as sgdp
from run_similarity_metric_experiments import BEAT_KEYS
from run_structure_baseline import normalize_ssm, novelty_curve


OUT_DIR = Path("structure_outputs/rwc_p_100_disruptive_ideas_v1")
AD01_CONFIG = fus.AD01_CONFIG


@dataclass(frozen=True)
class StateDecoderConfig:
    config_id: str
    base_config: sgdp.StabilityDPConfig
    repetition_w: float
    beam: int


@dataclass(frozen=True)
class MarginConfig:
    config_id: str
    quantile: float
    min_keep: int


@dataclass(frozen=True)
class GateConfig:
    config_id: str
    learned_w: float
    agreement_w: float
    stability_w: float
    min_segment_s: float


def base_configs() -> list[sgdp.StabilityDPConfig]:
    return [
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.40", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.40),
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.20", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h1.0_n0.9_r0.20_s0.20", 14.0, 5.5, 42.0, 1.0, 0.9, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h1.0_n0.9_r0.20_s0.40", 14.0, 5.5, 42.0, 1.0, 0.9, 0.20, 0.40),
        sgdp.StabilityDPConfig("t16_h0.6_n0.9_r0.20_s0.40", 16.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.40),
    ]


def norm_curve(curve: np.ndarray) -> np.ndarray:
    return sgdp.norm_curve(curve)


def ad01_for_tracks(tracks: list[lit.TrackInfo]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    matrices: dict[str, np.ndarray] = {}
    rows = []
    for idx, track in enumerate(tracks, 1):
        matrix, weights = fus.ad01_matrix(track)
        matrices[track.track_id] = matrix
        rows.append(
            lit.evaluate_matrix(
                track,
                matrix,
                "AD01_PR00_ssm_preregistered",
                extra={"weights": json.dumps(weights, sort_keys=True), "density_weight": fus.AD01_DENSITY_WEIGHT},
            )
        )
        if idx % 10 == 0:
            print(f"  AD01 matrices: {idx}/{len(tracks)}", flush=True)
    return matrices, pd.DataFrame(rows)


def ad01_preregistered_dp(
    tracks: list[lit.TrackInfo],
    matrices: dict[str, np.ndarray],
    stability: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for track in tracks:
        indices = sgdp.stability_guided_dp_indices(matrices[track.track_id], track.beat_boundaries, stability[track.track_id], AD01_CONFIG)
        rows.append(
            lit.evaluate_indices(
                track,
                indices,
                "AD01_PR01_sgdp_preregistered",
                extra={"config_id": AD01_CONFIG.config_id},
            )
        )
    return pd.DataFrame(rows)


def block_cross_integral(ssm: np.ndarray):
    csum = np.pad(ssm, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    def mean(a: int, b: int, c: int, d: int) -> float:
        if b <= a or d <= c:
            return 0.0
        total = csum[b, d] - csum[a, d] - csum[b, c] + csum[a, c]
        return float(total / max((b - a) * (d - c), 1))

    return mean


def segment_score_helpers(ssm: np.ndarray, beat_boundaries: np.ndarray, stability: np.ndarray, config: sgdp.StabilityDPConfig):
    novelty = novelty_curve(ssm)
    stability = norm_curve(stability[: len(novelty)])
    confidence = sgdp.confidence_curve(novelty, stability, config.stability_w)
    mean_fn = lit.block_mean_integral(normalize_ssm(ssm))

    def seg_score(a: int, b: int) -> float:
        length_s = float(beat_boundaries[b] - beat_boundaries[a])
        if length_s < config.min_s or length_s > config.max_s:
            return -1e9
        hom = mean_fn(a, b)
        boundary_bonus = 0.0
        if 0 < a < len(confidence):
            boundary_bonus += float(confidence[a])
        if 0 < b < len(confidence):
            boundary_bonus += float(confidence[b])
        reg = -abs(np.log(max(length_s, 1e-3) / config.target_s))
        return config.hom_w * hom + config.nov_w * boundary_bonus + config.reg_w * reg

    return novelty, stability, confidence, seg_score


def limited_candidates(ssm: np.ndarray, beat_boundaries: np.ndarray, stability: np.ndarray, config: sgdp.StabilityDPConfig, max_internal: int = 88) -> np.ndarray:
    novelty = novelty_curve(ssm)
    stability = norm_curve(stability[: len(novelty)])
    candidates = sgdp.guided_candidate_boundaries(novelty, stability, beat_boundaries, config.stability_w)
    if candidates.size > max_internal:
        confidence = sgdp.confidence_curve(novelty, stability, config.stability_w)
        order = np.argsort(confidence[candidates])[::-1][:max_internal]
        candidates = np.sort(candidates[order])
    return candidates.astype(np.int64)


def state_decoder_configs() -> list[StateDecoderConfig]:
    configs = []
    for base_config in base_configs()[:4]:
        for repetition_w in (0.08, 0.16, 0.28):
            configs.append(
                StateDecoderConfig(
                    config_id=f"{base_config.config_id}_rep{repetition_w:.2f}",
                    base_config=base_config,
                    repetition_w=repetition_w,
                    beam=42,
                )
            )
    return configs


def state_decoder_indices(ssm: np.ndarray, beat_boundaries: np.ndarray, stability: np.ndarray, config: StateDecoderConfig) -> np.ndarray:
    base = config.base_config
    n = ssm.shape[0]
    if n < 12:
        return np.zeros(0, dtype=np.int64)
    candidates = np.unique(np.r_[0, limited_candidates(ssm, beat_boundaries, stability, base), n]).astype(np.int64)
    duration = float(beat_boundaries[-1])
    target_segments = int(np.clip(round(duration / base.target_s), 2, min(34, len(candidates) - 1)))
    _novelty, _stability, _confidence, seg_score = segment_score_helpers(ssm, beat_boundaries, stability, base)
    cross_mean = block_cross_integral(normalize_ssm(ssm))
    positions = {int(value): idx for idx, value in enumerate(candidates)}

    # State tuple: (score, current_endpoint, path_endpoints, intervals)
    states = [(0.0, 0, [0], [])]
    for step in range(target_segments):
        final_step = step == target_segments - 1
        new_states = []
        for score, current, path, intervals in states:
            current_pos = positions.get(int(current), 0)
            rights = [n] if final_step else candidates[current_pos + 1 : -1]
            for right in rights:
                right = int(right)
                base_score = seg_score(int(current), right)
                if base_score <= -1e8:
                    continue
                repetition = 0.0
                if len(intervals) > 1:
                    similarities = [
                        cross_mean(int(a), int(b), int(current), right)
                        for a, b in intervals[:-1]
                        if abs(int(current) - int(b)) > 2
                    ]
                    if similarities:
                        repetition = max(0.0, max(similarities) - 0.42)
                new_states.append(
                    (
                        score + base_score + config.repetition_w * repetition,
                        right,
                        [*path, right],
                        [*intervals, (int(current), right)],
                    )
                )
        if not new_states:
            return sgdp.stability_guided_dp_indices(ssm, beat_boundaries, stability, base)
        # Keep diverse high-scoring beams by current endpoint.
        new_states.sort(key=lambda item: item[0], reverse=True)
        by_endpoint: dict[int, list[tuple]] = {}
        for state in new_states:
            by_endpoint.setdefault(int(state[1]), []).append(state)
        states = []
        for endpoint_states in by_endpoint.values():
            states.extend(endpoint_states[:3])
        states = sorted(states, key=lambda item: item[0], reverse=True)[: config.beam]
    final_states = [state for state in states if int(state[1]) == n]
    if not final_states:
        return sgdp.stability_guided_dp_indices(ssm, beat_boundaries, stability, base)
    best = max(final_states, key=lambda item: item[0])
    return np.asarray(best[2][1:-1], dtype=np.int64)


def combined_robust_features(track: lit.TrackInfo) -> np.ndarray:
    with np.load(track.feature_path, allow_pickle=True) as data:
        mfcc = ad.robust_standardize(np.asarray(data[BEAT_KEYS["mfcc"]], dtype=np.float32))
        density = ad.robust_standardize(np.asarray(data[BEAT_KEYS["density"]], dtype=np.float32))
    return np.hstack([np.sqrt(0.45) * mfcc, np.sqrt(0.55) * density]).astype(np.float32)


def energy_distance_curve(x: np.ndarray, half_width: int) -> np.ndarray:
    n = x.shape[0]
    curve = np.zeros(n, dtype=np.float32)
    if n < 2 * half_width + 4:
        return curve
    for idx in range(half_width, n - half_width):
        left = x[idx - half_width : idx]
        right = x[idx : idx + half_width]
        cross = cdist(left, right, metric="euclidean").mean()
        left_d = cdist(left, left, metric="euclidean")
        right_d = cdist(right, right, metric="euclidean")
        within = 0.5 * (left_d.mean() + right_d.mean())
        curve[idx] = max(0.0, 2.0 * cross - 2.0 * within)
    return norm_curve(gaussian_filter1d(curve, sigma=max(1.0, half_width / 8.0)))


def statistical_curve(track: lit.TrackInfo) -> np.ndarray:
    x = combined_robust_features(track)
    curves = [energy_distance_curve(x, half_width) for half_width in (8, 16, 24)]
    min_len = min(len(curve) for curve in curves)
    return norm_curve(np.mean([curve[:min_len] for curve in curves], axis=0))


def pool_features(features: np.ndarray, pool: int) -> np.ndarray:
    if pool <= 1:
        return np.asarray(features, dtype=np.float32)
    dim, n = features.shape
    groups = int(np.ceil(n / pool))
    out = np.zeros((dim, groups), dtype=np.float32)
    for group in range(groups):
        start = group * pool
        end = min(n, start + pool)
        out[:, group] = np.mean(features[:, start:end], axis=1)
    return out


def multiscale_ssm_curve(track: lit.TrackInfo) -> np.ndarray:
    with np.load(track.feature_path, allow_pickle=True) as data:
        n = int(np.asarray(data[BEAT_KEYS["mfcc"]]).shape[1])
        curves = []
        for pool in (1, 2, 4, 8):
            matrices = {
                "mfcc": ad.robust_diag_rbf_ssm(pool_features(np.asarray(data[BEAT_KEYS["mfcc"]], dtype=np.float32), pool)),
                "density": ad.robust_diag_rbf_ssm(pool_features(np.asarray(data[BEAT_KEYS["density"]], dtype=np.float32), pool)),
            }
            matrix, _weights = lit.reliability_fuse(matrices, {"mfcc": 0.45, "density": 0.55})
            curve = novelty_curve(matrix)
            upsampled = np.repeat(curve, pool)[:n]
            if len(upsampled) < n:
                upsampled = np.pad(upsampled, (0, n - len(upsampled)), mode="edge")
            curves.append(norm_curve(upsampled))
    return norm_curve(np.mean(curves, axis=0))


def margin_configs() -> list[MarginConfig]:
    return [
        MarginConfig("q0.10_min8", 0.10, 8),
        MarginConfig("q0.20_min8", 0.20, 8),
        MarginConfig("q0.30_min8", 0.30, 8),
        MarginConfig("q0.20_min10", 0.20, 10),
    ]


def local_margin_filter_indices(
    indices: np.ndarray,
    ssm: np.ndarray,
    beat_boundaries: np.ndarray,
    stability: np.ndarray,
    config: MarginConfig,
) -> np.ndarray:
    if indices.size <= config.min_keep:
        return indices
    _novelty, _stability, _confidence, seg_score = segment_score_helpers(ssm, beat_boundaries, stability, AD01_CONFIG)
    sequence = np.r_[0, indices, ssm.shape[0]]
    margins = []
    for pos, idx in enumerate(indices, start=1):
        prev_idx = int(sequence[pos - 1])
        next_idx = int(sequence[pos + 1])
        current = seg_score(prev_idx, int(idx)) + seg_score(int(idx), next_idx)
        left_t = float(beat_boundaries[prev_idx]) + AD01_CONFIG.min_s
        right_t = float(beat_boundaries[next_idx]) - AD01_CONFIG.min_s
        left = int(np.searchsorted(beat_boundaries, left_t))
        right = min(int(np.searchsorted(beat_boundaries, right_t)), next_idx - 1)
        if right <= left:
            margins.append(0.0)
            continue
        alternatives = [seg_score(prev_idx, cand) + seg_score(cand, next_idx) for cand in range(left, right + 1) if abs(cand - int(idx)) > 1]
        best_alt = max(alternatives) if alternatives else current
        margins.append(float(current - best_alt))
    margins_arr = np.asarray(margins, dtype=np.float32)
    threshold = float(np.quantile(margins_arr, config.quantile))
    keep = indices[margins_arr >= threshold]
    if keep.size < config.min_keep:
        order = np.argsort(margins_arr)[::-1][: config.min_keep]
        keep = np.sort(indices[order])
    return np.asarray(keep, dtype=np.int64)


def gate_configs() -> list[GateConfig]:
    configs = []
    for learned_w in (0.25, 0.45, 0.65):
        for agreement_w in (0.10, 0.25):
            for stability_w in (0.20, 0.40):
                configs.append(
                    GateConfig(
                        config_id=f"lw{learned_w:.2f}_aw{agreement_w:.2f}_sw{stability_w:.2f}",
                        learned_w=learned_w,
                        agreement_w=agreement_w,
                        stability_w=stability_w,
                        min_segment_s=8.5,
                    )
                )
    return configs


def nearest_distance(indices: np.ndarray, target: int) -> int:
    if indices.size == 0:
        return 9999
    return int(np.min(np.abs(indices - int(target))))


def enforce_min_distance(indices: np.ndarray, scores: np.ndarray, beat_boundaries: np.ndarray, min_segment_s: float) -> np.ndarray:
    return sgdp.enforce_min_distance(np.asarray(indices, dtype=np.int64), np.asarray(scores, dtype=np.float32), beat_boundaries, min_segment_s)


def nonlinear_gate_indices(
    track: lit.TrackInfo,
    ad01_idx: np.ndarray,
    learned_idx: np.ndarray,
    ad01_matrix: np.ndarray,
    learned_matrix: np.ndarray,
    ad01_stability: np.ndarray,
    learned_stability: np.ndarray,
    config: GateConfig,
) -> np.ndarray:
    ad_conf = sgdp.confidence_curve(novelty_curve(ad01_matrix), ad01_stability, 0.40)
    learned_conf = sgdp.confidence_curve(novelty_curve(learned_matrix), learned_stability, 0.40)
    n = min(len(ad_conf), len(learned_conf))
    candidates = np.unique(np.r_[ad01_idx, learned_idx]).astype(np.int64)
    candidates = candidates[(candidates > 0) & (candidates < n)]
    if candidates.size == 0:
        return ad01_idx
    scores = []
    for idx in candidates:
        ad_agree = np.exp(-(nearest_distance(ad01_idx, int(idx)) ** 2) / (2.0 * 3.0**2))
        learned_agree = np.exp(-(nearest_distance(learned_idx, int(idx)) ** 2) / (2.0 * 3.0**2))
        agreement = float(ad_agree * learned_agree)
        stability = max(float(ad01_stability[min(idx, len(ad01_stability) - 1)]), float(learned_stability[min(idx, len(learned_stability) - 1)]))
        score = (
            float(ad_conf[idx])
            + config.learned_w * float(learned_conf[idx]) * (0.5 + config.stability_w * stability)
            + config.agreement_w * agreement
        )
        scores.append(score)
    selected = enforce_min_distance(candidates, np.asarray(scores), track.beat_boundaries, config.min_segment_s)
    target_count = max(2, int(round(0.96 * len(ad01_idx))))
    if selected.size > target_count:
        selected_scores = np.asarray([scores[np.where(candidates == idx)[0][0]] for idx in selected], dtype=np.float32)
        keep_order = np.argsort(selected_scores)[::-1][:target_count]
        selected = np.sort(selected[keep_order])
    return selected.astype(np.int64)


def fixed_best_diagnostic(candidate_df: pd.DataFrame, group_cols: list[str], exp_id: str, out_path: Path) -> pd.DataFrame:
    agg = (
        candidate_df.groupby(group_cols, as_index=False)
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
    agg.to_csv(out_path.with_name(out_path.stem + "_aggregates.csv"), index=False)
    best = agg.iloc[0]
    mask = pd.Series(True, index=candidate_df.index)
    for col in group_cols:
        mask &= candidate_df[col] == best[col]
    fixed = candidate_df[mask].copy()
    fixed["exp_id"] = exp_id
    fixed.to_csv(out_path, index=False)
    return fixed


def run_first_four_ideas(
    tracks: list[lit.TrackInfo],
    ad01_matrices: dict[str, np.ndarray],
    ad01_stability: dict[str, np.ndarray],
    baseline_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    idea_frames = []

    print("Idea 1: latent repetition/state decoder", flush=True)
    state_rows = []
    state_cfgs = state_decoder_configs()
    pd.DataFrame(
        [
            {"config_id": cfg.config_id, "base_config": cfg.base_config.config_id, "repetition_w": cfg.repetition_w, "beam": cfg.beam}
            for cfg in state_cfgs
        ]
    ).to_csv(out_dir / "idea1_state_decoder_grid.csv", index=False)
    for idx, track in enumerate(tracks, 1):
        matrix = ad01_matrices[track.track_id]
        stability = ad01_stability[track.track_id]
        for cfg in state_cfgs:
            pred = state_decoder_indices(matrix, track.beat_boundaries, stability, cfg)
            state_rows.append(lit.evaluate_indices(track, pred, "I01_candidate_state_decoder", extra={"config_id": cfg.config_id}))
        if idx % 10 == 0:
            print(f"  idea 1 tracks: {idx}/{len(tracks)}", flush=True)
    state_df = pd.DataFrame(state_rows)
    state_df.to_csv(out_dir / "idea1_state_decoder_candidate_scores.csv", index=False)
    state_sel, state_chosen = sgdp.fold_select(state_df, tracks, ["config_id"], "I01_state_decoder_foldcv")
    state_sel.to_csv(out_dir / "I01_state_decoder_foldcv_metrics.csv", index=False)
    state_chosen.to_csv(out_dir / "I01_state_decoder_foldcv_chosen_configs.csv", index=False)
    fixed_best_diagnostic(state_df, ["config_id"], "I01_state_decoder_best_fixed_diagnostic", out_dir / "I01_state_decoder_best_fixed_diagnostic_metrics.csv")
    idea_frames.append(state_sel)

    print("Idea 2: local statistical-test novelty prior", flush=True)
    stat_curves = {track.track_id: statistical_curve(track) for track in tracks}
    stat_rows = []
    for idx, track in enumerate(tracks, 1):
        matrix = ad01_matrices[track.track_id]
        for cfg in base_configs():
            pred = sgdp.stability_guided_dp_indices(matrix, track.beat_boundaries, stat_curves[track.track_id], cfg)
            stat_rows.append(lit.evaluate_indices(track, pred, "I02_candidate_statistical_novelty", extra={"config_id": cfg.config_id}))
        if idx % 10 == 0:
            print(f"  idea 2 tracks: {idx}/{len(tracks)}", flush=True)
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_dir / "idea2_statistical_novelty_candidate_scores.csv", index=False)
    stat_sel, stat_chosen = sgdp.fold_select(stat_df, tracks, ["config_id"], "I02_statistical_novelty_foldcv")
    stat_sel.to_csv(out_dir / "I02_statistical_novelty_foldcv_metrics.csv", index=False)
    stat_chosen.to_csv(out_dir / "I02_statistical_novelty_foldcv_chosen_configs.csv", index=False)
    fixed_best_diagnostic(stat_df, ["config_id"], "I02_statistical_novelty_best_fixed_diagnostic", out_dir / "I02_statistical_novelty_best_fixed_diagnostic_metrics.csv")
    idea_frames.append(stat_sel)

    print("Idea 3: multi-scale beat/phrase SSM prior", flush=True)
    multiscale_curves = {track.track_id: multiscale_ssm_curve(track) for track in tracks}
    multi_rows = []
    for idx, track in enumerate(tracks, 1):
        matrix = ad01_matrices[track.track_id]
        for cfg in base_configs():
            pred = sgdp.stability_guided_dp_indices(matrix, track.beat_boundaries, multiscale_curves[track.track_id], cfg)
            multi_rows.append(lit.evaluate_indices(track, pred, "I03_candidate_multiscale_ssm", extra={"config_id": cfg.config_id}))
        if idx % 10 == 0:
            print(f"  idea 3 tracks: {idx}/{len(tracks)}", flush=True)
    multi_df = pd.DataFrame(multi_rows)
    multi_df.to_csv(out_dir / "idea3_multiscale_ssm_candidate_scores.csv", index=False)
    multi_sel, multi_chosen = sgdp.fold_select(multi_df, tracks, ["config_id"], "I03_multiscale_ssm_foldcv")
    multi_sel.to_csv(out_dir / "I03_multiscale_ssm_foldcv_metrics.csv", index=False)
    multi_chosen.to_csv(out_dir / "I03_multiscale_ssm_foldcv_chosen_configs.csv", index=False)
    fixed_best_diagnostic(multi_df, ["config_id"], "I03_multiscale_ssm_best_fixed_diagnostic", out_dir / "I03_multiscale_ssm_best_fixed_diagnostic_metrics.csv")
    idea_frames.append(multi_sel)

    print("Idea 4: DP margin confidence", flush=True)
    margin_rows = []
    by_track_baseline = {row.track_id: np.asarray(json.loads(row.pred_indices), dtype=np.int64) for row in baseline_df.itertuples(index=False)}
    for idx, track in enumerate(tracks, 1):
        matrix = ad01_matrices[track.track_id]
        stability = ad01_stability[track.track_id]
        baseline_idx = by_track_baseline[track.track_id]
        for cfg in margin_configs():
            pred = local_margin_filter_indices(baseline_idx, matrix, track.beat_boundaries, stability, cfg)
            margin_rows.append(lit.evaluate_indices(track, pred, "I04_candidate_dp_margin", extra={"config_id": cfg.config_id, "quantile": cfg.quantile}))
        if idx % 10 == 0:
            print(f"  idea 4 tracks: {idx}/{len(tracks)}", flush=True)
    margin_df = pd.DataFrame(margin_rows)
    margin_df.to_csv(out_dir / "idea4_dp_margin_candidate_scores.csv", index=False)
    margin_sel, margin_chosen = sgdp.fold_select(margin_df, tracks, ["config_id"], "I04_dp_margin_foldcv")
    margin_sel.to_csv(out_dir / "I04_dp_margin_foldcv_metrics.csv", index=False)
    margin_chosen.to_csv(out_dir / "I04_dp_margin_foldcv_chosen_configs.csv", index=False)
    fixed_best_diagnostic(margin_df, ["config_id"], "I04_dp_margin_best_fixed_diagnostic", out_dir / "I04_dp_margin_best_fixed_diagnostic_metrics.csv")
    idea_frames.append(margin_sel)
    return state_sel, stat_sel, multi_sel, margin_sel, idea_frames


def run_idea5(
    tracks: list[lit.TrackInfo],
    ad01_matrices: dict[str, np.ndarray],
    ad01_stability: dict[str, np.ndarray],
    baseline_df: pd.DataFrame,
    out_dir: Path,
    samples_per_class: int,
    folds: int,
) -> pd.DataFrame:
    print("Idea 5: non-linear AD01/learned gating", flush=True)
    samples = lit.precompute_pair_samples(tracks, samples_per_class)
    models = lit.fold_models(tracks, samples, folds)
    learned_matrices = {}
    learned_stability = {}
    learned_rows = []
    for idx, track in enumerate(tracks, 1):
        matrix, weights = sgdp.learned_metric_matrix(track, models)
        learned_matrices[track.track_id] = matrix
        learned_stability[track.track_id] = sgdp.stability_curves_for_tracks([track], {track.track_id: matrix})[track.track_id]
        learned_rows.append(lit.evaluate_matrix(track, matrix, "ML03_pair_metric_preregistered", extra={"weights": json.dumps(weights, sort_keys=True)}))
        if idx % 10 == 0:
            print(f"  learned matrices: {idx}/{len(tracks)}", flush=True)
    pd.DataFrame(learned_rows).to_csv(out_dir / "ML03_pair_metric_preregistered_metrics.csv", index=False)
    learned_idx_by_track = {
        track.track_id: sgdp.fixed_previous_dp_indices(learned_matrices[track.track_id], track.beat_boundaries)
        for track in tracks
    }
    ad01_idx_by_track = {row.track_id: np.asarray(json.loads(row.pred_indices), dtype=np.int64) for row in baseline_df.itertuples(index=False)}

    gate_rows = []
    gate_cfgs = gate_configs()
    pd.DataFrame([asdict(cfg) for cfg in gate_cfgs]).to_csv(out_dir / "idea5_gating_grid.csv", index=False)
    for idx, track in enumerate(tracks, 1):
        for cfg in gate_cfgs:
            pred = nonlinear_gate_indices(
                track,
                ad01_idx_by_track[track.track_id],
                learned_idx_by_track[track.track_id],
                ad01_matrices[track.track_id],
                learned_matrices[track.track_id],
                ad01_stability[track.track_id],
                learned_stability[track.track_id],
                cfg,
            )
            gate_rows.append(lit.evaluate_indices(track, pred, "I05_candidate_nonlinear_gate", extra={"config_id": cfg.config_id}))
        if idx % 10 == 0:
            print(f"  idea 5 tracks: {idx}/{len(tracks)}", flush=True)
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out_dir / "idea5_nonlinear_gate_candidate_scores.csv", index=False)
    gate_sel, gate_chosen = sgdp.fold_select(gate_df, tracks, ["config_id"], "I05_nonlinear_gate_foldcv")
    gate_sel.to_csv(out_dir / "I05_nonlinear_gate_foldcv_metrics.csv", index=False)
    gate_chosen.to_csv(out_dir / "I05_nonlinear_gate_foldcv_chosen_configs.csv", index=False)
    fixed_best_diagnostic(gate_df, ["config_id"], "I05_nonlinear_gate_best_fixed_diagnostic", out_dir / "I05_nonlinear_gate_best_fixed_diagnostic_metrics.csv")
    return gate_sel


def previous_rows() -> pd.DataFrame:
    rows = []
    previous = sgdp.previous_rows()
    if not previous.empty:
        rows.append(previous)
    path = Path("structure_outputs/rwc_p_100_ad01_learned_fusion_v1/aggregate_metrics.csv")
    if path.exists():
        df = pd.read_csv(path)
        keep = df[df["exp_id"].isin(["AD01_PR01_sgdp_preregistered", "FUS01_ad01_learned_sgdp_foldcv"])]
        if not keep.empty:
            keep = keep.copy()
            keep["phase"] = "previous_best"
            rows.append(keep[["exp_id", "f_0p5", "f_3p0", "precision_3p0", "recall_3p0", "pred_count", "tracks", "phase"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    shown = comparison.sort_values("f_3p0", ascending=True)
    colors = []
    for exp_id in shown["exp_id"]:
        if "ORACLE" in exp_id.upper():
            colors.append("#7C3AED")
        elif exp_id.startswith("I01"):
            colors.append("#2563EB")
        elif exp_id.startswith("I02"):
            colors.append("#C2410C")
        elif exp_id.startswith("I03"):
            colors.append("#DB2777")
        elif exp_id.startswith("I04"):
            colors.append("#D97706")
        elif exp_id.startswith("I05"):
            colors.append("#0F766E")
        elif exp_id.startswith("AD01") or exp_id.startswith("FUS"):
            colors.append("#059669")
        else:
            colors.append("#94A3B8")
    fig, ax = plt.subplots(figsize=(12.6, 8.2), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, shown["f_3p0"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shown["exp_id"], fontsize=8)
    ax.set_xlabel("F@3s")
    ax.set_title("Cinco ideas disruptivas sobre AD01")
    ax.grid(axis="x", alpha=0.25)
    lo = max(0.48, float(shown["f_3p0"].min()) - 0.02)
    hi = min(0.73, float(shown["f_3p0"].max()) + 0.02)
    ax.set_xlim(lo, hi)
    for i, row in shown.reset_index(drop=True).iterrows():
        ax.text(float(row["f_3p0"]) + 0.002, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary(out_dir: Path, comparison: pd.DataFrame, selected_frames: list[pd.DataFrame]) -> None:
    lines = [
        "# Cinco ideas disruptivas evaluadas",
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
    idea_rows = comparison[comparison["phase"].astype(str).str.startswith("idea")]
    if not idea_rows.empty:
        best = idea_rows.sort_values(["f_3p0", "f_0p5"], ascending=False).iloc[0]
        lines.append(f"- Mejor idea nueva fold-CV: `{best.exp_id}` con F@3s={best.f_3p0:.3f}.")
    descriptions = {
        "I01": "decoder con recompensa de repeticion/estado latente",
        "I02": "novelty por test estadistico local",
        "I03": "SSM multiescala beat/frase",
        "I04": "margen local del DP como confianza",
        "I05": "gating no lineal AD01/learned",
    }
    for prefix, text in descriptions.items():
        local = comparison[comparison["exp_id"].str.startswith(prefix)]
        if not local.empty:
            row = local.sort_values(["f_3p0", "f_0p5"], ascending=False).iloc[0]
            lines.append(f"- {prefix}: {text}; mejor F@3s={row.f_3p0:.3f}.")
    lines.extend(["", "## Decision", ""])
    lines.append("Si una idea supera a AD01 pre-registrada, conviene aislarla y pre-registrar su mejor configuracion fija. Si no, buscar combinaciones donde AD01 siga como geometria principal y las nuevas senales operen como confianza o reranking.")
    lines.append("")
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

    print("Preparing AD01 baseline", flush=True)
    ad01_matrices, ad01_ssm_df = ad01_for_tracks(tracks)
    ad01_stability = sgdp.stability_curves_for_tracks(tracks, ad01_matrices)
    baseline_df = ad01_preregistered_dp(tracks, ad01_matrices, ad01_stability)
    ad01_ssm_df.to_csv(args.out_dir / "AD01_PR00_ssm_preregistered_metrics.csv", index=False)
    baseline_df.to_csv(args.out_dir / "AD01_PR01_sgdp_preregistered_metrics.csv", index=False)

    state_sel, stat_sel, multi_sel, margin_sel, idea_frames = run_first_four_ideas(
        tracks,
        ad01_matrices,
        ad01_stability,
        baseline_df,
        args.out_dir,
    )
    gate_sel = run_idea5(tracks, ad01_matrices, ad01_stability, baseline_df, args.out_dir, args.samples_per_class, args.folds)
    idea_frames.append(gate_sel)

    frames = []
    for phase, frame in [
        ("ad01_ssm", ad01_ssm_df),
        ("ad01_sgdp", baseline_df),
        ("idea1_state_decoder", state_sel),
        ("idea2_statistical_novelty", stat_sel),
        ("idea3_multiscale_ssm", multi_sel),
        ("idea4_dp_margin", margin_sel),
        ("idea5_nonlinear_gate", gate_sel),
    ]:
        agg = lit.aggregate(frame)
        agg["phase"] = phase
        frames.append(agg)
    previous = previous_rows()
    comparison = pd.concat([previous, *frames], ignore_index=True, sort=False)
    comparison = comparison.sort_values(["f_3p0", "f_0p5"], ascending=False).reset_index(drop=True)
    comparison.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    plot_comparison(comparison, args.out_dir / "aggregate_metrics.png")
    sgdp.comparison_vs_g21(args.out_dir, [ad01_ssm_df, baseline_df, *idea_frames])
    write_summary(args.out_dir, comparison, idea_frames)
    print(comparison[["exp_id", "phase", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
