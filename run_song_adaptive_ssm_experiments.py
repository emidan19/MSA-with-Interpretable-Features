#!/usr/bin/env python3
"""Song-adaptive SSM metrics for MFCC/density embeddings.

Tests three per-song metric choices:

1. Robust diagonal whitening + RBF.
2. Shrunk per-song Mahalanobis + RBF.
3. Local self-tuning kernel.

Each metric is evaluated as an MFCC/density SSM feeding the strongest current
decoder branch: global DP guided by hierarchical stability confidence.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
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
from sklearn.covariance import LedoitWolf

import run_literature_guided_iterations as lit
import run_stability_guided_dp_iteration as sgdp
from run_similarity_metric_experiments import BEAT_KEYS
from run_structure_baseline import normalize_ssm


OUT_DIR = Path("structure_outputs/rwc_p_100_song_adaptive_ssm_v1")

METRICS = {
    "AD01_robust_diag": "Robust per-song diagonal whitening + RBF",
    "AD02_mahalanobis_shrink": "Per-song Ledoit-Wolf Mahalanobis + RBF",
    "AD03_self_tuning": "Per-song local self-tuning kernel",
}
DENSITY_WEIGHTS = (0.55, 0.65, 0.75)


def safe_features(features: np.ndarray) -> np.ndarray:
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def robust_standardize(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = safe_features(features).T
    med = np.median(x, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    scale = np.where(mad > eps, mad, np.maximum(std, eps))
    z = (x - med) / scale
    z = np.clip(z, -8.0, 8.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pairwise_sqeuclidean(x: np.ndarray) -> np.ndarray:
    norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (x @ x.T)
    return np.maximum(d2, 0.0).astype(np.float32)


def rbf_from_d2(d2: np.ndarray, percentile: float = 35.0) -> np.ndarray:
    positive = d2[d2 > 1e-8]
    scale = float(np.percentile(positive, percentile)) if positive.size else 1.0
    return normalize_ssm(np.exp(-d2 / max(scale, 1e-6)).astype(np.float32))


def robust_diag_rbf_ssm(features: np.ndarray) -> np.ndarray:
    x = robust_standardize(features)
    d2 = pairwise_sqeuclidean(x) / max(1, x.shape[1])
    return rbf_from_d2(d2, percentile=30.0)


def mahalanobis_shrink_rbf_ssm(features: np.ndarray) -> np.ndarray:
    x = robust_standardize(features)
    if x.shape[0] <= x.shape[1] + 2:
        return robust_diag_rbf_ssm(features)
    try:
        lw = LedoitWolf(assume_centered=False).fit(x)
        precision = np.asarray(lw.precision_, dtype=np.float32)
    except Exception:
        return robust_diag_rbf_ssm(features)
    transformed = x @ precision
    q = np.sum(transformed * x, axis=1, keepdims=True)
    d2 = q + q.T - 2.0 * (transformed @ x.T)
    d2 = np.maximum(d2 / max(1, x.shape[1]), 0.0).astype(np.float32)
    return rbf_from_d2(d2, percentile=30.0)


def self_tuning_ssm(features: np.ndarray, k: int = 7) -> np.ndarray:
    x = robust_standardize(features)
    d2 = pairwise_sqeuclidean(x) / max(1, x.shape[1])
    n = d2.shape[0]
    if n <= 2:
        return normalize_ssm(np.ones((n, n), dtype=np.float32))
    kk = int(np.clip(k, 1, n - 1))
    masked = d2 + np.eye(n, dtype=np.float32) * 1e9
    kth = np.partition(masked, kk, axis=1)[:, kk]
    sigma = np.sqrt(np.maximum(kth, 1e-6))
    denom = np.maximum(sigma[:, None] * sigma[None, :], 1e-6)
    return normalize_ssm(np.exp(-d2 / denom).astype(np.float32))


def adaptive_stream_ssm(metric_id: str, features: np.ndarray) -> np.ndarray:
    if metric_id == "AD01_robust_diag":
        return robust_diag_rbf_ssm(features)
    if metric_id == "AD02_mahalanobis_shrink":
        return mahalanobis_shrink_rbf_ssm(features)
    if metric_id == "AD03_self_tuning":
        return self_tuning_ssm(features)
    raise ValueError(f"Unknown metric_id: {metric_id}")


def targeted_stability_configs() -> list[sgdp.StabilityDPConfig]:
    return [
        sgdp.StabilityDPConfig("t14_h1.0_n0.9_r0.20_s0.20", 14.0, 5.5, 42.0, 1.0, 0.9, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.20", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.40", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.40),
        sgdp.StabilityDPConfig("t14_h1.0_n0.9_r0.20_s0.40", 14.0, 5.5, 42.0, 1.0, 0.9, 0.20, 0.40),
        sgdp.StabilityDPConfig("t14_h0.6_n1.2_r0.20_s0.20", 14.0, 5.5, 42.0, 0.6, 1.2, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h1.0_n1.2_r0.20_s0.20", 14.0, 5.5, 42.0, 1.0, 1.2, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.60", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.60),
        sgdp.StabilityDPConfig("t16_h1.0_n0.9_r0.20_s0.20", 16.0, 5.5, 42.0, 1.0, 0.9, 0.20, 0.20),
    ]


def build_adaptive_candidates(
    tracks: list[lit.TrackInfo],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, float], np.ndarray], dict[tuple[str, str, float], np.ndarray]]:
    ssm_rows = []
    dp_rows = []
    matrices: dict[tuple[str, str, float], np.ndarray] = {}
    stability_curves: dict[tuple[str, str, float], np.ndarray] = {}
    configs = targeted_stability_configs()
    pd.DataFrame([asdict(config) for config in configs]).to_csv(out_dir / "targeted_stability_dp_config_grid.csv", index=False)
    pd.DataFrame(
        [
            {"metric_id": metric_id, "description": description, "density_weight": density_weight}
            for metric_id, description in METRICS.items()
            for density_weight in DENSITY_WEIGHTS
        ]
    ).to_csv(out_dir / "adaptive_metric_grid.csv", index=False)

    for track_idx, track in enumerate(tracks, 1):
        with np.load(track.feature_path, allow_pickle=True) as data:
            stream_ssms = {
                metric_id: {
                    "mfcc": adaptive_stream_ssm(metric_id, np.asarray(data[BEAT_KEYS["mfcc"]], dtype=np.float32)),
                    "density": adaptive_stream_ssm(metric_id, np.asarray(data[BEAT_KEYS["density"]], dtype=np.float32)),
                }
                for metric_id in METRICS
            }
        for metric_id in METRICS:
            for density_weight in DENSITY_WEIGHTS:
                matrix, weights = lit.reliability_fuse(
                    stream_ssms[metric_id],
                    {"mfcc": 1.0 - density_weight, "density": density_weight},
                )
                key = (track.track_id, metric_id, float(density_weight))
                matrices[key] = matrix
                ssm_rows.append(
                    lit.evaluate_matrix(
                        track,
                        matrix,
                        "ADSSM_candidate",
                        extra={
                            "metric_id": metric_id,
                            "density_weight": density_weight,
                            "weights": json.dumps(weights, sort_keys=True),
                        },
                    )
                )
                stability = sgdp.stability_curves_for_tracks([track], {track.track_id: matrix})[track.track_id]
                stability_curves[key] = stability
                for config in configs:
                    indices = sgdp.stability_guided_dp_indices(matrix, track.beat_boundaries, stability, config)
                    dp_rows.append(
                        lit.evaluate_indices(
                            track,
                            indices,
                            "ADSGDP_candidate",
                            extra={
                                "metric_id": metric_id,
                                "density_weight": density_weight,
                                "config_id": config.config_id,
                            },
                        )
                    )
        if track_idx % 10 == 0:
            print(f"  adaptive metrics scored: {track_idx}/{len(tracks)}", flush=True)
    ssm_df = pd.DataFrame(ssm_rows)
    dp_df = pd.DataFrame(dp_rows)
    ssm_df.to_csv(out_dir / "adaptive_ssm_candidate_scores.csv", index=False)
    dp_df.to_csv(out_dir / "adaptive_sgdp_candidate_scores.csv", index=False)
    return ssm_df, dp_df, matrices, stability_curves


def select_per_metric(
    candidate_df: pd.DataFrame,
    tracks: list[lit.TrackInfo],
    group_cols: list[str],
    exp_suffix: str,
) -> pd.DataFrame:
    rows = []
    for metric_id in METRICS:
        local = candidate_df[candidate_df["metric_id"] == metric_id]
        selected, chosen = sgdp.fold_select(local, tracks, group_cols, f"{metric_id}_{exp_suffix}")
        selected["metric_id"] = metric_id
        chosen.insert(0, "metric_id", metric_id)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def aggregate_candidates(out_dir: Path, ssm_df: pd.DataFrame, dp_df: pd.DataFrame) -> None:
    ssm_agg = (
        ssm_df.groupby(["metric_id", "density_weight"], as_index=False)
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
    ssm_agg.to_csv(out_dir / "adaptive_ssm_candidate_aggregates.csv", index=False)
    dp_agg = (
        dp_df.groupby(["metric_id", "density_weight", "config_id"], as_index=False)
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
    dp_agg.to_csv(out_dir / "adaptive_sgdp_candidate_aggregates.csv", index=False)


def previous_rows() -> pd.DataFrame:
    rows = []
    previous = sgdp.previous_rows()
    if not previous.empty:
        rows.append(previous)
    path = Path("structure_outputs/rwc_p_100_stability_guided_dp_v1/aggregate_metrics.csv")
    if path.exists():
        df = pd.read_csv(path)
        keep = df[df["exp_id"].isin(["SGDP04_stability_prior_dp_foldcv", "SGDP02_confidence_filter_loo"])]
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
        elif exp_id.startswith("SGDP04") or exp_id.startswith("SGDP02"):
            colors.append("#059669")
        elif exp_id.startswith("AD01"):
            colors.append("#2563EB")
        elif exp_id.startswith("AD02"):
            colors.append("#C2410C")
        elif exp_id.startswith("AD03"):
            colors.append("#DB2777")
        elif exp_id.startswith("G21"):
            colors.append("#64748B")
        else:
            colors.append("#94A3B8")
    fig, ax = plt.subplots(figsize=(12.2, 7.8), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, shown["f_3p0"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shown["exp_id"], fontsize=8)
    ax.set_xlabel("F@3s")
    ax.set_title("Metricas SSM adaptativas por cancion")
    ax.grid(axis="x", alpha=0.25)
    lo = max(0.48, float(shown["f_3p0"].min()) - 0.02)
    hi = min(0.73, float(shown["f_3p0"].max()) + 0.02)
    ax.set_xlim(lo, hi)
    for i, row in shown.reset_index(drop=True).iterrows():
        ax.text(float(row["f_3p0"]) + 0.002, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary(out_dir: Path, comparison: pd.DataFrame, ssm_selected: pd.DataFrame, dp_selected: pd.DataFrame) -> None:
    lines = [
        "# Metricas adaptativas por cancion para SSM",
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
    adaptive = comparison[comparison["phase"].astype(str).str.contains("adaptive")]
    if not adaptive.empty:
        best = adaptive.sort_values(["f_3p0", "f_0p5"], ascending=False).iloc[0]
        lines.append(f"- Mejor metrica adaptativa: `{best.exp_id}` con F@3s={best.f_3p0:.3f}.")
    lines.append("- AD01 prueba una normalizacion robusta por dimension; es la variante mas conservadora.")
    lines.append("- AD02 prueba una metrica Mahalanobis shrinkage por cancion; puede capturar correlaciones entre dimensiones, pero puede suavizar demasiado.")
    lines.append("- AD03 prueba un kernel self-tuning local; adapta la escala segun la densidad local del espacio de embeddings.")
    previous_best = comparison[comparison["exp_id"] == "SGDP04_stability_prior_dp_foldcv"]
    ad01 = comparison[comparison["exp_id"] == "AD01_robust_diag_sgdp_foldcv"]
    if not previous_best.empty and not ad01.empty:
        delta = float(ad01.iloc[0]["f_3p0"] - previous_best.iloc[0]["f_3p0"])
        lines.append(f"- AD01 queda a {delta:+.4f} F@3s del mejor sistema previo, con una implementacion no supervisada por pares.")
    lines.extend(["", "## Configuraciones seleccionadas por metrica"])
    for metric_id in METRICS:
        local = dp_selected[dp_selected["metric_id"] == metric_id]
        if local.empty:
            continue
        lines.append(f"{metric_id}:")
        for key, value in local["config_id"].value_counts().head(4).items():
            lines.append(f"- {value} tracks con `{key}`.")
    lines.extend(["", "## Decision", ""])
    lines.append("Si ninguna metrica adaptativa supera a la SSM aprendida previa, la conclusion es que la adaptacion estadistica por cancion debe usarse como feature adicional o como calibracion, no como reemplazo de la metrica aprendida.")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = lit.load_tracks(args.limit, args.folds)
    print(f"Loaded {len(tracks)} tracks in {args.folds} folds", flush=True)
    ssm_df, dp_df, _matrices, _stability = build_adaptive_candidates(tracks, args.out_dir)
    aggregate_candidates(args.out_dir, ssm_df, dp_df)

    ssm_selected = select_per_metric(ssm_df, tracks, ["density_weight"], "ssm_foldcv")
    ssm_selected.to_csv(args.out_dir / "adaptive_ssm_foldcv_metrics.csv", index=False)
    dp_selected = select_per_metric(dp_df, tracks, ["density_weight", "config_id"], "sgdp_foldcv")
    dp_selected.to_csv(args.out_dir / "adaptive_sgdp_foldcv_metrics.csv", index=False)
    all_dp_selected, all_dp_configs = sgdp.fold_select(
        dp_df,
        tracks,
        ["metric_id", "density_weight", "config_id"],
        "AD00_best_adaptive_sgdp_foldcv",
    )
    all_dp_selected.to_csv(args.out_dir / "best_adaptive_sgdp_foldcv_metrics.csv", index=False)
    all_dp_configs.to_csv(args.out_dir / "best_adaptive_sgdp_foldcv_chosen_configs.csv", index=False)

    frames = []
    for phase, frame in [
        ("adaptive_ssm", ssm_selected),
        ("adaptive_sgdp", dp_selected),
        ("adaptive_sgdp_best", all_dp_selected),
    ]:
        agg = lit.aggregate(frame)
        agg["phase"] = phase
        frames.append(agg)
    previous = previous_rows()
    comparison = pd.concat([previous, *frames], ignore_index=True, sort=False)
    comparison = comparison.sort_values(["f_3p0", "f_0p5"], ascending=False).reset_index(drop=True)
    comparison.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    plot_comparison(comparison, args.out_dir / "aggregate_metrics.png")
    sgdp.comparison_vs_g21(args.out_dir, [ssm_selected, dp_selected, all_dp_selected])
    write_summary(args.out_dir, comparison, ssm_selected, dp_selected)
    print(comparison[["exp_id", "phase", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
