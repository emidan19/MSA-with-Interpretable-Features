#!/usr/bin/env python3
"""Pre-registered AD01 and AD01 + learned-SSM fusion experiments."""

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

import run_literature_guided_iterations as lit
import run_song_adaptive_ssm_experiments as ad
import run_stability_guided_dp_iteration as sgdp
from run_similarity_metric_experiments import BEAT_KEYS
from run_structure_baseline import normalize_ssm


OUT_DIR = Path("structure_outputs/rwc_p_100_ad01_learned_fusion_v1")
AD01_DENSITY_WEIGHT = 0.55
AD01_CONFIG = sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.40", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.40)
FUSION_AD01_WEIGHTS = (0.15, 0.30, 0.45, 0.60, 0.75)


def ad01_matrix(track: lit.TrackInfo) -> tuple[np.ndarray, dict[str, float]]:
    with np.load(track.feature_path, allow_pickle=True) as data:
        matrices = {
            "mfcc": ad.robust_diag_rbf_ssm(np.asarray(data[BEAT_KEYS["mfcc"]], dtype=np.float32)),
            "density": ad.robust_diag_rbf_ssm(np.asarray(data[BEAT_KEYS["density"]], dtype=np.float32)),
        }
    return lit.reliability_fuse(matrices, {"mfcc": 1.0 - AD01_DENSITY_WEIGHT, "density": AD01_DENSITY_WEIGHT})


def fusion_configs() -> list[sgdp.StabilityDPConfig]:
    return [
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.40", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.40),
        sgdp.StabilityDPConfig("t14_h1.0_n0.9_r0.20_s0.20", 14.0, 5.5, 42.0, 1.0, 0.9, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.20", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.20),
        sgdp.StabilityDPConfig("t14_h1.0_n0.9_r0.20_s0.40", 14.0, 5.5, 42.0, 1.0, 0.9, 0.20, 0.40),
        sgdp.StabilityDPConfig("t14_h0.6_n0.9_r0.20_s0.60", 14.0, 5.5, 42.0, 0.6, 0.9, 0.20, 0.60),
        sgdp.StabilityDPConfig("t14_h0.6_n1.2_r0.20_s0.20", 14.0, 5.5, 42.0, 0.6, 1.2, 0.20, 0.20),
    ]


def build_matrices(
    tracks: list[lit.TrackInfo],
    models: dict[int, dict[str, object]],
    out_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    learned: dict[str, np.ndarray] = {}
    adaptive: dict[str, np.ndarray] = {}
    learned_rows = []
    ad01_rows = []
    for idx, track in enumerate(tracks, 1):
        learned_matrix, learned_weights = sgdp.learned_metric_matrix(track, models)
        adaptive_matrix, adaptive_weights = ad01_matrix(track)
        learned[track.track_id] = learned_matrix
        adaptive[track.track_id] = adaptive_matrix
        learned_rows.append(
            lit.evaluate_matrix(
                track,
                learned_matrix,
                "ML03_pair_metric_preregistered",
                extra={"weights": json.dumps(learned_weights, sort_keys=True)},
            )
        )
        ad01_rows.append(
            lit.evaluate_matrix(
                track,
                adaptive_matrix,
                "AD01_PR00_ssm_preregistered",
                extra={"density_weight": AD01_DENSITY_WEIGHT, "weights": json.dumps(adaptive_weights, sort_keys=True)},
            )
        )
        if idx % 10 == 0:
            print(f"  matrices rebuilt: {idx}/{len(tracks)}", flush=True)
    learned_df = pd.DataFrame(learned_rows)
    ad01_df = pd.DataFrame(ad01_rows)
    learned_df.to_csv(out_dir / "ML03_pair_metric_preregistered_metrics.csv", index=False)
    ad01_df.to_csv(out_dir / "AD01_PR00_ssm_preregistered_metrics.csv", index=False)
    return learned, adaptive, learned_df, ad01_df


def run_ad01_preregistered(
    tracks: list[lit.TrackInfo],
    ad01_matrices: dict[str, np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for idx, track in enumerate(tracks, 1):
        matrix = ad01_matrices[track.track_id]
        stability = sgdp.stability_curves_for_tracks([track], {track.track_id: matrix})[track.track_id]
        indices = sgdp.stability_guided_dp_indices(matrix, track.beat_boundaries, stability, AD01_CONFIG)
        rows.append(
            lit.evaluate_indices(
                track,
                indices,
                "AD01_PR01_sgdp_preregistered",
                extra={
                    "density_weight": AD01_DENSITY_WEIGHT,
                    "config_id": AD01_CONFIG.config_id,
                },
            )
        )
        if idx % 10 == 0:
            print(f"  AD01 preregistered DP: {idx}/{len(tracks)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "AD01_PR01_sgdp_preregistered_metrics.csv", index=False)
    return df


def run_fusion_candidates(
    tracks: list[lit.TrackInfo],
    learned_matrices: dict[str, np.ndarray],
    ad01_matrices: dict[str, np.ndarray],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = fusion_configs()
    pd.DataFrame([asdict(config) for config in configs]).to_csv(out_dir / "fusion_sgdp_config_grid.csv", index=False)
    pd.DataFrame([{"ad01_weight": weight} for weight in FUSION_AD01_WEIGHTS]).to_csv(out_dir / "fusion_weight_grid.csv", index=False)
    ssm_rows = []
    dp_rows = []
    for idx, track in enumerate(tracks, 1):
        learned = learned_matrices[track.track_id]
        adaptive = ad01_matrices[track.track_id]
        for ad01_weight in FUSION_AD01_WEIGHTS:
            matrix = normalize_ssm((1.0 - ad01_weight) * learned + ad01_weight * adaptive)
            ssm_rows.append(
                lit.evaluate_matrix(
                    track,
                    matrix,
                    "FUS_candidate_ssm",
                    extra={"ad01_weight": ad01_weight},
                )
            )
            stability = sgdp.stability_curves_for_tracks([track], {track.track_id: matrix})[track.track_id]
            for config in configs:
                indices = sgdp.stability_guided_dp_indices(matrix, track.beat_boundaries, stability, config)
                dp_rows.append(
                    lit.evaluate_indices(
                        track,
                        indices,
                        "FUS_candidate_sgdp",
                        extra={"ad01_weight": ad01_weight, "config_id": config.config_id},
                    )
                )
        if idx % 10 == 0:
            print(f"  fusion candidates: {idx}/{len(tracks)}", flush=True)
    ssm_df = pd.DataFrame(ssm_rows)
    dp_df = pd.DataFrame(dp_rows)
    ssm_df.to_csv(out_dir / "fusion_ssm_candidate_scores.csv", index=False)
    dp_df.to_csv(out_dir / "fusion_sgdp_candidate_scores.csv", index=False)
    return ssm_df, dp_df


def write_candidate_aggregates(out_dir: Path, ssm_df: pd.DataFrame, dp_df: pd.DataFrame) -> None:
    ssm_agg = (
        ssm_df.groupby("ad01_weight", as_index=False)
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
    ssm_agg.to_csv(out_dir / "fusion_ssm_candidate_aggregates.csv", index=False)
    dp_agg = (
        dp_df.groupby(["ad01_weight", "config_id"], as_index=False)
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
    dp_agg.to_csv(out_dir / "fusion_sgdp_candidate_aggregates.csv", index=False)
    if not dp_agg.empty:
        best = dp_agg.iloc[0]
        mask = (dp_df["ad01_weight"] == best["ad01_weight"]) & (dp_df["config_id"] == best["config_id"])
        dp_df[mask].assign(exp_id="FUS03_best_fixed_diagnostic").to_csv(
            out_dir / "FUS03_best_fixed_diagnostic_metrics.csv",
            index=False,
        )


def previous_rows() -> pd.DataFrame:
    rows = []
    previous = sgdp.previous_rows()
    if not previous.empty:
        rows.append(previous)
    for path, keep_ids in [
        (
            Path("structure_outputs/rwc_p_100_stability_guided_dp_v1/aggregate_metrics.csv"),
            ["SGDP04_stability_prior_dp_foldcv", "SGDP02_confidence_filter_loo"],
        ),
        (
            Path("structure_outputs/rwc_p_100_song_adaptive_ssm_v1/aggregate_metrics.csv"),
            ["AD01_robust_diag_sgdp_foldcv"],
        ),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            keep = df[df["exp_id"].isin(keep_ids)].copy()
            if not keep.empty:
                keep["phase"] = "previous_best"
                rows.append(keep[["exp_id", "f_0p5", "f_3p0", "precision_3p0", "recall_3p0", "pred_count", "tracks", "phase"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    shown = comparison.sort_values("f_3p0", ascending=True)
    colors = []
    for exp_id in shown["exp_id"]:
        if "ORACLE" in exp_id.upper():
            colors.append("#7C3AED")
        elif exp_id.startswith("FUS"):
            colors.append("#C2410C")
        elif exp_id.startswith("AD01_PR"):
            colors.append("#2563EB")
        elif exp_id.startswith("AD01_robust") or exp_id.startswith("SGDP"):
            colors.append("#059669")
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
    ax.set_title("AD01 pre-registrada y fusion con SSM aprendida")
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
    fusion_selected: pd.DataFrame,
    fusion_agg: pd.DataFrame,
) -> None:
    lines = [
        "# AD01 pre-registrada y fusion con SSM aprendida",
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
    for exp_id in ("AD01_PR01_sgdp_preregistered", "FUS01_ad01_learned_sgdp_foldcv"):
        local = comparison[comparison["exp_id"] == exp_id]
        if not local.empty:
            row = local.iloc[0]
            lines.append(f"- `{exp_id}`: F@3s={row.f_3p0:.3f}, F@0.5s={row.f_0p5:.3f}.")
    if not fusion_selected.empty:
        lines.extend(["", "## Fusion seleccionada por fold"])
        for key, value in fusion_selected["ad01_weight"].value_counts().sort_index().items():
            lines.append(f"- ad01_weight={key:.2f}: {value} tracks.")
        for key, value in fusion_selected["config_id"].value_counts().head(5).items():
            lines.append(f"- {value} tracks con `{key}`.")
    if not fusion_agg.empty:
        best = fusion_agg.iloc[0]
        lines.append(
            f"- Mejor diagnostico fijo del grid: ad01_weight={best.ad01_weight:.2f}, "
            f"`{best.config_id}`, F@3s={best.f_3p0:.3f}."
        )
    lines.extend(["", "## Decision", ""])
    lines.append("Si la fusion no supera a AD01 pre-registrada, conviene mantener AD01 como alternativa simple y usar la SSM aprendida como feature adicional para un selector/gating posterior, no como mezcla lineal directa.")
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

    print("Training learned pair-label SSM models", flush=True)
    samples = lit.precompute_pair_samples(tracks, args.samples_per_class)
    models = lit.fold_models(tracks, samples, args.folds)

    print("Rebuilding learned and AD01 matrices", flush=True)
    learned_matrices, ad01_matrices, learned_df, ad01_ssm_df = build_matrices(tracks, models, args.out_dir)

    print("Running AD01 pre-registered DP", flush=True)
    ad01_pr_df = run_ad01_preregistered(tracks, ad01_matrices, args.out_dir)

    print("Running AD01 + learned fusion candidates", flush=True)
    fusion_ssm_df, fusion_dp_df = run_fusion_candidates(tracks, learned_matrices, ad01_matrices, args.out_dir)
    write_candidate_aggregates(args.out_dir, fusion_ssm_df, fusion_dp_df)
    fusion_ssm_selected, fusion_ssm_configs = sgdp.fold_select(
        fusion_ssm_df,
        tracks,
        ["ad01_weight"],
        "FUS00_ad01_learned_ssm_foldcv",
    )
    fusion_ssm_selected.to_csv(args.out_dir / "FUS00_ad01_learned_ssm_foldcv_metrics.csv", index=False)
    fusion_ssm_configs.to_csv(args.out_dir / "FUS00_ad01_learned_ssm_foldcv_chosen_configs.csv", index=False)
    fusion_selected, fusion_configs_df = sgdp.fold_select(
        fusion_dp_df,
        tracks,
        ["ad01_weight", "config_id"],
        "FUS01_ad01_learned_sgdp_foldcv",
    )
    fusion_selected.to_csv(args.out_dir / "FUS01_ad01_learned_sgdp_foldcv_metrics.csv", index=False)
    fusion_configs_df.to_csv(args.out_dir / "FUS01_ad01_learned_sgdp_foldcv_chosen_configs.csv", index=False)

    frames = []
    for phase, frame in [
        ("learned_ssm", learned_df),
        ("ad01_preregistered_ssm", ad01_ssm_df),
        ("ad01_preregistered_sgdp", ad01_pr_df),
        ("fusion_ssm", fusion_ssm_selected),
        ("fusion_sgdp", fusion_selected),
    ]:
        agg = lit.aggregate(frame)
        agg["phase"] = phase
        frames.append(agg)
    previous = previous_rows()
    comparison = pd.concat([previous, *frames], ignore_index=True, sort=False)
    comparison = comparison.sort_values(["f_3p0", "f_0p5"], ascending=False).reset_index(drop=True)
    comparison.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    plot_comparison(comparison, args.out_dir / "aggregate_metrics.png")
    sgdp.comparison_vs_g21(args.out_dir, [learned_df, ad01_ssm_df, ad01_pr_df, fusion_ssm_selected, fusion_selected])
    fusion_agg = pd.read_csv(args.out_dir / "fusion_sgdp_candidate_aggregates.csv")
    write_summary(args.out_dir, comparison, fusion_selected, fusion_agg)

    print(comparison[["exp_id", "phase", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
