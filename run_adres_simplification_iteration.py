#!/usr/bin/env python3
"""Simplify ADRES01 while preserving most of its performance.

This script reuses the ADRES01 candidate table and tests:

1. Hand-scored candidate rankings with reduced source sets.
2. Small/out-of-fold model ablations that remove curves, sources, or RF
   complexity.

The goal is to identify what matters and find a simpler variant close to
ADRES01 and above AD01.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import run_ad01_candidate_rescue_iteration as adres
import run_literature_guided_iterations as lit
import run_stability_guided_dp_iteration as sgdp


RES_DIR = Path("structure_outputs/rwc_p_100_ad01_candidate_rescue_v1")
DIS_DIR = Path("structure_outputs/rwc_p_100_disruptive_ideas_v1")
OUT_DIR = Path("structure_outputs/rwc_p_100_adres_simplification_v1")


@dataclass(frozen=True)
class SimpleConfig:
    config_id: str
    source_set: str
    curve_mode: str
    ad_bonus: float
    support_w: float
    idea_w: float
    curve_w: float
    i04_w: float
    dist_w: float
    target_delta: int
    min_segment_s: float = 6.5


@dataclass(frozen=True)
class ModelVariant:
    model_id: str
    family: str
    source_set: str
    feature_mode: str


def parse_indices(value: str) -> np.ndarray:
    try:
        return np.asarray(json.loads(value), dtype=np.int64)
    except Exception:
        return np.zeros(0, dtype=np.int64)


def load_tracks(limit: int, folds: int) -> list[lit.TrackInfo]:
    return lit.load_tracks(limit, folds)


def load_predictions() -> dict[tuple[str, str], np.ndarray]:
    frames = adres.load_source_predictions(DIS_DIR)
    return adres.predictions_by_source(frames)


def source_sets() -> dict[str, tuple[str, ...]]:
    return {
        "ad01_only": ("ad01",),
        "ad01_i03": ("ad01", "i03"),
        "ad01_i02_i03": ("ad01", "i02", "i03"),
        "ad01_i02_i03_i04": ("ad01", "i02", "i03", "i04"),
        "ad01_i02_i03_i05": ("ad01", "i02", "i03", "i05"),
        "full": ("ad01", "i01", "i02", "i03", "i04", "i05"),
    }


def filter_source_set(candidates: pd.DataFrame, source_set: str) -> pd.DataFrame:
    sources = source_sets()[source_set]
    cols = [f"{src}_near" for src in sources if f"{src}_near" in candidates.columns]
    if not cols:
        return candidates.iloc[0:0].copy()
    mask = candidates[cols].sum(axis=1) > 0
    return candidates[mask].copy()


def curve_value(local: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(0.0, index=local.index)
    if mode == "ad_conf":
        return local["ad_conf_local"]
    if mode == "stat_multi":
        return 0.5 * local["stat_local"] + 0.5 * local["multi_local"]
    if mode == "mean":
        return local["curve_mean"]
    if mode == "max":
        return local["curve_max"]
    raise ValueError(mode)


def simple_configs() -> list[SimpleConfig]:
    configs: list[SimpleConfig] = []
    for source_set in ("ad01_only", "ad01_i03", "ad01_i02_i03", "ad01_i02_i03_i05", "full"):
        for curve_mode in ("none", "ad_conf", "stat_multi", "mean", "max"):
            for ad_bonus in (0.20, 0.35, 0.50):
                for support_w in (0.00, 0.10, 0.22):
                    for idea_w in (0.00, 0.10, 0.20):
                        for curve_w in (0.00, 0.10, 0.22):
                            if curve_mode == "none" and curve_w > 0:
                                continue
                            for target_delta in (-1, 0, 1):
                                configs.append(
                                    SimpleConfig(
                                        config_id=(
                                            f"simple_{source_set}_{curve_mode}"
                                            f"_a{ad_bonus:.2f}_s{support_w:.2f}_i{idea_w:.2f}"
                                            f"_c{curve_w:.2f}_d{target_delta:+d}"
                                        ),
                                        source_set=source_set,
                                        curve_mode=curve_mode,
                                        ad_bonus=ad_bonus,
                                        support_w=support_w,
                                        idea_w=idea_w,
                                        curve_w=curve_w,
                                        i04_w=0.08,
                                        dist_w=0.04,
                                        target_delta=target_delta,
                                    )
                                )
    return configs


def score_simple(local: pd.DataFrame, config: SimpleConfig) -> pd.Series:
    support_norm = local["support_near"] / 6.0
    idea_norm = local["support_ideas_near"] / 3.0
    score = (
        config.ad_bonus * local["ad01_exact"]
        + config.support_w * support_norm
        + config.idea_w * idea_norm
        + config.curve_w * curve_value(local, config.curve_mode)
        + config.i04_w * local.get("i04_near", 0.0)
        - config.dist_w * local["ad01_dist_s"]
    )
    return score.astype(float)


def evaluate_simple_configs(
    tracks: list[lit.TrackInfo],
    candidates: pd.DataFrame,
    pred_lookup: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    print("Evaluating hand-scored simplifications", flush=True)
    configs = simple_configs()
    pd.DataFrame([cfg.__dict__ for cfg in configs]).to_csv(out_dir / "simple_config_grid.csv", index=False)
    rows = []
    track_groups = {track_id: group for track_id, group in candidates.groupby("track_id")}
    for cfg_idx, cfg in enumerate(configs, 1):
        for track in tracks:
            local = filter_source_set(track_groups[track.track_id], cfg.source_set)
            if local.empty:
                pred = pred_lookup[(track.track_id, "ad01")]
            else:
                local = local.copy()
                local["prob_boundary"] = score_simple(local, cfg)
                local["ad01_exact"] = local["ad01_exact"].astype(float)
                target = max(2, len(pred_lookup[(track.track_id, "ad01")]) + cfg.target_delta)
                pred = adres.rank_select(
                    local,
                    track.beat_boundaries,
                    adres.SelectionConfig(cfg.config_id, "rank", "simple", cfg.min_segment_s, 0.0, target_delta=cfg.target_delta),
                    target,
                )
            rows.append(
                lit.evaluate_indices(
                    track,
                    pred,
                    "SIMP_candidate_simple_rule",
                    extra={
                        "config_id": cfg.config_id,
                        "source_set": cfg.source_set,
                        "curve_mode": cfg.curve_mode,
                        "family": "simple_rule",
                    },
                )
            )
        if cfg_idx % 250 == 0:
            print(f"  simple configs: {cfg_idx}/{len(configs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "simple_rule_candidate_scores.csv", index=False)
    return df


def feature_modes() -> dict[str, list[str]]:
    support = [
        "time_norm",
        "pool_size",
        "support_exact",
        "support_near",
        "support_ideas_exact",
        "support_ideas_near",
        "ad01_exact",
        "ad01_near",
        "ad01_dist_s",
        "i02_near",
        "i03_near",
        "i04_near",
        "i05_near",
        "i02_dist_s",
        "i03_dist_s",
        "i05_dist_s",
    ]
    curves = [
        "ad_conf",
        "stat_curve",
        "multi_curve",
        "ad_conf_local",
        "stat_local",
        "multi_local",
        "curve_mean",
        "curve_max",
        "curve_disagreement",
    ]
    gaps = ["left_gap_ad01_s", "right_gap_ad01_s", "min_gap_ad01_s", "max_gap_ad01_s"]
    small = ["time_norm", "support_near", "support_ideas_near", "ad01_exact", "ad01_dist_s", "ad_conf_local", "stat_local", "multi_local", "i04_near"]
    return {
        "support_only": support + gaps,
        "curves_only": ["time_norm", "ad01_exact", *curves, *gaps],
        "small_linear": small,
        "support_plus_curves": support + curves + gaps,
        "no_i05": [col for col in support + curves + gaps if not col.startswith("i05")],
    }


def model_variants() -> list[ModelVariant]:
    variants = []
    for source_set in ("ad01_i02_i03", "ad01_i02_i03_i05", "full"):
        variants.extend(
            [
                ModelVariant(f"logit_small_{source_set}", "logit", source_set, "small_linear"),
                ModelVariant(f"tree3_support_{source_set}", "tree3", source_set, "support_only"),
                ModelVariant(f"rf_support_{source_set}", "rf_small", source_set, "support_only"),
                ModelVariant(f"rf_small_{source_set}", "rf_small", source_set, "support_plus_curves"),
                ModelVariant(f"rf_nocurves_{source_set}", "rf_small", source_set, "support_only"),
            ]
        )
    variants.extend(
        [
            ModelVariant("rf_no_i05_fullpool", "rf_small", "full", "no_i05"),
            ModelVariant("extra_support_full", "extra", "full", "support_only"),
            ModelVariant("tree3_small_full", "tree3", "full", "small_linear"),
        ]
    )
    return variants


def make_model(family: str):
    if family == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, class_weight="balanced", solver="liblinear", random_state=20260508),
        )
    if family == "tree3":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=26, class_weight="balanced", random_state=20260508)
    if family == "rf_small":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=18,
            class_weight="balanced_subsample",
            max_features=0.75,
            random_state=20260508,
            n_jobs=-1,
        )
    if family == "extra":
        return ExtraTreesClassifier(
            n_estimators=160,
            max_depth=5,
            min_samples_leaf=18,
            class_weight="balanced",
            max_features=0.75,
            random_state=20260508,
            n_jobs=-1,
        )
    raise ValueError(family)


def train_model_variant_oof(candidates: pd.DataFrame, tracks: list[lit.TrackInfo], variant: ModelVariant) -> pd.DataFrame:
    fold_by_track = {track.track_id: track.fold for track in tracks}
    data = filter_source_set(candidates, variant.source_set)
    data = data.copy()
    data["fold"] = data["track_id"].map(fold_by_track)
    xcols = [col for col in feature_modes()[variant.feature_mode] if col in data.columns]
    rows = []
    for fold in sorted(data["fold"].dropna().unique()):
        train = data[data["fold"] != fold]
        test = data[data["fold"] == fold].copy()
        model = make_model(variant.family)
        x_train = train[xcols].to_numpy(dtype=float)
        y_train = train["label_3p0"].to_numpy(dtype=int)
        x_test = test[xcols].to_numpy(dtype=float)
        if variant.family in {"tree3"}:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        local = test[["track_id", "idx", "time_s", "label_3p0", "label_0p5", "ad01_exact", "support_near", "support_exact"]].copy()
        local["model_id"] = variant.model_id
        local["prob_boundary"] = prob
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


def model_selection_configs(variant: ModelVariant) -> list[adres.SelectionConfig]:
    configs: list[adres.SelectionConfig] = []
    for ad_bonus in (0.15, 0.20, 0.35):
        for target_delta in (-1, 0, 1):
            configs.append(
                adres.SelectionConfig(
                    config_id=f"{variant.model_id}_rank_b{ad_bonus:.2f}_d{target_delta:+d}",
                    strategy="rank",
                    model_id=variant.model_id,
                    min_segment_s=6.5,
                    ad01_bonus=ad_bonus,
                    target_delta=target_delta,
                )
            )
    return configs


def evaluate_model_variants(
    tracks: list[lit.TrackInfo],
    candidates: pd.DataFrame,
    pred_lookup: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Evaluating compact model ablations", flush=True)
    prob_frames = []
    metric_rows = []
    variant_rows = []
    track_by_id = {track.track_id: track for track in tracks}
    for variant_idx, variant in enumerate(model_variants(), 1):
        probs = train_model_variant_oof(candidates, tracks, variant)
        prob_frames.append(probs)
        variant_rows.append(variant.__dict__)
        grouped = {track_id: group for track_id, group in probs.groupby("track_id")}
        for cfg in model_selection_configs(variant):
            for track in tracks:
                local = grouped.get(track.track_id)
                if local is None or local.empty:
                    pred = pred_lookup[(track.track_id, "ad01")]
                else:
                    pred = adres.predict_for_config(local, pred_lookup[(track.track_id, "ad01")], track.beat_boundaries, cfg)
                metric_rows.append(
                    lit.evaluate_indices(
                        track_by_id[track.track_id],
                        pred,
                        "SIMP_candidate_model_ablation",
                        extra={
                            "config_id": cfg.config_id,
                            "model_id": variant.model_id,
                            "family": variant.family,
                            "source_set": variant.source_set,
                            "feature_mode": variant.feature_mode,
                        },
                    )
                )
        print(f"  model variants: {variant_idx}/{len(model_variants())} {variant.model_id}", flush=True)
    probs_df = pd.concat(prob_frames, ignore_index=True)
    probs_df.to_csv(out_dir / "model_ablation_probabilities.csv", index=False)
    pd.DataFrame(variant_rows).to_csv(out_dir / "model_ablation_variants.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "model_ablation_candidate_scores.csv", index=False)
    return metrics, probs_df


def aggregate_candidates(df: pd.DataFrame, group_cols: list[str], out_path: Path) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, as_index=False)
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
    agg.to_csv(out_path, index=False)
    return agg


def select_fold(df: pd.DataFrame, tracks: list[lit.TrackInfo], group_cols: list[str], exp_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, chosen = sgdp.fold_select(df, tracks, group_cols, exp_id)
    return selected, chosen


def comparison_rows(simple_sel: pd.DataFrame, model_sel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    prev = pd.read_csv(RES_DIR / "aggregate_metrics.csv")
    keep = prev[
        prev["exp_id"].isin(
            [
                "ORACLE_ad01_candidate_pool",
                "ADRES01_ad01_anchored_candidate_rescue_foldcv",
                "AD01_PR01_sgdp_preregistered",
                "FUS01_ad01_learned_sgdp_foldcv",
                "I05_nonlinear_gate_foldcv",
                "I03_multiscale_ssm_foldcv",
                "I02_statistical_novelty_foldcv",
            ]
        )
    ].copy()
    keep["phase"] = keep.get("phase", "previous")
    rows = [
        keep[["exp_id", "f_0p5", "f_3p0", "precision_3p0", "recall_3p0", "pred_count", "tracks", "phase"]],
        lit.aggregate(simple_sel).assign(phase="simple_rule_foldcv"),
        lit.aggregate(model_sel).assign(phase="compact_model_foldcv"),
    ]
    comparison = pd.concat(rows, ignore_index=True, sort=False).sort_values(["f_3p0", "f_0p5"], ascending=False)
    comparison.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    return comparison


def compare_vs(df: pd.DataFrame, baseline_path: Path, out_path: Path) -> None:
    base = pd.read_csv(baseline_path).set_index("track_id")
    rows = []
    for exp_id, local_df in df.groupby("exp_id"):
        local = local_df.set_index("track_id")
        diff = local["f_3p0"] - base["f_3p0"]
        rows.append(
            {
                "exp_id": exp_id,
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "wins": int((diff > 1e-9).sum()),
                "ties": int((np.abs(diff) <= 1e-9).sum()),
                "losses": int((diff < -1e-9).sum()),
                "best_gain": float(diff.max()),
                "worst_loss": float(diff.min()),
            }
        )
    pd.DataFrame(rows).sort_values("mean_diff", ascending=False).to_csv(out_path, index=False)


def plot_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    shown = comparison.drop_duplicates("exp_id").head(12).sort_values("f_3p0", ascending=True)
    colors = []
    for exp_id in shown["exp_id"]:
        if "ORACLE" in exp_id:
            colors.append("#7C3AED")
        elif exp_id.startswith("SIMP"):
            colors.append("#0F766E")
        elif exp_id.startswith("ADRES"):
            colors.append("#2563EB")
        elif exp_id.startswith("AD01"):
            colors.append("#64748B")
        else:
            colors.append("#94A3B8")
    fig, ax = plt.subplots(figsize=(12.0, 7.2), constrained_layout=True)
    y = np.arange(len(shown))
    ax.barh(y, shown["f_3p0"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(shown["exp_id"], fontsize=8.5)
    ax.set_xlim(max(0.64, float(shown["f_3p0"].min()) - 0.01), min(0.80, float(shown["f_3p0"].max()) + 0.015))
    ax.set_xlabel("F@3s")
    ax.set_title("Simplificación de ADRES01")
    ax.grid(axis="x", alpha=0.25)
    for i, row in shown.reset_index(drop=True).iterrows():
        ax.text(float(row["f_3p0"]) + 0.0015, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8.5)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    shown = df if max_rows is None else df.head(max_rows)
    if shown.empty:
        return "_Sin filas._"
    cols = list(shown.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in shown.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.4f}")
            elif isinstance(value, (int, np.integer)):
                vals.append(str(int(value)))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(
    comparison: pd.DataFrame,
    simple_agg: pd.DataFrame,
    model_agg: pd.DataFrame,
    simple_chosen: pd.DataFrame,
    model_chosen: pd.DataFrame,
    out_dir: Path,
) -> None:
    best_simple = lit.aggregate(pd.read_csv(out_dir / "SIMP01_simple_rule_foldcv_metrics.csv")).iloc[0]
    best_model = lit.aggregate(pd.read_csv(out_dir / "SIMP02_compact_model_foldcv_metrics.csv")).iloc[0]
    lines = [
        "# Simplificación de ADRES01",
        "",
        "## Resultado principal",
        "",
        f"- Mejor regla simple fold-CV: `{best_simple.exp_id}` F@3s={best_simple.f_3p0:.4f}.",
        f"- Mejor modelo compacto fold-CV: `{best_model.exp_id}` F@3s={best_model.f_3p0:.4f}.",
        "- Referencias: AD01=0.6781, ADRES01=0.6834, oráculo del pool=0.7829.",
        "",
        "## Lectura",
        "",
        "- Si la regla simple queda cerca de ADRES01, se puede reemplazar el ranker por una fórmula interpretable.",
        "- Si el modelo compacto supera claramente a la regla, la pieza importante es la interacción no lineal entre soporte, distancia a AD01 y curvas.",
        "- Si quitar I05 no degrada mucho, la rama aprendida previa no es imprescindible para esta etapa.",
        "",
        "## Top reglas simples",
        "",
        markdown_table(simple_agg, max_rows=12),
        "",
        "## Top modelos compactos",
        "",
        markdown_table(model_agg, max_rows=12),
        "",
        "## Configuración fold-CV regla simple",
        "",
        markdown_table(simple_chosen),
        "",
        "## Configuración fold-CV modelo compacto",
        "",
        markdown_table(model_chosen),
        "",
    ]
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
    tracks = load_tracks(args.limit, args.folds)
    candidates = pd.read_csv(RES_DIR / "boundary_candidate_features.csv")
    candidates = candidates[candidates["track_id"].isin([track.track_id for track in tracks])].copy()
    pred_lookup = load_predictions()
    print(f"Loaded {len(tracks)} tracks and {len(candidates)} candidate boundaries", flush=True)

    simple_scores = evaluate_simple_configs(tracks, candidates, pred_lookup, args.out_dir)
    simple_agg = aggregate_candidates(simple_scores, ["config_id", "source_set", "curve_mode", "family"], args.out_dir / "simple_rule_candidate_aggregates.csv")
    simple_sel, simple_chosen = select_fold(simple_scores, tracks, ["config_id"], "SIMP01_simple_rule_foldcv")
    simple_sel.to_csv(args.out_dir / "SIMP01_simple_rule_foldcv_metrics.csv", index=False)
    simple_chosen.to_csv(args.out_dir / "SIMP01_simple_rule_foldcv_chosen_configs.csv", index=False)

    model_scores, _probs = evaluate_model_variants(tracks, candidates, pred_lookup, args.out_dir)
    model_agg = aggregate_candidates(model_scores, ["config_id", "model_id", "family", "source_set", "feature_mode"], args.out_dir / "model_ablation_candidate_aggregates.csv")
    model_sel, model_chosen = select_fold(model_scores, tracks, ["config_id"], "SIMP02_compact_model_foldcv")
    model_sel.to_csv(args.out_dir / "SIMP02_compact_model_foldcv_metrics.csv", index=False)
    model_chosen.to_csv(args.out_dir / "SIMP02_compact_model_foldcv_chosen_configs.csv", index=False)

    comparison = comparison_rows(simple_sel, model_sel, args.out_dir)
    compare_vs(pd.concat([simple_sel, model_sel], ignore_index=True), DIS_DIR / "AD01_PR01_sgdp_preregistered_metrics.csv", args.out_dir / "comparison_vs_ad01.csv")
    compare_vs(pd.concat([simple_sel, model_sel], ignore_index=True), RES_DIR / "ADRES01_ad01_anchored_candidate_rescue_foldcv_metrics.csv", args.out_dir / "comparison_vs_adres01.csv")
    plot_comparison(comparison, args.out_dir / "aggregate_metrics.png")
    write_summary(comparison, simple_agg, model_agg, simple_chosen, model_chosen, args.out_dir)
    print(comparison[["exp_id", "phase", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
