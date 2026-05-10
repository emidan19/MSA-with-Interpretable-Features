#!/usr/bin/env python3
"""Minimal support-only models for boundary candidate rescue."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_CACHE_ROOT = os.path.join(os.getcwd(), ".cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import run_ad01_candidate_rescue_iteration as adres
import run_literature_guided_iterations as lit
import run_stability_guided_dp_iteration as sgdp
import run_adres_simplification_iteration as simp


OUT_DIR = Path("structure_outputs/rwc_p_100_adres_minimal_models_v1")


@dataclass(frozen=True)
class MinimalVariant:
    model_id: str
    family: str
    source_set: str
    feature_mode: str


FEATURE_SETS = {
    "rule3": ["support_near", "support_ideas_near", "max_gap_ad01_s"],
    "rule4": ["support_near", "support_ideas_near", "max_gap_ad01_s", "ad01_dist_s"],
    "rule5": ["support_near", "support_ideas_near", "max_gap_ad01_s", "ad01_dist_s", "i03_dist_s"],
    "consensus4": ["support_near", "support_ideas_near", "time_norm", "pool_size"],
    "consensus5_gap": ["support_near", "support_ideas_near", "time_norm", "pool_size", "max_gap_ad01_s"],
    "top8": [
        "support_ideas_near",
        "support_near",
        "time_norm",
        "pool_size",
        "max_gap_ad01_s",
        "i05_dist_s",
        "right_gap_ad01_s",
        "ad01_dist_s",
    ],
    "gap_context": [
        "support_near",
        "support_ideas_near",
        "max_gap_ad01_s",
        "min_gap_ad01_s",
        "left_gap_ad01_s",
        "right_gap_ad01_s",
        "ad01_dist_s",
    ],
    "no_position_top": [
        "support_ideas_near",
        "support_near",
        "max_gap_ad01_s",
        "i05_dist_s",
        "right_gap_ad01_s",
        "ad01_dist_s",
        "i03_dist_s",
        "i02_dist_s",
    ],
    "source_distances": [
        "support_near",
        "support_ideas_near",
        "ad01_dist_s",
        "i02_dist_s",
        "i03_dist_s",
        "i05_dist_s",
        "max_gap_ad01_s",
    ],
}


def variants() -> list[MinimalVariant]:
    rows: list[MinimalVariant] = []
    for source_set in ("ad01_i02_i03", "ad01_i02_i03_i05", "full"):
        for feature_mode in FEATURE_SETS:
            rows.extend(
                [
                    MinimalVariant(f"logit_{feature_mode}_{source_set}", "logit", source_set, feature_mode),
                    MinimalVariant(f"tree2_{feature_mode}_{source_set}", "tree2", source_set, feature_mode),
                    MinimalVariant(f"tree3_{feature_mode}_{source_set}", "tree3", source_set, feature_mode),
                    MinimalVariant(f"rf_tiny_{feature_mode}_{source_set}", "rf_tiny", source_set, feature_mode),
                ]
            )
    return rows


def make_model(family: str):
    if family == "logit":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.45, class_weight="balanced", solver="liblinear", random_state=20260508),
        )
    if family == "tree2":
        return DecisionTreeClassifier(max_depth=2, min_samples_leaf=36, class_weight="balanced", random_state=20260508)
    if family == "tree3":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=26, class_weight="balanced", random_state=20260508)
    if family == "rf_tiny":
        return RandomForestClassifier(
            n_estimators=72,
            max_depth=4,
            min_samples_leaf=22,
            class_weight="balanced_subsample",
            max_features=0.8,
            random_state=20260508,
            n_jobs=-1,
        )
    raise ValueError(family)


def train_oof(candidates: pd.DataFrame, tracks: list[lit.TrackInfo], variant: MinimalVariant) -> pd.DataFrame:
    fold_by_track = {track.track_id: track.fold for track in tracks}
    data = simp.filter_source_set(candidates, variant.source_set).copy()
    data["fold"] = data["track_id"].map(fold_by_track)
    xcols = [col for col in FEATURE_SETS[variant.feature_mode] if col in data.columns]
    frames = []
    for fold in sorted(data["fold"].dropna().unique()):
        train = data[data["fold"] != fold]
        test = data[data["fold"] == fold].copy()
        model = make_model(variant.family)
        model.fit(train[xcols].to_numpy(dtype=float), train["label_3p0"].to_numpy(dtype=int))
        prob = model.predict_proba(test[xcols].to_numpy(dtype=float))[:, 1]
        local = test[["track_id", "idx", "time_s", "label_3p0", "label_0p5", "ad01_exact", "support_near", "support_exact"]].copy()
        local["model_id"] = variant.model_id
        local["prob_boundary"] = prob
        frames.append(local)
    return pd.concat(frames, ignore_index=True)


def selection_configs(variant: MinimalVariant) -> list[adres.SelectionConfig]:
    configs = []
    for ad_bonus in (0.10, 0.15, 0.20, 0.35):
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


def evaluate(tracks: list[lit.TrackInfo], candidates: pd.DataFrame, pred_lookup: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    variant_rows = []
    track_by_id = {track.track_id: track for track in tracks}
    for idx, variant in enumerate(variants(), 1):
        probs = train_oof(candidates, tracks, variant)
        variant_rows.append(variant.__dict__ | {"features": ",".join(FEATURE_SETS[variant.feature_mode])})
        grouped = {track_id: group for track_id, group in probs.groupby("track_id")}
        for cfg in selection_configs(variant):
            for track in tracks:
                local = grouped.get(track.track_id)
                if local is None or local.empty:
                    pred = pred_lookup[(track.track_id, "ad01")]
                else:
                    pred = adres.predict_for_config(local, pred_lookup[(track.track_id, "ad01")], track.beat_boundaries, cfg)
                rows.append(
                    lit.evaluate_indices(
                        track_by_id[track.track_id],
                        pred,
                        "SIMP03_minimal_model_foldcv",
                        extra={
                            "config_id": cfg.config_id,
                            "model_id": variant.model_id,
                            "family": variant.family,
                            "source_set": variant.source_set,
                            "feature_mode": variant.feature_mode,
                        },
                    )
                )
        print(f"minimal variants: {idx}/{len(variants())} {variant.model_id}", flush=True)
    pd.DataFrame(variant_rows).to_csv(OUT_DIR / "minimal_model_variants.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "minimal_model_scores.csv", index=False)
    return metrics


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["config_id", "model_id", "family", "source_set", "feature_mode"], as_index=False)
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
    agg.to_csv(OUT_DIR / "minimal_model_aggregates.csv", index=False)
    return agg


def grouped_diagnostics(agg: pd.DataFrame) -> None:
    by_mode = (
        agg.groupby(["source_set", "feature_mode", "family"], as_index=False)
        .agg(best_f_3p0=("f_3p0", "max"), best_f_0p5=("f_0p5", "max"), median_f_3p0=("f_3p0", "median"))
        .sort_values(["best_f_3p0", "best_f_0p5"], ascending=False)
    )
    by_mode.to_csv(OUT_DIR / "minimal_model_grouped_diagnostics.csv", index=False)
    lines = [
        "# Modelos mínimos",
        "",
        "## Top configuraciones",
        "",
        simp.markdown_table(agg.head(20)),
        "",
        "## Agrupado por fuente, feature set y familia",
        "",
        simp.markdown_table(by_mode.head(30)),
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot(agg: pd.DataFrame) -> None:
    prev = pd.read_csv(simp.OUT_DIR / "aggregate_metrics.csv")
    base = prev[
        prev["exp_id"].isin(
            [
                "AD01_PR01_sgdp_preregistered",
                "ADRES01_ad01_anchored_candidate_rescue_foldcv",
                "SIMP02_compact_model_foldcv",
            ]
        )
    ].drop_duplicates("exp_id")
    rows = base[["exp_id", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].copy()
    selected_path = OUT_DIR / "SIMP03_minimal_model_foldcv_aggregate.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path).iloc[0]
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "exp_id": "SIMP03_minimal_model_foldcv",
                            **selected[["f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_dict(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "exp_id": "SIMP03_best_minimal_fixed",
                        **agg.iloc[0][["f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count"]].to_dict(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rows.to_csv(OUT_DIR / "aggregate_metrics.csv", index=False)
    shown = rows.sort_values("f_3p0", ascending=True)
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    ax.barh(np.arange(len(shown)), shown["f_3p0"], color=["#64748B", "#2563EB", "#0F766E", "#14B8A6"][: len(shown)])
    ax.set_yticks(np.arange(len(shown)))
    ax.set_yticklabels(shown["exp_id"], fontsize=8.5)
    ax.set_xlabel("F@3s")
    ax.set_title("Modelo minimalista vs referencias")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(shown["f_3p0"]):
        ax.text(float(value) + 0.001, i, f"{value:.4f}", va="center", fontsize=8.5)
    fig.savefig(OUT_DIR / "aggregate_metrics.png", dpi=170)
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracks = simp.load_tracks(args.limit, args.folds)
    candidates = pd.read_csv(simp.RES_DIR / "boundary_candidate_features.csv")
    candidates = candidates[candidates["track_id"].isin([track.track_id for track in tracks])].copy()
    pred_lookup = simp.load_predictions()
    print(f"Loaded {len(tracks)} tracks and {len(candidates)} candidates", flush=True)
    scores = evaluate(tracks, candidates, pred_lookup)
    agg = aggregate(scores)
    selected, chosen = sgdp.fold_select(scores, tracks, ["config_id"], "SIMP03_minimal_model_foldcv")
    selected.to_csv(OUT_DIR / "SIMP03_minimal_model_foldcv_metrics.csv", index=False)
    chosen.to_csv(OUT_DIR / "SIMP03_minimal_model_foldcv_chosen_configs.csv", index=False)
    lit.aggregate(selected).to_csv(OUT_DIR / "SIMP03_minimal_model_foldcv_aggregate.csv", index=False)
    grouped_diagnostics(agg)
    plot(agg)
    print(agg.head(15).to_string(index=False), flush=True)
    print(lit.aggregate(selected).to_string(index=False), flush=True)
    print(f"Wrote {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
