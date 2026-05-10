#!/usr/bin/env python3
"""Full LOO training for alternative SSM similarity metrics under E33."""

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

from run_structure_baseline import (
    BoundaryMetrics,
    boundary_metrics,
    boundaries_from_indices,
    normalize_ssm,
    novelty_curve,
    parse_chorus_file,
    pick_boundary_indices,
    reference_path_for_track,
    ssm_reliability_features,
)
from run_similarity_metric_experiments import (
    BEAT_KEYS,
    SSM_KEYS,
    STREAMS,
    VARIANTS,
    TrackStats,
    compute_supervised_stats,
    cosine_ssm,
    novelty_salience,
    variant_stream_ssm,
)


FEATURE_DIR = Path("feature_outputs/rwc_p_100_orthogonal_v1")
ANNOTATION_DIR = Path("data/rwc-annotations-archive")
OUT_DIR = Path("structure_outputs/rwc_p_100_similarity_full_training_v1")
BASELINE_AGG = Path("structure_outputs/rwc_p_100_best_v1/aggregate_metrics.csv")


@dataclass
class TrackPack:
    track_id: str
    beat_boundaries: np.ndarray
    reference_boundaries: np.ndarray
    matrices: dict[str, np.ndarray]
    reliability: dict[str, float]


def simplex_weight_grid(channels: list[str], step: float) -> tuple[list[dict[str, float]], np.ndarray]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0.")
    rows: list[dict[str, float]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            values = prefix + [remaining]
            rows.append({channel: value / units for channel, value in zip(channels, values) if value > 0})
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], units, len(channels))
    arr = np.zeros((len(rows), len(channels)), dtype=np.float32)
    for idx, weights in enumerate(rows):
        for channel, weight in weights.items():
            arr[idx, channels.index(channel)] = float(weight)
    return rows, arr


def reliability_factor(matrix: np.ndarray) -> float:
    matrix = normalize_ssm(matrix)
    diag = ssm_reliability_features(matrix)
    local_bonus = 1.0 + 1.5 * max(0.0, diag["local_gap"])
    contrast_bonus = 0.65 + diag["contrast"]
    recurrence_bonus = 1.0 + 0.5 * min(0.25, diag["recurrence"])
    return max(1e-6, novelty_salience(matrix) * local_bonus * contrast_bonus * recurrence_bonus)


def cosine_stream_matrix(data: np.lib.npyio.NpzFile, stream: str) -> np.ndarray:
    if SSM_KEYS[stream] in data:
        return normalize_ssm(np.asarray(data[SSM_KEYS[stream]]))
    return cosine_ssm(np.asarray(data[BEAT_KEYS[stream]]))


def load_track_pack(
    feature_path: Path,
    variant: str,
    supervised_stats: dict[str, dict[str, TrackStats]] | None,
) -> TrackPack | None:
    track_id = feature_path.name.replace("_features.npz", "")
    ref_path = reference_path_for_track(ANNOTATION_DIR, track_id)
    if not ref_path:
        return None
    reference = parse_chorus_file(ref_path)
    with np.load(feature_path, allow_pickle=True) as data:
        beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
        matrices = {
            stream: variant_stream_ssm(variant, stream, data, track_id, supervised_stats)
            for stream in STREAMS
        }
    return TrackPack(
        track_id=track_id,
        beat_boundaries=beat_boundaries,
        reference_boundaries=reference.internal_boundaries,
        matrices=matrices,
        reliability={stream: reliability_factor(matrix) for stream, matrix in matrices.items()},
    )


def load_combo_track_pack(
    feature_path: Path,
    best_variant: str,
    supervised_stats: dict[str, dict[str, TrackStats]] | None,
) -> TrackPack | None:
    track_id = feature_path.name.replace("_features.npz", "")
    ref_path = reference_path_for_track(ANNOTATION_DIR, track_id)
    if not ref_path:
        return None
    reference = parse_chorus_file(ref_path)
    matrices: dict[str, np.ndarray] = {}
    with np.load(feature_path, allow_pickle=True) as data:
        beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
        for stream in STREAMS:
            matrices[f"{stream}:cosine"] = cosine_stream_matrix(data, stream)
            matrices[f"{stream}:{best_variant}"] = variant_stream_ssm(
                best_variant,
                stream,
                data,
                track_id,
                supervised_stats,
            )
    return TrackPack(
        track_id=track_id,
        beat_boundaries=beat_boundaries,
        reference_boundaries=reference.internal_boundaries,
        matrices=matrices,
        reliability={channel: reliability_factor(matrix) for channel, matrix in matrices.items()},
    )


def evaluate_candidate(pack: TrackPack, channels: list[str], candidate: np.ndarray) -> dict:
    factors = np.asarray([pack.reliability[channel] for channel in channels], dtype=np.float32)
    weights = candidate * factors
    if float(np.sum(weights)) <= 0:
        weights = candidate
    weights = weights / max(float(np.sum(weights)), 1e-8)
    fused = np.zeros_like(pack.matrices[channels[0]], dtype=np.float32)
    used_weights = {}
    for channel, weight in zip(channels, weights):
        if weight <= 0:
            continue
        fused += float(weight) * normalize_ssm(pack.matrices[channel])
        used_weights[channel] = float(weight)
    fused = normalize_ssm(fused)
    novelty = novelty_curve(fused)
    indices = pick_boundary_indices(novelty, pack.beat_boundaries)
    predicted = boundaries_from_indices(indices, pack.beat_boundaries)
    m05 = boundary_metrics(predicted, pack.reference_boundaries, 0.5)
    m30 = boundary_metrics(predicted, pack.reference_boundaries, 3.0)
    return {
        "weights": used_weights,
        "pred_count": int(len(predicted)),
        "ref_count": int(len(pack.reference_boundaries)),
        "precision_0p5": m05.precision,
        "recall_0p5": m05.recall,
        "f_0p5": m05.f_measure,
        "precision_3p0": m30.precision,
        "recall_3p0": m30.recall,
        "f_3p0": m30.f_measure,
        "matches_3p0": m30.matches,
    }


def train_and_evaluate(
    exp_id: str,
    description: str,
    packs: list[TrackPack],
    channels: list[str],
    step: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_dicts, candidate_array = simplex_weight_grid(channels, step)
    scored_rows = []
    for track_idx, pack in enumerate(packs, 1):
        for cand_idx, candidate in enumerate(candidate_array):
            result = evaluate_candidate(pack, channels, candidate)
            scored_rows.append(
                {
                    "track_id": pack.track_id,
                    "candidate_idx": cand_idx,
                    "base_weights": json.dumps(candidate_dicts[cand_idx], sort_keys=True),
                    **{key: result[key] for key in ("f_3p0", "f_0p5")},
                }
            )
        if track_idx % 10 == 0:
            print(f"  {exp_id}: scored {track_idx}/{len(packs)} tracks", flush=True)
    scored = pd.DataFrame(scored_rows)

    selected_rows = []
    for pack in packs:
        train = scored[scored["track_id"] != pack.track_id]
        grid = (
            train.groupby(["candidate_idx", "base_weights"], as_index=False)
            .agg(f_3p0=("f_3p0", "mean"), f_0p5=("f_0p5", "mean"))
            .sort_values(["f_3p0", "f_0p5"], ascending=False)
        )
        best = grid.iloc[0]
        result = evaluate_candidate(pack, channels, candidate_array[int(best["candidate_idx"])])
        selected_rows.append(
            {
                "track_id": pack.track_id,
                "exp_id": exp_id,
                "description": description,
                "candidate_idx": int(best["candidate_idx"]),
                "base_weights": best["base_weights"],
                "weights": json.dumps(result["weights"], sort_keys=True),
                **{key: result[key] for key in result if key != "weights"},
            }
        )
    return pd.DataFrame(selected_rows), scored


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["exp_id", "description"], as_index=False)
        .agg(
            f_0p5=("f_0p5", "mean"),
            f_3p0=("f_3p0", "mean"),
            precision_3p0=("precision_3p0", "mean"),
            recall_3p0=("recall_3p0", "mean"),
            pred_count=("pred_count", "mean"),
            ref_count=("ref_count", "mean"),
            tracks=("track_id", "nunique"),
        )
        .sort_values("f_3p0", ascending=False)
    )


def plot_aggregate(comparison: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.6), constrained_layout=True)
    y = np.arange(len(comparison))
    ax.barh(y, comparison["f_3p0"], color="#3B6EA8")
    ax.set_yticks(y)
    ax.set_yticklabels(comparison["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.50, 0.69)
    ax.set_xlabel("F@3s")
    ax.set_title("Entrenamiento LOO completo por funcion de similitud")
    ax.grid(axis="x", alpha=0.25)
    for i, row in comparison.iterrows():
        ax.text(row["f_3p0"] + 0.003, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8.5)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    lines = ["| Metodo | F@3s | F@0.5s | Precision@3s | Recall@3s | Pred. prom. |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.f_3p0:.3f} | {row.f_0p5:.3f} | "
            f"{row.precision_3p0:.3f} | {row.recall_3p0:.3f} | {row.pred_count:.2f} |"
        )
    return "\n".join(lines)


def write_summary(comparison: pd.DataFrame, all_rows: pd.DataFrame, best_variant: str) -> None:
    baseline = comparison[comparison["exp_id"] == "E33_reliability_gated_loo"].iloc[0]
    best = comparison[comparison["exp_id"] != "E33_reliability_gated_loo"].iloc[0]
    pivot = all_rows.pivot(index="track_id", columns="exp_id", values="f_3p0")
    lines = [
        "# Entrenamiento completo de metricas de similitud",
        "",
        "## Resultado agregado",
        "",
        markdown_table(comparison),
        "",
        "## Lectura",
        "",
        f"- Baseline E33 cosine: F@3s={baseline.f_3p0:.3f}.",
        f"- Mejor metrica nueva con entrenamiento completo: `{best.exp_id}`, F@3s={best.f_3p0:.3f}, delta={best.f_3p0 - baseline.f_3p0:+.3f}.",
        f"- La combinacion entrenada `cosine + {best_variant}` queda incluida como `C01_cosine_plus_{best_variant}`.",
        "",
        "## Deltas contra E33",
        "",
    ]
    for exp_id in comparison["exp_id"]:
        if exp_id == "E33_reliability_gated_loo" or exp_id not in pivot:
            continue
        delta = pivot[exp_id] - pivot["E33_reliability_gated_loo"]
        lines.append(
            f"- `{exp_id}`: media={delta.mean():+.3f}, gana={(delta > 0).sum()}/100, "
            f"pierde={(delta < 0).sum()}/100, empata={(delta == 0).sum()}/100."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Esta corrida reentrena los pesos de fusion para cada metrica. Por lo tanto, mide mejor si la funcion de similitud por si sola desplaza a cosine. Si la mejor alternativa sigue cerca pero no supera a E33, el camino mas razonable es usarla selectivamente por feature o como canal adicional con una grilla mas fina/regularizada, no reemplazar toda la geometria.",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--combo-step", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_paths = sorted(FEATURE_DIR.glob("*_features.npz"))[: args.limit]
    print("Computing supervised pair statistics...", flush=True)
    supervised_stats = compute_supervised_stats(feature_paths)

    all_selected = []
    all_scored = []
    variant_aggs = []
    for variant, description in VARIANTS.items():
        print(f"[{variant}] full LOO training", flush=True)
        packs = [
            pack
            for path in feature_paths
            if (pack := load_track_pack(path, variant, supervised_stats)) is not None
        ]
        selected, scored = train_and_evaluate(variant, description, packs, list(STREAMS), args.step)
        selected.to_csv(OUT_DIR / f"{variant}_selected_metrics.csv", index=False)
        scored.to_csv(OUT_DIR / f"{variant}_candidate_scores.csv", index=False)
        all_selected.append(selected)
        all_scored.append(scored.assign(exp_id=variant))
        variant_aggs.append(aggregate(selected))
        print(aggregate(selected).to_string(index=False), flush=True)

    variant_agg = pd.concat(variant_aggs, ignore_index=True).sort_values("f_3p0", ascending=False)
    best_variant = str(variant_agg.iloc[0]["exp_id"])
    combo_id = f"C01_cosine_plus_{best_variant}"
    print(f"[{combo_id}] full LOO training", flush=True)
    combo_packs = [
        pack
        for path in feature_paths
        if (pack := load_combo_track_pack(path, best_variant, supervised_stats)) is not None
    ]
    combo_channels = list(combo_packs[0].matrices)
    combo_selected, combo_scored = train_and_evaluate(
        combo_id,
        f"Cosine + {best_variant}",
        combo_packs,
        combo_channels,
        args.combo_step,
    )
    combo_selected.to_csv(OUT_DIR / f"{combo_id}_selected_metrics.csv", index=False)
    combo_scored.to_csv(OUT_DIR / f"{combo_id}_candidate_scores.csv", index=False)
    all_selected.append(combo_selected)
    all_scored.append(combo_scored.assign(exp_id=combo_id))

    selected_df = pd.concat(all_selected, ignore_index=True)
    selected_df.to_csv(OUT_DIR / "experiment_metrics.csv", index=False)
    pd.concat(all_scored, ignore_index=True).to_csv(OUT_DIR / "candidate_scores.csv", index=False)

    baseline = pd.read_csv(BASELINE_AGG)
    e33 = baseline[baseline["exp_id"] == "E33_reliability_gated_loo"].copy()
    e33["description"] = "Cosine actual con entrenamiento E33"
    comparison = pd.concat([e33, aggregate(selected_df)], ignore_index=True, sort=False)
    labels = {"E33_reliability_gated_loo": "E33 cosine actual"}
    labels.update({variant: variant.replace("_", " ") for variant in VARIANTS})
    labels[combo_id] = f"cosine + {best_variant}"
    comparison["label"] = comparison["exp_id"].map(labels).fillna(comparison["exp_id"])
    comparison = comparison.sort_values("f_3p0", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "aggregate_metrics.csv", index=False)
    plot_aggregate(comparison, OUT_DIR / "aggregate_metrics.png")

    baseline_rows = pd.read_csv("structure_outputs/rwc_p_100_best_v1/experiment_metrics.csv").query(
        "exp_id == 'E33_reliability_gated_loo'"
    )
    all_for_summary = pd.concat(
        [baseline_rows[["track_id", "exp_id", "f_3p0"]], selected_df[["track_id", "exp_id", "f_3p0"]]],
        ignore_index=True,
    )
    write_summary(comparison, all_for_summary, best_variant)
    print(comparison[["exp_id", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count", "tracks"]].to_string(index=False))
    print(f"Wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
