#!/usr/bin/env python3
"""Evaluate alternative SSM similarity functions under the E33 fusion recipe."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

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


FEATURE_DIR = Path("feature_outputs/rwc_p_100_orthogonal_v1")
ANNOTATION_DIR = Path("data/rwc-annotations-archive")
E33_WEIGHTS_PATH = Path("structure_outputs/rwc_p_100_best_v1/learned_E33_reliability_gated_loo_weights.csv")
E33_BASELINE_AGG_PATH = Path("structure_outputs/rwc_p_100_best_v1/aggregate_metrics.csv")
OUT_DIR = Path("structure_outputs/rwc_p_100_similarity_metrics_v1")

STREAMS = ("stm", "mfcc", "cens", "arrangement", "density")
BEAT_KEYS = {
    "stm": "beat_stm",
    "mfcc": "beat_mfcc",
    "cens": "beat_cens",
    "arrangement": "beat_arrangement",
    "density": "beat_density",
}
SSM_KEYS = {
    "stm": "ssm_stm",
    "mfcc": "ssm_mfcc",
    "cens": "ssm_cens",
    "arrangement": "ssm_arrangement",
    "density": "ssm_density",
}

VARIANTS = {
    "S01_self_tuning_rbf": "RBF self-tuning por stream",
    "S02_stream_specific": "Metrica elegida por tipo de feature",
    "S03_temporal_context": "Cosine con contexto temporal +/-2 beats",
    "S04_supervised_scalar_rbf": "RBF con escala supervisada por stream",
    "S05_supervised_diagonal_metric": "RBF con pesos diagonales supervisados",
}


@dataclass
class TrackStats:
    pos_sum: np.ndarray
    neg_sum: np.ndarray
    pos_count: int
    neg_count: int


def safe_features(features: np.ndarray) -> np.ndarray:
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def zscore_dims(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = safe_features(features)
    return (x - np.mean(x, axis=1, keepdims=True)) / np.maximum(np.std(x, axis=1, keepdims=True), eps)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def cosine_ssm(features: np.ndarray) -> np.ndarray:
    x = zscore_dims(features).T
    x = l2_normalize_rows(x)
    return normalize_ssm(x @ x.T)


def correlation_ssm(features: np.ndarray) -> np.ndarray:
    x = zscore_dims(features).T
    x = x - np.mean(x, axis=1, keepdims=True)
    x = l2_normalize_rows(x)
    return normalize_ssm(x @ x.T)


def pairwise_sqeuclidean(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is not None:
        x = x * np.sqrt(np.maximum(weights, 1e-8))[None, :]
    norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (x @ x.T)
    return np.maximum(d2, 0.0).astype(np.float32)


def rbf_from_distance(d2: np.ndarray, scale: float) -> np.ndarray:
    scale = max(float(scale), 1e-6)
    return normalize_ssm(np.exp(-d2 / scale).astype(np.float32))


def self_tuning_rbf_ssm(features: np.ndarray, k: int = 7) -> np.ndarray:
    x = zscore_dims(features).T
    d2 = pairwise_sqeuclidean(x)
    n = d2.shape[0]
    if n <= 2:
        return normalize_ssm(np.ones((n, n), dtype=np.float32))
    kk = int(np.clip(k, 1, n - 1))
    sorted_d = np.partition(d2 + np.eye(n, dtype=np.float32) * 1e12, kk, axis=1)
    sigma = np.sqrt(np.maximum(sorted_d[:, kk], 1e-6))
    denom = np.maximum(sigma[:, None] * sigma[None, :], 1e-6)
    return normalize_ssm(np.exp(-d2 / denom).astype(np.float32))


def l1_rbf_ssm(features: np.ndarray) -> np.ndarray:
    x = zscore_dims(features).T
    d = np.mean(np.abs(x[:, None, :] - x[None, :, :]), axis=-1)
    scale = float(np.percentile(d[d > 0], 35)) if np.any(d > 0) else 1.0
    return normalize_ssm(np.exp(-d / max(scale, 1e-6)).astype(np.float32))


def transposition_invariant_cens_ssm(features: np.ndarray) -> np.ndarray:
    base = zscore_dims(features)
    best = None
    for shift in range(base.shape[0]):
        rolled = np.roll(base, shift=shift, axis=0)
        a = l2_normalize_rows(base.T)
        b = l2_normalize_rows(rolled.T)
        sim = a @ b.T
        best = sim if best is None else np.maximum(best, sim)
    return normalize_ssm(best.astype(np.float32))


def temporal_context_features(features: np.ndarray, radius: int = 2) -> np.ndarray:
    x = zscore_dims(features)
    pieces = []
    for offset in range(-radius, radius + 1):
        if offset < 0:
            pad = np.repeat(x[:, :1], -offset, axis=1)
            shifted = np.concatenate([pad, x[:, :offset]], axis=1)
        elif offset > 0:
            pad = np.repeat(x[:, -1:], offset, axis=1)
            shifted = np.concatenate([x[:, offset:], pad], axis=1)
        else:
            shifted = x
        pieces.append(shifted)
    return np.vstack(pieces).astype(np.float32)


def segment_ids_for_track(track_id: str, beat_boundaries: np.ndarray) -> np.ndarray:
    ref_path = reference_path_for_track(ANNOTATION_DIR, track_id)
    if not ref_path:
        return np.full(max(0, len(beat_boundaries) - 1), -1, dtype=np.int32)
    reference = parse_chorus_file(ref_path)
    centers = 0.5 * (beat_boundaries[:-1] + beat_boundaries[1:])
    seg_ids = np.full(len(centers), -1, dtype=np.int32)
    for idx, (start, end, _label) in enumerate(reference.segments):
        if idx == len(reference.segments) - 1:
            mask = (centers >= start) & (centers <= end)
        else:
            mask = (centers >= start) & (centers < end)
        seg_ids[mask] = idx
    return seg_ids


def sampled_pair_stats(
    features: np.ndarray,
    seg_ids: np.ndarray,
    rng: np.random.Generator,
    samples_per_class: int = 900,
) -> TrackStats:
    x = zscore_dims(features).T
    n, dim = x.shape
    valid = np.flatnonzero(seg_ids >= 0)
    if valid.size < 2:
        zeros = np.zeros(dim, dtype=np.float64)
        return TrackStats(zeros, zeros, 0, 0)
    pos_acc = np.zeros(dim, dtype=np.float64)
    neg_acc = np.zeros(dim, dtype=np.float64)
    pos_count = 0
    neg_count = 0
    by_segment: dict[int, np.ndarray] = {
        int(seg): valid[seg_ids[valid] == seg] for seg in np.unique(seg_ids[valid])
    }
    segments = [seg for seg, idxs in by_segment.items() if idxs.size >= 2]
    for _ in range(samples_per_class):
        if segments:
            seg = int(rng.choice(segments))
            i, j = rng.choice(by_segment[seg], size=2, replace=False)
            diff = x[i] - x[j]
            pos_acc += diff * diff
            pos_count += 1
        i, j = rng.choice(valid, size=2, replace=False)
        tries = 0
        while seg_ids[i] == seg_ids[j] and tries < 20:
            i, j = rng.choice(valid, size=2, replace=False)
            tries += 1
        if seg_ids[i] != seg_ids[j]:
            diff = x[i] - x[j]
            neg_acc += diff * diff
            neg_count += 1
    return TrackStats(pos_acc, neg_acc, pos_count, neg_count)


def compute_supervised_stats(feature_paths: list[Path]) -> dict[str, dict[str, TrackStats]]:
    stats: dict[str, dict[str, TrackStats]] = {stream: {} for stream in STREAMS}
    rng = np.random.default_rng(20260506)
    for feature_path in feature_paths:
        track_id = feature_path.name.replace("_features.npz", "")
        with np.load(feature_path, allow_pickle=True) as data:
            seg_ids = segment_ids_for_track(track_id, np.asarray(data["beat_boundaries"], dtype=np.float32))
            for stream in STREAMS:
                stats[stream][track_id] = sampled_pair_stats(np.asarray(data[BEAT_KEYS[stream]]), seg_ids, rng)
    return stats


def combine_stats(stats_by_track: dict[str, TrackStats], held_out: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    pos_sum = None
    neg_sum = None
    pos_count = 0
    neg_count = 0
    for track_id, stats in stats_by_track.items():
        if track_id == held_out:
            continue
        pos_sum = stats.pos_sum.copy() if pos_sum is None else pos_sum + stats.pos_sum
        neg_sum = stats.neg_sum.copy() if neg_sum is None else neg_sum + stats.neg_sum
        pos_count += stats.pos_count
        neg_count += stats.neg_count
    if pos_sum is None:
        pos_sum = np.zeros(1, dtype=np.float64)
        neg_sum = np.zeros(1, dtype=np.float64)
    return pos_sum, neg_sum, pos_count, neg_count


def supervised_scalar_rbf_ssm(features: np.ndarray, stream_stats: dict[str, TrackStats], held_out: str) -> np.ndarray:
    x = zscore_dims(features).T
    pos_sum, neg_sum, pos_count, neg_count = combine_stats(stream_stats, held_out)
    pos_mean = float(np.sum(pos_sum) / max(1, pos_count * x.shape[1]))
    neg_mean = float(np.sum(neg_sum) / max(1, neg_count * x.shape[1]))
    scale = max(pos_mean + 0.25 * max(0.0, neg_mean - pos_mean), 1e-5)
    return rbf_from_distance(pairwise_sqeuclidean(x) / max(1, x.shape[1]), scale)


def supervised_diagonal_ssm(features: np.ndarray, stream_stats: dict[str, TrackStats], held_out: str) -> np.ndarray:
    x = zscore_dims(features).T
    pos_sum, neg_sum, pos_count, neg_count = combine_stats(stream_stats, held_out)
    pos = pos_sum / max(1, pos_count)
    neg = neg_sum / max(1, neg_count)
    if pos.shape[0] != x.shape[1]:
        return supervised_scalar_rbf_ssm(features, stream_stats, held_out)
    weights = np.maximum(neg - pos, 0.0) / np.maximum(pos, 0.05)
    if not np.any(weights > 0):
        weights = np.ones_like(pos)
    weights = weights / max(float(np.mean(weights)), 1e-8)
    d2 = pairwise_sqeuclidean(x, weights=weights) / max(1, x.shape[1])
    pos_scale = float(np.mean(pos * weights))
    neg_scale = float(np.mean(neg * weights))
    scale = max(pos_scale + 0.25 * max(0.0, neg_scale - pos_scale), 1e-5)
    return rbf_from_distance(d2, scale)


def variant_stream_ssm(
    variant: str,
    stream: str,
    data: np.lib.npyio.NpzFile,
    track_id: str,
    supervised_stats: dict[str, dict[str, TrackStats]] | None,
) -> np.ndarray:
    features = np.asarray(data[BEAT_KEYS[stream]])
    if variant == "S01_self_tuning_rbf":
        return self_tuning_rbf_ssm(features)
    if variant == "S02_stream_specific":
        if stream == "cens":
            return transposition_invariant_cens_ssm(features)
        if stream == "density":
            return l1_rbf_ssm(features)
        if stream in {"stm", "arrangement"}:
            return self_tuning_rbf_ssm(features, k=9)
        return correlation_ssm(features)
    if variant == "S03_temporal_context":
        return cosine_ssm(temporal_context_features(features, radius=2))
    if variant == "S04_supervised_scalar_rbf":
        assert supervised_stats is not None
        return supervised_scalar_rbf_ssm(features, supervised_stats[stream], track_id)
    if variant == "S05_supervised_diagonal_metric":
        assert supervised_stats is not None
        return supervised_diagonal_ssm(features, supervised_stats[stream], track_id)
    raise ValueError(f"Unknown variant: {variant}")


def novelty_salience(matrix: np.ndarray) -> float:
    novelty = novelty_curve(matrix)
    if novelty.size == 0:
        return 0.0
    spread = float(np.percentile(novelty, 95) - np.percentile(novelty, 50))
    variability = float(np.std(novelty))
    peakiness = float(np.max(novelty))
    return max(1e-4, spread + 0.5 * variability + 0.15 * peakiness)


def reliability_gated_from_matrices(base_weights: dict[str, float], matrices: dict[str, np.ndarray]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for stream, base_weight in base_weights.items():
        if stream not in matrices:
            continue
        matrix = normalize_ssm(matrices[stream])
        diag = ssm_reliability_features(matrix)
        local_bonus = 1.0 + 1.5 * max(0.0, diag["local_gap"])
        contrast_bonus = 0.65 + diag["contrast"]
        recurrence_bonus = 1.0 + 0.5 * min(0.25, diag["recurrence"])
        scored[stream] = float(base_weight) * novelty_salience(matrix) * local_bonus * contrast_bonus * recurrence_bonus
    total = sum(scored.values())
    if total <= 0:
        total_base = sum(max(float(v), 0.0) for v in base_weights.values())
        return {k: float(v) / total_base for k, v in base_weights.items() if v > 0} if total_base > 0 else {}
    return {stream: value / total for stream, value in scored.items() if value > 0}


def fuse_matrices(matrices: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    used = [(stream, weight) for stream, weight in weights.items() if stream in matrices]
    if not used:
        raise ValueError("No matrices to fuse")
    total = sum(float(weight) for _stream, weight in used)
    fused = np.zeros_like(next(iter(matrices.values())), dtype=np.float32)
    for stream, weight in used:
        fused += float(weight) / max(total, 1e-8) * normalize_ssm(matrices[stream])
    return normalize_ssm(fused)


def load_e33_weights() -> dict[str, dict[str, float]]:
    df = pd.read_csv(E33_WEIGHTS_PATH)
    out = {}
    for row in df.itertuples(index=False):
        weights = {}
        for stream in STREAMS:
            value = float(getattr(row, f"w_{stream}"))
            if value > 0:
                weights[stream] = value
        out[row.held_out_track_id] = weights
    return out


def evaluate_variant(
    variant: str,
    feature_paths: list[Path],
    e33_weights: dict[str, dict[str, float]],
    supervised_stats: dict[str, dict[str, TrackStats]] | None,
) -> list[dict]:
    rows = []
    for idx, feature_path in enumerate(feature_paths, 1):
        track_id = feature_path.name.replace("_features.npz", "")
        ref_path = reference_path_for_track(ANNOTATION_DIR, track_id)
        if not ref_path:
            continue
        reference = parse_chorus_file(ref_path)
        with np.load(feature_path, allow_pickle=True) as data:
            matrices = {
                stream: variant_stream_ssm(variant, stream, data, track_id, supervised_stats)
                for stream in STREAMS
            }
            weights = reliability_gated_from_matrices(e33_weights[track_id], matrices)
            fused = fuse_matrices(matrices, weights)
            novelty = novelty_curve(fused)
            beat_boundaries = np.asarray(data["beat_boundaries"], dtype=np.float32)
            indices = pick_boundary_indices(novelty, beat_boundaries)
            predicted = boundaries_from_indices(indices, beat_boundaries)
        m05 = boundary_metrics(predicted, reference.internal_boundaries, 0.5)
        m30 = boundary_metrics(predicted, reference.internal_boundaries, 3.0)
        rows.append(
            {
                "track_id": track_id,
                "exp_id": variant,
                "description": VARIANTS[variant],
                "pred_count": int(len(predicted)),
                "ref_count": int(len(reference.internal_boundaries)),
                "precision_0p5": m05.precision,
                "recall_0p5": m05.recall,
                "f_0p5": m05.f_measure,
                "precision_3p0": m30.precision,
                "recall_3p0": m30.recall,
                "f_3p0": m30.f_measure,
                "matches_3p0": m30.matches,
                "weights": json.dumps(weights, sort_keys=True),
            }
        )
        if idx % 10 == 0:
            print(f"  {variant}: {idx}/{len(feature_paths)}", flush=True)
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return (
        df.groupby(["exp_id", "description"], as_index=False)
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
    labels = comparison["label"].to_list()
    y = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    ax.barh(y, comparison["f_3p0"], color="#3B6EA8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.48, 0.68)
    ax.set_xlabel("F@3s")
    ax.set_title("E33 con funciones de similitud alternativas")
    ax.grid(axis="x", alpha=0.25)
    for i, row in comparison.iterrows():
        ax.text(row["f_3p0"] + 0.003, i, f"{row['f_3p0']:.3f}", va="center", fontsize=8.5)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary(comparison: pd.DataFrame, rows: pd.DataFrame) -> None:
    baseline = comparison[comparison["exp_id"] == "E33_reliability_gated_loo"].iloc[0]
    best_alt = comparison[comparison["exp_id"] != "E33_reliability_gated_loo"].iloc[0]
    pivot = rows.pivot(index="track_id", columns="exp_id", values="f_3p0")
    lines = [
        "# Pruebas de funciones de similitud con E33",
        "",
        "## Resultado agregado",
        "",
        "| Metodo | F@3s | F@0.5s | Precision@3s | Recall@3s | Pred. prom. |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.f_3p0:.3f} | {row.f_0p5:.3f} | "
            f"{row.precision_3p0:.3f} | {row.recall_3p0:.3f} | {row.pred_count:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Lectura",
            "",
            f"- Referencia: E33 actual con cosine SSM queda en F@3s={baseline.f_3p0:.3f}.",
            f"- Mejor alternativa en esta primera pasada: {best_alt.label}, F@3s={best_alt.f_3p0:.3f}, delta={best_alt.f_3p0 - baseline.f_3p0:+.3f}.",
            "- Estas pruebas mantienen los pesos LOO de E33 aprendidos con cosine. Si una metrica alternativa se acerca o supera la referencia, el siguiente paso justo es reaprender los pesos de fusion para esa metrica.",
            "",
            "## Deltas por track contra E33",
            "",
        ]
    )
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
            "## Conclusion operacional",
            "",
            "Si ninguna variante supera a E33 con pesos fijos, no conviene reemplazar directamente la similitud en todo el pipeline. La via mas razonable es tomar las mejores variantes por feature como canales adicionales y reaprender la fusion, o entrenar una metrica supervisada pero con una validacion mas estricta y regularizada por feature.",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_paths = sorted(FEATURE_DIR.glob("*_features.npz"))
    e33_weights = load_e33_weights()
    print("Computing supervised pair statistics...", flush=True)
    supervised_stats = compute_supervised_stats(feature_paths)

    all_rows = []
    for variant in VARIANTS:
        print(f"[{variant}] {VARIANTS[variant]}", flush=True)
        all_rows.extend(evaluate_variant(variant, feature_paths, e33_weights, supervised_stats))

    rows_df = pd.DataFrame(all_rows)
    rows_df.to_csv(OUT_DIR / "experiment_metrics.csv", index=False)
    agg = aggregate(all_rows)
    baseline = pd.read_csv(E33_BASELINE_AGG_PATH)
    e33 = baseline[baseline["exp_id"] == "E33_reliability_gated_loo"].copy()
    e33["description"] = "Cosine actual"
    comparison = pd.concat([e33, agg], ignore_index=True, sort=False)
    labels = {"E33_reliability_gated_loo": "E33 cosine actual", **{k: k.replace("_", " ") for k in VARIANTS}}
    comparison["label"] = comparison["exp_id"].map(labels)
    comparison = comparison.sort_values("f_3p0", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "aggregate_metrics.csv", index=False)

    plot_aggregate(comparison, OUT_DIR / "aggregate_metrics.png")
    rows_with_base = pd.concat(
        [
            pd.read_csv("structure_outputs/rwc_p_100_best_v1/experiment_metrics.csv").query(
                "exp_id == 'E33_reliability_gated_loo'"
            )[["track_id", "exp_id", "f_3p0"]],
            rows_df[["track_id", "exp_id", "f_3p0"]],
        ],
        ignore_index=True,
    )
    write_summary(comparison, rows_with_base)
    print(comparison[["exp_id", "f_3p0", "f_0p5", "precision_3p0", "recall_3p0", "pred_count", "tracks"]].to_string(index=False))
    print(f"Wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
