"""Orchestration for feature extraction, beat-syncing, and previews."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from runtime_env import ensure_local_cache

ensure_local_cache()

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from beat_tracking import estimate_beats_and_downbeats
from data_loading import FeatureConfig, load_audio, load_sections, write_summary
from feature_types import (
    aggregate_by_intervals,
    build_section_ssm,
    build_ssm,
    count_beats_per_stm_window,
    extract_mfcc_chroma_cens,
    extract_orthogonal_proxy_features,
    extract_stm,
)


def save_previews(
    output_paths: list[Path],
    ssm_map: dict[str, np.ndarray],
    y: np.ndarray,
    sr: int,
    beat_boundaries: np.ndarray,
    downbeat_boundaries: np.ndarray | None,
    sections: list[tuple[float, float, str]],
    preview_features: tuple[str, ...],
) -> None:
    preview_items = [(name, ssm_map[name]) for name in preview_features if name in ssm_map]
    n_panels = len(preview_items) + 1
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), constrained_layout=True)
    axes = axes.ravel()

    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    boundaries_to_show = downbeat_boundaries if downbeat_boundaries is not None else beat_boundaries
    for boundary in boundaries_to_show:
        axes[0].axvline(boundary, color="k", alpha=0.08, linewidth=0.5)
    axes[0].set_title("Audio and downbeat grid" if downbeat_boundaries is not None else "Audio and beat grid")

    section_colors = plt.cm.tab10(np.linspace(0, 1, max(len(sections), 1)))
    if sections:
        for idx, (start, end, label) in enumerate(sections):
            color = section_colors[idx % len(section_colors)]
            axes[0].axvspan(start, end, alpha=0.1, color=color)
            mid_time = (start + end) / 2
            v_pos = ((0.95 * len(sections) - idx) / len(sections) - 0.5)
            axes[0].text(
                mid_time,
                2 * v_pos % 1 - 0.5,
                label,
                ha="center",
                va="top",
                fontsize=6,
                rotation=0,
                transform=axes[0].transData,
            )

    for ax, (name, matrix) in zip(axes[1:], preview_items):
        if matrix.size:
            ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=-1, vmax=1)
        ax.set_title(f"SSM: {name}")
        ax.set_xlabel("beat interval")
        ax.set_ylabel("section" if sections else "beat interval")
        if sections:
            ax.set_yticks([])
        ax.tick_params(axis="both", labelsize=6)

        for idx, (start, _end, label) in enumerate(sections):
            boundary_idx = max(0, np.searchsorted(beat_boundaries, start, side="left") - 1)
            if boundary_idx < matrix.shape[0]:
                color = section_colors[idx % len(section_colors)]
                ax.axvline(boundary_idx, color=color, linestyle="--", linewidth=1)
                ax.axhline(boundary_idx, color=color, linestyle="--", linewidth=1)
                ax.text(-0.5, boundary_idx, label, ha="right", va="center", fontsize=6, transform=ax.transData)

    for ax in axes[n_panels:]:
        ax.axis("off")

    for output_path in output_paths:
        save_kwargs = {"dpi": 160} if output_path.suffix.lower() == ".png" else {}
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def extract_all(audio_path: Path, out_dir: Path, cfg: FeatureConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y, duration = load_audio(audio_path, cfg.sr)
    sections = load_sections(cfg.sections_dir, audio_path.stem)
    requested = set(cfg.computing_features)

    need_frame_features = bool(
        requested
        & {
            "mfcc",
            "chroma",
            "cens",
            "arrangement",
            "vocal",
            "bass",
            "tonnetz",
            "density",
        }
    )
    need_stm = "stm" in requested

    frame_times = np.zeros(0, dtype=np.float32)
    mfcc = chroma = cens = None
    orthogonal_features: dict[str, np.ndarray] = {}
    orthogonal_diagnostics: dict[str, float] = {}

    if need_frame_features:
        mfcc, chroma, cens, frame_times = extract_mfcc_chroma_cens(y, cfg)
        need_orthogonal = bool(requested & {"arrangement", "vocal", "bass", "tonnetz", "density"})
        if need_orthogonal:
            orthogonal_features, orthogonal_diagnostics = extract_orthogonal_proxy_features(y, cfg, chroma, cens, frame_times)

    beat_boundaries, downbeat_boundaries = estimate_beats_and_downbeats(y, cfg)
    stm = np.zeros((0, 0), dtype=np.float32)
    stm_times = np.zeros(0, dtype=np.float32)
    onset_env = np.zeros(0, dtype=np.float32)
    stm_effective_window_s = cfg.stm_window_s
    if need_stm:
        stm, stm_times, onset_env, stm_effective_window_s = extract_stm(y, cfg, beat_boundaries)

    beat_level_features: dict[str, np.ndarray] = {}
    if mfcc is not None and "mfcc" in requested:
        beat_level_features["mfcc"] = aggregate_by_intervals(mfcc, frame_times, beat_boundaries)
    if chroma is not None and "chroma" in requested:
        beat_level_features["chroma"] = aggregate_by_intervals(chroma, frame_times, beat_boundaries)
    if cens is not None and "cens" in requested:
        beat_level_features["cens"] = aggregate_by_intervals(cens, frame_times, beat_boundaries)
    if need_stm:
        beat_level_features["stm"] = aggregate_by_intervals(stm, stm_times, beat_boundaries)
    for name, features in orthogonal_features.items():
        if name in requested:
            beat_level_features[name] = aggregate_by_intervals(features, frame_times, beat_boundaries)

    ssm_map = {name: build_ssm(features) for name, features in beat_level_features.items()}
    if "sections" in requested and sections:
        ssm_map["sections"] = build_section_ssm(sections, beat_boundaries)
    if "fused" in requested:
        fusion_inputs = [matrix for name, matrix in ssm_map.items() if name != "sections" and matrix.size]
        if fusion_inputs:
            ssm_map["fused"] = np.mean(fusion_inputs, axis=0).astype(np.float32)

    stem = audio_path.stem
    npz_path = out_dir / f"{stem}_features.npz"
    preview_paths = [out_dir / f"{stem}_preview.{fmt}" for fmt in cfg.preview_formats]
    summary_path = out_dir / f"{stem}_summary.json"

    npz_payload: dict[str, object] = {
        "audio_path": str(audio_path),
        "sr": cfg.sr,
        "duration": duration,
        "beat_boundaries": beat_boundaries,
        "beat_downbeat_boundaries": downbeat_boundaries if downbeat_boundaries is not None else np.array([]),
    }
    if frame_times.size:
        npz_payload["frame_times"] = frame_times
    if need_stm:
        npz_payload["stm"] = stm
        npz_payload["stm_times"] = stm_times
        npz_payload["stm_effective_window_s"] = stm_effective_window_s
        npz_payload["onset_env"] = onset_env
    raw_feature_map = {
        "mfcc": mfcc,
        "chroma": chroma,
        "cens": cens,
        **orthogonal_features,
    }
    for name, features in raw_feature_map.items():
        if features is not None and name in requested:
            npz_payload[name] = features
    for name, features in beat_level_features.items():
        npz_payload[f"beat_{name}"] = features
    for name, matrix in ssm_map.items():
        npz_payload[f"ssm_{name}"] = matrix
    np.savez_compressed(npz_path, **npz_payload)

    save_previews(
        preview_paths,
        ssm_map,
        y,
        cfg.sr,
        beat_boundaries,
        downbeat_boundaries,
        sections,
        cfg.preview_features,
    )

    summary = {
        "audio_path": str(audio_path),
        "duration_s": duration,
        "config": asdict(cfg),
        "computed_features": sorted(beat_level_features.keys()) + ([name for name in ("sections", "fused") if name in ssm_map]),
        "outputs": {
            "features_npz": str(npz_path),
            "preview_png": str(preview_paths[0]) if "png" in cfg.preview_formats else "",
            "preview_svg": str(out_dir / f"{stem}_preview.svg") if "svg" in cfg.preview_formats else "",
            "preview_files": [str(path) for path in preview_paths],
            "summary_json": str(summary_path),
        },
        "shapes": {},
        "diagnostics": {
            "beat_intervals": int(max(0, len(beat_boundaries) - 1)),
            "section_annotations_loaded": bool(sections),
        },
    }
    if need_stm:
        summary["shapes"]["stm"] = list(stm.shape)
        summary["shapes"]["beat_stm"] = list(beat_level_features["stm"].shape)
    for name, features in raw_feature_map.items():
        if features is not None and name in requested:
            summary["shapes"][name] = list(features.shape)
    for name, features in beat_level_features.items():
        summary["shapes"][f"beat_{name}"] = list(features.shape)
    for name, matrix in ssm_map.items():
        summary["shapes"][f"ssm_{name}"] = list(matrix.shape)
    summary["diagnostics"].update(orthogonal_diagnostics)
    beat_intervals = np.diff(beat_boundaries)
    stm_beat_counts = (
        count_beats_per_stm_window(stm_times, beat_boundaries, stm_effective_window_s, duration)
        if need_stm
        else np.zeros(0, dtype=np.int32)
    )
    summary["diagnostics"].update(
        {
            "beat_median_interval_s": float(np.median(beat_intervals)) if beat_intervals.size else 0.0,
            "beat_max_interval_s": float(np.max(beat_intervals)) if beat_intervals.size else 0.0,
            "stm_effective_window_s": float(stm_effective_window_s),
            "stm_min_beats_requested": int(cfg.stm_min_beats),
            "stm_min_estimated_beats_per_window": int(np.min(stm_beat_counts)) if stm_beat_counts.size else 0,
            "stm_median_estimated_beats_per_window": float(np.median(stm_beat_counts)) if stm_beat_counts.size else 0.0,
        }
    )
    write_summary(summary_path, summary)
    return summary
