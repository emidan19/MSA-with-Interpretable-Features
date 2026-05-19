#!/usr/bin/env python3
"""Batch wrapper for `extract_msa_features.py` over a directory of audio files."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from extract_msa_features import (
    FeatureConfig,
    extract_all,
    load_runtime_config,
    normalize_preview_features,
    resolve_value,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--sections-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument(
        "--beat-tracking-method",
        choices=["librosa", "beat_this", "madmom"],
        default=None,
    )
    parser.add_argument("--stm-coeffs", type=int, default=None)
    parser.add_argument("--stm-window-s", type=float, default=None)
    parser.add_argument("--stm-hop-s", type=float, default=None)
    parser.add_argument("--stm-min-beats", type=int, default=None)
    parser.add_argument("--preview-features", nargs="+", default=None)
    return parser.parse_args(argv)


def write_index(out_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "batch_summary.json"
    csv_path = out_dir / "batch_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    raw_config, yaml_cfg = load_runtime_config(args.config)
    cfg = FeatureConfig(
        **{
            **asdict(yaml_cfg),
            "sr": resolve_value(args.sr, yaml_cfg.sr, FeatureConfig.sr),
            "hop_length": resolve_value(args.hop_length, yaml_cfg.hop_length, FeatureConfig.hop_length),
            "beat_tracking_method": resolve_value(
                args.beat_tracking_method, yaml_cfg.beat_tracking_method, FeatureConfig.beat_tracking_method
            ),
            "sections_dir": resolve_value(args.sections_dir, yaml_cfg.sections_dir, FeatureConfig.sections_dir),
            "preview_features": normalize_preview_features(args.preview_features)
            if args.preview_features is not None
            else yaml_cfg.preview_features,
            "preview_formats": yaml_cfg.preview_formats,
            "stm_coeffs": resolve_value(args.stm_coeffs, yaml_cfg.stm_coeffs, FeatureConfig.stm_coeffs),
            "stm_window_s": resolve_value(args.stm_window_s, yaml_cfg.stm_window_s, FeatureConfig.stm_window_s),
            "stm_hop_s": resolve_value(args.stm_hop_s, yaml_cfg.stm_hop_s, FeatureConfig.stm_hop_s),
            "stm_min_beats": resolve_value(args.stm_min_beats, yaml_cfg.stm_min_beats, FeatureConfig.stm_min_beats),
        }
    )
    audio_dir = args.audio_dir
    if audio_dir is None and "audio_dir" in raw_config:
        audio_dir = Path(raw_config["audio_dir"])
    if audio_dir is None:
        audio_dir = Path("data/rwc_p_20/audio")
    out_dir = args.out_dir
    if out_dir is None and "out_dir" in raw_config:
        out_dir = Path(raw_config["out_dir"])
    if out_dir is None:
        out_dir = Path("feature_outputs/rwc_p_20")
    limit = args.limit if args.limit is not None else int(raw_config.get("limit", 20))

    audio_paths = sorted(audio_dir.glob("*.wav"))[:limit]
    if not audio_paths:
        raise SystemExit(f"No WAV files found in {audio_dir}")

    rows: list[dict] = []
    for idx, audio_path in enumerate(audio_paths, start=1):
        print(f"[{idx:02d}/{len(audio_paths):02d}] {audio_path.name}", flush=True)
        start = time.perf_counter()
        try:
            summary = extract_all(audio_path, out_dir, cfg)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "audio": audio_path.name,
                    "status": "ok",
                    "elapsed_s": round(elapsed, 3),
                    "duration_s": round(summary["duration_s"], 3),
                    "beat_intervals": summary["diagnostics"]["beat_intervals"],
                    "beat_median_interval_s": round(summary["diagnostics"]["beat_median_interval_s"], 6),
                    "beat_max_interval_s": round(summary["diagnostics"]["beat_max_interval_s"], 6),
                    "stm_effective_window_s": round(summary["diagnostics"]["stm_effective_window_s"], 6),
                    "stm_min_estimated_beats_per_window": summary["diagnostics"]["stm_min_estimated_beats_per_window"],
                    "stm_median_estimated_beats_per_window": round(summary["diagnostics"]["stm_median_estimated_beats_per_window"], 6),
                    "arrangement_harmonic_ratio_mean": round(
                        summary["diagnostics"]["arrangement_harmonic_ratio_mean"], 6
                    ),
                    "arrangement_percussive_ratio_mean": round(
                        summary["diagnostics"]["arrangement_percussive_ratio_mean"], 6
                    ),
                    "vocal_proxy_mean": round(summary["diagnostics"]["vocal_proxy_mean"], 6),
                    "bass_fraction_mean": round(summary["diagnostics"]["bass_fraction_mean"], 6),
                    "density_onset_mean": round(summary["diagnostics"]["density_onset_mean"], 6),
                    "tonnetz_hcdf_mean": round(summary["diagnostics"]["tonnetz_hcdf_mean"], 6),
                    "stm_shape": "x".join(map(str, summary["shapes"]["stm"])),
                    "mfcc_shape": "x".join(map(str, summary["shapes"]["mfcc"])),
                    "chroma_shape": "x".join(map(str, summary["shapes"]["chroma"])),
                    "cens_shape": "x".join(map(str, summary["shapes"]["cens"])),
                    "arrangement_shape": "x".join(map(str, summary["shapes"]["arrangement"])),
                    "vocal_shape": "x".join(map(str, summary["shapes"]["vocal"])),
                    "bass_shape": "x".join(map(str, summary["shapes"]["bass"])),
                    "tonnetz_shape": "x".join(map(str, summary["shapes"]["tonnetz"])),
                    "density_shape": "x".join(map(str, summary["shapes"]["density"])),
                    "ssm_shape": "x".join(map(str, summary["shapes"]["ssm_fused"])),
                    "features_npz": summary["outputs"]["features_npz"],
                    "preview_png": summary["outputs"]["preview_png"],
                    "preview_svg": summary["outputs"]["preview_svg"],
                    "summary_json": summary["outputs"]["summary_json"],
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "audio": audio_path.name,
                    "status": "error",
                    "elapsed_s": round(elapsed, 3),
                    "error": repr(exc),
                }
            )
            write_index(out_dir, rows)
            raise
        write_index(out_dir, rows)

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
