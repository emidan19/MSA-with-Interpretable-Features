#!/usr/bin/env python3
"""Batch wrapper for `extract_msa_features.py` over a directory of audio files."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

from extract_msa_features import FeatureConfig, extract_all


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path("data/rwc_p_20/audio"))
    parser.add_argument("--out-dir", type=Path, default=Path("feature_outputs/rwc_p_20"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sr", type=int, default=FeatureConfig.sr)
    parser.add_argument("--hop-length", type=int, default=FeatureConfig.hop_length)
    parser.add_argument("--stm-coeffs", type=int, default=FeatureConfig.stm_coeffs)
    parser.add_argument("--stm-window-s", type=float, default=FeatureConfig.stm_window_s)
    parser.add_argument("--stm-hop-s", type=float, default=FeatureConfig.stm_hop_s)
    parser.add_argument("--stm-min-beats", type=int, default=FeatureConfig.stm_min_beats)
    parser.add_argument("--f0-confidence-threshold", type=float, default=FeatureConfig.f0_confidence_threshold)
    parser.add_argument("--f0-max-interp-gap-s", type=float, default=FeatureConfig.f0_max_interp_gap_s)
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
    cfg = FeatureConfig(
        sr=args.sr,
        hop_length=args.hop_length,
        stm_coeffs=args.stm_coeffs,
        stm_window_s=args.stm_window_s,
        stm_hop_s=args.stm_hop_s,
        stm_min_beats=args.stm_min_beats,
        f0_confidence_threshold=args.f0_confidence_threshold,
        f0_max_interp_gap_s=args.f0_max_interp_gap_s,
    )
    audio_paths = sorted(args.audio_dir.glob("*.wav"))[: args.limit]
    if not audio_paths:
        raise SystemExit(f"No WAV files found in {args.audio_dir}")

    rows: list[dict] = []
    for idx, audio_path in enumerate(audio_paths, start=1):
        print(f"[{idx:02d}/{len(audio_paths):02d}] {audio_path.name}", flush=True)
        start = time.perf_counter()
        try:
            summary = extract_all(audio_path, args.out_dir, cfg)
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
                    "f0_voiced_ratio": round(summary["diagnostics"]["f0_voiced_ratio"], 6),
                    "f0_mean_confidence": round(summary["diagnostics"]["f0_mean_confidence"], 6),
                    "f0_confidence_threshold": round(summary["diagnostics"]["f0_confidence_threshold"], 6),
                    "f0_max_interp_gap_s": round(summary["diagnostics"]["f0_max_interp_gap_s"], 6),
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
                    "f0_shape": "x".join(map(str, summary["shapes"]["f0_features"])),
                    "f0_contour_shape": "x".join(map(str, summary["shapes"]["f0_contour_features"])),
                    "ssm_shape": "x".join(map(str, summary["shapes"]["ssm_fused"])),
                    "features_npz": summary["outputs"]["features_npz"],
                    "preview_png": summary["outputs"]["preview_png"],
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
            write_index(args.out_dir, rows)
            raise
        write_index(args.out_dir, rows)

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
