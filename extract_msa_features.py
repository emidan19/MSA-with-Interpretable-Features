#!/usr/bin/env python3
"""CLI entrypoint for single-file MSA feature extraction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from data_loading import (
    FEATURE_NAME_ORDER,
    FeatureConfig,
    load_runtime_config,
    make_demo_audio,
    normalize_computing_features,
    normalize_preview_features,
    resolve_value,
)
from pipeline import extract_all


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path, help="Path to an audio file.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--demo", action="store_true", help="Generate and process a synthetic demo audio file.")
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--beat-tracking-method", choices=["librosa", "beat_this", "madmom"], default=None)
    parser.add_argument("--sections-dir", type=Path, default=None)
    parser.add_argument("--stm-coeffs", type=int, default=None)
    parser.add_argument("--stm-window-s", type=float, default=None)
    parser.add_argument("--stm-hop-s", type=float, default=None)
    parser.add_argument("--stm-min-beats", type=int, default=None)
    parser.add_argument(
        "--preview-features",
        nargs="+",
        default=None,
        help=f"Subset of preview SSMs to render. Options: {', '.join(FEATURE_NAME_ORDER)}",
    )
    return parser.parse_args(argv)


def build_effective_config(args: argparse.Namespace, yaml_cfg: FeatureConfig) -> FeatureConfig:
    return FeatureConfig(
        **{
            **asdict(yaml_cfg),
            "sr": resolve_value(args.sr, yaml_cfg.sr, FeatureConfig.sr),
            "hop_length": resolve_value(args.hop_length, yaml_cfg.hop_length, FeatureConfig.hop_length),
            "beat_tracking_method": resolve_value(
                args.beat_tracking_method,
                yaml_cfg.beat_tracking_method,
                FeatureConfig.beat_tracking_method,
            ),
            "sections_dir": resolve_value(args.sections_dir, yaml_cfg.sections_dir, FeatureConfig.sections_dir),
            "computing_features": normalize_computing_features(yaml_cfg.computing_features),
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


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    raw_config, yaml_cfg = load_runtime_config(args.config)
    cfg = build_effective_config(args, yaml_cfg)

    audio_path = args.audio
    if audio_path is None and "audio_path" in raw_config:
        audio_path = Path(raw_config["audio_path"])
    out_dir = args.out_dir
    if out_dir is None and "out_dir" in raw_config:
        out_dir = Path(raw_config["out_dir"])
    if out_dir is None:
        out_dir = Path("feature_outputs")
    if args.demo:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / "demo_aba_song.wav"
        make_demo_audio(audio_path, cfg.sr)
    if audio_path is None:
        raise SystemExit("Provide an audio path, or pass --demo.")
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    summary = extract_all(audio_path, out_dir, cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
