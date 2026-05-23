"""Configuration, audio loading, and annotation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from runtime_env import ensure_local_cache

ensure_local_cache()

import librosa
import numpy as np
import soundfile as sf
import yaml


SECTION_LABEL_MAP = {
    "bridge": "bridge",
    "chorus": "chorus",
    "ending": "ending",
    "intro": "intro",
    "nothing": "nothing",
    "post-chorus": "post-chorus",
    "pre-chorus": "pre-chorus",
    "verse": "verse",
}


@dataclass
class FeatureConfig:
    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 40
    n_mfcc: int = 20
    beat_tracking_method: str = "beat_this"
    beat_tracking_model_path: str | None = None
    sections_dir: Path | None = None
    computing_features: tuple[str, ...] = ("stm", "cens", "sections", "fused")
    preview_features: tuple[str, ...] = ("stm", "cens", "sections")
    preview_formats: tuple[str, ...] = ("png", "svg")
    stm_window_s: float = 8.0
    stm_hop_s: float = 0.5
    stm_min_beats: int = 5
    stm_coeffs: int = 400
    f0_min_note: str = "C2"
    f0_max_note: str = "C7"
    f0_confidence_threshold: float = 0.2
    f0_max_interp_gap_s: float = 0.25

FEATURE_NAME_ORDER = (
    "stm",
    "mfcc",
    "chroma",
    "cens",
    "arrangement",
    "vocal",
    "bass",
    "tonnetz",
    "density",
    "sections",
    "fused",
)
PREVIEW_FEATURE_ORDER = FEATURE_NAME_ORDER


def normalize_section_label(label: str) -> str:
    normalized = label.strip().strip('"').lower()
    for prefix, grouped in SECTION_LABEL_MAP.items():
        if normalized == prefix or normalized.startswith(f"{prefix} "):
            return grouped
    return normalized


def load_sections(sections_dir: Path | None, audio_id: str) -> list[tuple[float, float, str]]:
    if sections_dir is None:
        return []
    labels_path = sections_dir / f"{audio_id}.CHORUS.TXT"
    if not labels_path.exists():
        return []
    sections = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            start_cs, end_cs, label = parts
            sections.append(
                (
                    int(start_cs) / 100.0,
                    int(end_cs) / 100.0,
                    normalize_section_label(label),
                )
            )
    return sections


def _normalize_feature_names(features: Iterable[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not features:
        return default
    normalized = []
    for name in features:
        key = str(name).strip().lower()
        if key not in FEATURE_NAME_ORDER:
            raise ValueError(
                f"Unknown feature name: {name!r}. Valid options: {', '.join(FEATURE_NAME_ORDER)}"
            )
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def normalize_computing_features(features: Iterable[str] | None) -> tuple[str, ...]:
    return _normalize_feature_names(features, FeatureConfig.computing_features)


def normalize_preview_features(features: Iterable[str] | None) -> tuple[str, ...]:
    return _normalize_feature_names(features, FeatureConfig.preview_features)


def normalize_preview_formats(formats: Iterable[str] | None) -> tuple[str, ...]:
    valid_formats = {"png", "svg"}
    if not formats:
        return FeatureConfig.preview_formats
    normalized = []
    for fmt in formats:
        key = str(fmt).strip().lower()
        if key not in valid_formats:
            raise ValueError(f"Unknown preview format: {fmt!r}. Valid options: png, svg")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def config_from_mapping(data: dict[str, Any]) -> FeatureConfig:
    payload = dict(data)
    for path_key in (
        "sections_dir",
        # TODO: extend for ssl models config paths
        ):
        if path_key in payload and payload[path_key] is not None:
            payload[path_key] = Path(payload[path_key])
    if "computing_features" in payload:
        payload["computing_features"] = normalize_computing_features(payload["computing_features"])
    if "preview_features" in payload:
        payload["preview_features"] = normalize_preview_features(payload["preview_features"])
    if "preview_formats" in payload:
        payload["preview_formats"] = normalize_preview_formats(payload["preview_formats"])
    return FeatureConfig(**payload)


def load_yaml_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config at {path} must contain a mapping at the top level.")
    return payload


def load_runtime_config(config_path: Path | None) -> tuple[dict[str, Any], FeatureConfig]:
    raw = load_yaml_config(config_path) if config_path is not None else {}
    feature_cfg = raw.get("feature_config", raw)
    if not isinstance(feature_cfg, dict):
        raise ValueError("`feature_config` must be a mapping when present in the YAML config.")
    return raw, config_from_mapping(feature_cfg)


def resolve_value(cli_value: Any, yaml_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default_value


def load_audio(audio_path: Path, sr: int) -> tuple[np.ndarray, float]:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, float(duration)


def make_demo_audio(path: Path, sr: int) -> None:
    seconds = 36.0
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    y = np.zeros_like(t)
    sections = [
        (0.0, 12.0, 220.0, 2.0),
        (12.0, 24.0, 330.0, 3.0),
        (24.0, 36.0, 220.0, 2.0),
    ]
    for start, end, freq, beat_hz in sections:
        mask = (t >= start) & (t < end)
        local_t = t[mask] - start
        carrier = 0.35 * np.sin(2 * np.pi * freq * local_t)
        harmonic = 0.15 * np.sin(2 * np.pi * 2 * freq * local_t)
        pulse = 0.5 + 0.5 * np.maximum(0.0, np.sin(2 * np.pi * beat_hz * local_t))
        y[mask] = (carrier + harmonic) * (0.4 + 0.6 * pulse)
    sf.write(path, y.astype(np.float32), sr)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
