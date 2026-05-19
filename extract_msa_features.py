#!/usr/bin/env python3
"""Extract exploratory features for music structure analysis.

This script is intentionally conservative: it uses only librosa/numpy/scipy
and computes a small set of descriptors that can feed a downstream MSA
pipeline:

* STM: scale-transform magnitude over local onset-strength autocorrelations.
* MFCC: timbre/texture descriptor.
* Chroma and CENS: harmonic/pitch-class content.
* F0 contour: predominant melody via Essentia/MELODIA, with confidence.
* Arrangement proxies: HPSS/band-energy/source-activity descriptors.
* Bass, vocal, Tonnetz/HCDF, and density proxies for orthogonal SSMs.
* Beat-synchronous summaries and self-similarity matrices for each stream.

The STM implementation follows the usual MIR recipe at a prototype level:
mel/onset strength -> detrend -> local autocorrelation -> direct scale-domain
projection. It is enough to verify feasibility before committing to a more
paper-faithful implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# Keep runtime caches inside the project so sandboxed runs do not fail while
# importing numba-backed librosa modules or building matplotlib font caches.
_CACHE_ROOT = os.path.join(os.getcwd(), ".cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_CACHE_ROOT, "numba"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))
for _cache_dir in (
    os.environ["MPLCONFIGDIR"],
    os.environ["NUMBA_CACHE_DIR"],
    os.environ["XDG_CACHE_HOME"],
):
    os.makedirs(_cache_dir, exist_ok=True)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import detrend
from sklearn.metrics.pairwise import cosine_similarity
import yaml

try:
    import madmom as mm
    from madmom.features import beats as bt
    from madmom.features import downbeats as dbt
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False

try:
    from beat_this.inference import File2Beats
    BEAT_THIS_AVAILABLE = True
except ImportError:
    BEAT_THIS_AVAILABLE = False

import tempfile


def load_sections(sections_dir: Path | None, audio_id: str) -> list[tuple[float, float, str]]:
    """Load section annotations from RWC-style .CHORUS.TXT file.
    
    Assumes annotations are in centiseconds (1/100 s).
    """
    if sections_dir is None:
        return []
    labels_path = sections_dir / f"{audio_id}.CHORUS.TXT"
    if not labels_path.exists():
        return []
    sections = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                start_cs, end_cs, label = parts
                start_time = int(start_cs) / 100.0
                end_time = int(end_cs) / 100.0
                sections.append((start_time, end_time, label.strip('"')))
    return sections


@dataclass
class FeatureConfig:
    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 40
    n_mfcc: int = 20
    beat_tracking_method: str = "beat_this" # "librosa" | "beat_this" | "madmom"
    beat_tracking_model_path: str | None = None
    sections_dir: Path | None = None
    preview_features: tuple[str, ...] = ("stm", "cens", "sections")
    preview_formats: tuple[str, ...] = ("png", "svg")
    stm_window_s: float = 8.0
    stm_hop_s: float = 0.5
    stm_min_beats: int = 5
    stm_coeffs: int = 400


PREVIEW_FEATURE_ORDER = (
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


def safe_nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = safe_nan_to_num(x).astype(np.float32, copy=False)
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def build_ssm(features: np.ndarray) -> np.ndarray:
    """Return cosine self-similarity for columns as time steps."""
    if features.size == 0 or features.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x = zscore_columns(features).T
    return cosine_similarity(x).astype(np.float32)


def build_section_ssm(
    sections: list[tuple[float, float, str]],
    beat_boundaries: np.ndarray,
) -> np.ndarray:
    """Return a beat-level binary reference SSM from annotated sections."""
    n_intervals = max(0, len(beat_boundaries) - 1)
    if not sections or n_intervals == 0:
        return np.zeros((0, 0), dtype=np.float32)

    interval_labels = np.full(n_intervals, -1, dtype=np.int32)
    interval_centers = (beat_boundaries[:-1] + beat_boundaries[1:]) / 2.0

    for section_idx, (start, end, _label) in enumerate(sections):
        mask = (interval_centers >= start) & (interval_centers < end)
        interval_labels[mask] = section_idx

    valid = interval_labels[:, None] == interval_labels[None, :]
    assigned = (interval_labels[:, None] >= 0) & (interval_labels[None, :] >= 0)
    return np.where(assigned, valid, False).astype(np.float32)


def normalize_preview_features(features: Iterable[str] | None) -> tuple[str, ...]:
    if not features:
        return FeatureConfig.preview_features
    normalized = []
    for name in features:
        key = str(name).strip().lower()
        if key not in PREVIEW_FEATURE_ORDER:
            raise ValueError(
                f"Unknown preview feature: {name!r}. Valid options: {', '.join(PREVIEW_FEATURE_ORDER)}"
            )
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


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
    data = dict(data)
    if "sections_dir" in data and data["sections_dir"] is not None:
        data["sections_dir"] = Path(data["sections_dir"])
    if "preview_features" in data:
        data["preview_features"] = normalize_preview_features(data["preview_features"])
    if "preview_formats" in data:
        data["preview_formats"] = normalize_preview_formats(data["preview_formats"])
    return FeatureConfig(**data)


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


def aggregate_by_intervals(
    features: np.ndarray,
    frame_times: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Average feature columns between consecutive time boundaries."""
    features = safe_nan_to_num(features)
    out = np.zeros((features.shape[0], max(0, len(boundaries) - 1)), dtype=np.float32)
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            nearest = np.argmin(np.abs(frame_times - (start + end) / 2.0))
            out[:, idx] = features[:, nearest]
        else:
            out[:, idx] = np.mean(features[:, mask], axis=1)
    return out


def extract_mfcc_chroma_cens(
    y: np.ndarray,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=cfg.sr,
        hop_length=cfg.hop_length,
    )
    cens = librosa.feature.chroma_cens(
        y=y,
        sr=cfg.sr,
        hop_length=cfg.hop_length,
    )
    frame_times = librosa.frames_to_time(
        np.arange(mfcc.shape[1]),
        sr=cfg.sr,
        hop_length=cfg.hop_length,
    )
    return mfcc.astype(np.float32), chroma.astype(np.float32), cens.astype(np.float32), frame_times


def fix_feature_length(x: np.ndarray, n_frames: int) -> np.ndarray:
    if x.ndim == 1:
        return librosa.util.fix_length(x, size=n_frames).astype(np.float32)
    return librosa.util.fix_length(x, size=n_frames, axis=1).astype(np.float32)


def band_fraction(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    total = np.sum(power, axis=0) + 1e-10
    if not np.any(mask):
        return np.zeros(power.shape[1], dtype=np.float32)
    return (np.sum(power[mask], axis=0) / total).astype(np.float32)


def spectral_entropy(power: np.ndarray) -> np.ndarray:
    probs = power / (np.sum(power, axis=0, keepdims=True) + 1e-10)
    entropy = -np.sum(probs * np.log2(probs + 1e-10), axis=0)
    return (entropy / np.log2(max(2, power.shape[0]))).astype(np.float32)


def spectral_flux(mag: np.ndarray) -> np.ndarray:
    norm = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)
    diff = np.diff(norm, axis=1)
    flux = np.r_[0.0, np.sqrt(np.sum(diff * diff, axis=0))]
    return flux.astype(np.float32)


def low_band_centroid(power: np.ndarray, freqs: np.ndarray, high_hz: float = 300.0) -> np.ndarray:
    mask = (freqs >= 30.0) & (freqs <= high_hz)
    if not np.any(mask):
        return np.zeros(power.shape[1], dtype=np.float32)
    local = power[mask]
    local_freqs = freqs[mask, None]
    centroid = np.sum(local * local_freqs, axis=0) / (np.sum(local, axis=0) + 1e-10)
    return (centroid / high_hz).astype(np.float32)


def extract_orthogonal_proxy_features(
    y: np.ndarray,
    cfg: FeatureConfig,
    chroma: np.ndarray,
    cens: np.ndarray,
    frame_times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Build lightweight proxies for arrangement/source/function cues.

    This is not neural source separation. It is a reproducible first-pass
    approximation based on HPSS, spectral bands, low-end chroma and harmonic
    change, intended to test whether these axes add signal beyond MFCC/CENS.
    """
    stft = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    mag = np.abs(stft).astype(np.float32)
    power = (mag * mag).astype(np.float32)
    n_frames = mag.shape[1]
    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)

    harm_mag, perc_mag = librosa.decompose.hpss(mag)
    harm_power = (harm_mag * harm_mag).astype(np.float32)
    perc_power = (perc_mag * perc_mag).astype(np.float32)
    total_power = np.sum(power, axis=0) + 1e-10

    bands = {
        "sub": band_fraction(power, freqs, 20.0, 80.0),
        "bass": band_fraction(power, freqs, 80.0, 250.0),
        "low_mid": band_fraction(power, freqs, 250.0, 500.0),
        "mid": band_fraction(power, freqs, 500.0, 2000.0),
        "presence": band_fraction(power, freqs, 2000.0, 5000.0),
        "air": band_fraction(power, freqs, 5000.0, cfg.sr / 2.0),
    }
    harmonic_ratio = (np.sum(harm_power, axis=0) / total_power).astype(np.float32)
    percussive_ratio = (np.sum(perc_power, axis=0) / total_power).astype(np.float32)
    vocal_band = band_fraction(power, freqs, 300.0, 3400.0)
    vocal_harmonic = (np.sum(harm_power[(freqs >= 300.0) & (freqs < 3400.0)], axis=0) / total_power).astype(
        np.float32
    )
    bass_fraction = band_fraction(power, freqs, 40.0, 250.0)
    sub_fraction = band_fraction(power, freqs, 20.0, 80.0)

    rms = librosa.feature.rms(S=mag, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
    centroid = librosa.feature.spectral_centroid(S=mag, sr=cfg.sr)[0] / (cfg.sr / 2.0)
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=cfg.sr)[0] / (cfg.sr / 2.0)
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=cfg.sr, roll_percent=0.85)[0] / (cfg.sr / 2.0)
    flatness = librosa.feature.spectral_flatness(S=mag)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(mag, ref=np.max), sr=cfg.sr, hop_length=cfg.hop_length)
    onset_env = fix_feature_length(onset_env, n_frames)
    flux = spectral_flux(mag)
    entropy = spectral_entropy(power)
    contrast = librosa.feature.spectral_contrast(S=mag, sr=cfg.sr)
    contrast_mean = np.mean(contrast, axis=0).astype(np.float32)
    contrast_std = np.std(contrast, axis=0).astype(np.float32)

    low_mag = mag.copy()
    low_mag[freqs > 300.0, :] = 0.0
    bass_chroma = librosa.feature.chroma_stft(S=low_mag, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    low_flux = spectral_flux(low_mag)
    bass_centroid = low_band_centroid(power, freqs)

    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=cfg.sr)
    tonnetz = fix_feature_length(tonnetz, frame_times.size)
    cens_fixed = fix_feature_length(cens, frame_times.size)
    chroma_delta = np.r_[0.0, np.linalg.norm(np.diff(cens_fixed, axis=1), axis=0)].astype(np.float32)
    tonnetz_delta = np.r_[0.0, np.linalg.norm(np.diff(tonnetz, axis=1), axis=0)].astype(np.float32)
    harmonic_focus = np.max(cens_fixed, axis=0).astype(np.float32)

    arrangement = np.vstack(
        [
            bands["sub"],
            bands["bass"],
            bands["low_mid"],
            bands["mid"],
            bands["presence"],
            bands["air"],
            harmonic_ratio,
            percussive_ratio,
            rms,
            centroid,
            bandwidth,
            rolloff,
            flatness,
            entropy,
        ]
    )
    vocal = np.vstack(
        [
            vocal_band,
            vocal_harmonic,
            harmonic_ratio,
            bands["presence"],
            contrast_mean,
            contrast_std,
        ]
    )
    bass = np.vstack(
        [
            bass_chroma,
            sub_fraction,
            bass_fraction,
            bass_centroid,
            low_flux,
        ]
    )
    tonnetz_hcdf = np.vstack(
        [
            tonnetz,
            chroma_delta,
            tonnetz_delta,
            harmonic_focus,
        ]
    )
    density = np.vstack(
        [
            rms,
            onset_env,
            percussive_ratio,
            flux,
            zcr,
            flatness,
            entropy,
            bands["air"],
        ]
    )

    features = {
        "arrangement": safe_nan_to_num(arrangement).astype(np.float32),
        "vocal": safe_nan_to_num(vocal).astype(np.float32),
        "bass": safe_nan_to_num(bass).astype(np.float32),
        "tonnetz": safe_nan_to_num(tonnetz_hcdf).astype(np.float32),
        "density": safe_nan_to_num(density).astype(np.float32),
    }
    diagnostics = {
        "arrangement_harmonic_ratio_mean": float(np.mean(harmonic_ratio)),
        "arrangement_percussive_ratio_mean": float(np.mean(percussive_ratio)),
        "vocal_proxy_mean": float(np.mean(vocal_band * harmonic_ratio)),
        "bass_fraction_mean": float(np.mean(bass_fraction)),
        "density_onset_mean": float(np.mean(onset_env)),
        "tonnetz_hcdf_mean": float(np.mean(chroma_delta)),
    }
    return features, diagnostics


def correct_octave_jumps(midi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    corrected = midi.copy()
    previous = np.nan
    for idx, value in enumerate(corrected):
        if not valid[idx] or not np.isfinite(value):
            continue
        if np.isfinite(previous):
            candidates = np.asarray([value - 24.0, value - 12.0, value, value + 12.0, value + 24.0])
            value = float(candidates[np.argmin(np.abs(candidates - previous))])
            corrected[idx] = value
        previous = value
    return corrected


def interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap_frames: int) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    out[~valid] = np.nan
    valid_idx = np.flatnonzero(np.isfinite(out))
    if valid_idx.size < 2:
        return safe_nan_to_num(out)
    for left, right in zip(valid_idx[:-1], valid_idx[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap_frames:
            out[left + 1 : right] = np.linspace(out[left], out[right], gap + 2, dtype=np.float32)[1:-1]
    return safe_nan_to_num(out)


def extract_melodia(y: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract predominant melody F0 using Essentia's MELODIA implementation."""
    try:
        import essentia.standard as es
    except ImportError as exc:
        raise RuntimeError(
            "Essentia is required for MELODIA. Install it with: "
            "conda run -n emilio_msa_features python -m pip install essentia"
        ) from exc

    extractor = es.PredominantPitchMelodia(
        frameSize=cfg.n_fft,
        hopSize=cfg.hop_length,
        sampleRate=cfg.sr,
        minFrequency=librosa.note_to_hz(cfg.f0_min_note),
        maxFrequency=librosa.note_to_hz(cfg.f0_max_note),
        guessUnvoiced=False,
    )
    f0, confidence = extractor(y.astype(np.float32))
    f0 = safe_nan_to_num(np.asarray(f0)).astype(np.float32)
    confidence = safe_nan_to_num(np.asarray(confidence)).astype(np.float32)
    voiced_flag = ((f0 > 0.0) & (confidence >= cfg.f0_confidence_threshold)).astype(np.float32)

    midi = safe_nan_to_num(librosa.hz_to_midi(np.maximum(f0, 1e-3)))
    midi[f0 <= 0] = 0.0
    valid = voiced_flag > 0.5
    corrected_midi = correct_octave_jumps(midi, valid)
    max_gap_frames = max(1, int(round(cfg.f0_max_interp_gap_s * cfg.sr / cfg.hop_length)))
    smooth_midi = interpolate_short_gaps(corrected_midi, valid, max_gap_frames)
    smooth_hz = safe_nan_to_num(librosa.midi_to_hz(np.maximum(smooth_midi, 0.0))).astype(np.float32)
    smooth_hz[smooth_midi <= 0.0] = 0.0
    midi_delta = np.r_[0.0, np.diff(smooth_midi)].astype(np.float32)
    midi_delta[np.abs(midi_delta) > 24.0] = 0.0

    f0_features = np.vstack(
        [
            smooth_hz,
            smooth_midi,
            voiced_flag,
            confidence,
        ]
    ).astype(np.float32)
    f0_contour_features = np.vstack(
        [
            smooth_midi,
            midi_delta,
            confidence,
        ]
    ).astype(np.float32)
    f0_times = librosa.frames_to_time(
        np.arange(f0_features.shape[1]),
        sr=cfg.sr,
        hop_length=cfg.hop_length,
    )
    return f0_features, confidence, f0_times, f0_contour_features


def autocorrelation_positive(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2 :]
    if ac[0] > 0:
        ac = ac / ac[0]
    return ac


def direct_scale_transform_magnitude(
    autocorr: np.ndarray,
    lag_step_s: float,
    n_coeffs: int,
) -> np.ndarray:
    """Approximate scale-transform magnitudes for one autocorrelation frame.

    The direct scale transform is projected on exp(-j 2 pi c log(t)).
    We skip lag zero because log(0) is undefined.
    """
    x = safe_nan_to_num(autocorr[1:])
    if x.size < 4:
        return np.zeros(n_coeffs, dtype=np.float32)

    lag_times = np.arange(1, x.size + 1, dtype=np.float64) * lag_step_s
    log_lags = np.log(lag_times)
    x = x / np.sqrt(np.maximum(lag_times, 1e-12))

    coeffs = np.arange(n_coeffs, dtype=np.float64)[:, None]
    kernel = np.exp(-2j * np.pi * coeffs * log_lags[None, :] / (np.ptp(log_lags) + 1e-12))
    stm = np.abs(kernel @ x)
    norm = np.linalg.norm(stm)
    if norm > 0:
        stm = stm / norm
    return stm.astype(np.float32)


def robust_beat_period_s(beat_boundaries: np.ndarray) -> float | None:
    intervals = np.diff(np.asarray(beat_boundaries, dtype=np.float32))
    intervals = intervals[(intervals >= 0.25) & (intervals <= 2.0)]
    if intervals.size == 0:
        return None
    return float(np.percentile(intervals, 90))


def resolve_stm_window_s(cfg: FeatureConfig, beat_boundaries: np.ndarray | None = None) -> float:
    """Return an STM window long enough to include several beats.

    The configured 8 s default already covers many beats in pop music. This
    guard matters when a user runs a much shorter window or a slower track.
    """
    min_window_s = 0.0
    if beat_boundaries is not None:
        beat_period = robust_beat_period_s(beat_boundaries)
        if beat_period is not None:
            min_window_s = cfg.stm_min_beats * beat_period
    return float(max(cfg.stm_window_s, min_window_s))


def count_beats_per_stm_window(
    stm_times: np.ndarray,
    beat_boundaries: np.ndarray,
    window_s: float,
    duration_s: float,
) -> np.ndarray:
    counts = []
    half_window = window_s / 2.0
    for center in stm_times:
        start = max(0.0, float(center) - half_window)
        end = min(duration_s, float(center) + half_window)
        counts.append(int(np.count_nonzero((beat_boundaries >= start) & (beat_boundaries <= end))))
    return np.asarray(counts, dtype=np.int32)


def extract_stm(
    y: np.ndarray,
    cfg: FeatureConfig,
    beat_boundaries: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    onset_env = librosa.onset.onset_strength(
        S=log_mel,
        sr=cfg.sr,
        hop_length=cfg.hop_length,
        aggregate=np.mean,
    )
    onset_env = detrend(safe_nan_to_num(onset_env))

    frame_rate = cfg.sr / cfg.hop_length
    effective_window_s = resolve_stm_window_s(cfg, beat_boundaries)
    win_frames = max(8, int(round(effective_window_s * frame_rate)))
    hop_frames = max(1, int(round(cfg.stm_hop_s * frame_rate)))
    lag_step_s = cfg.hop_length / cfg.sr

    vectors = []
    centers = []
    if len(onset_env) < win_frames:
        padded = np.pad(onset_env, (0, win_frames - len(onset_env)))
        ac = autocorrelation_positive(padded)
        vectors.append(direct_scale_transform_magnitude(ac, lag_step_s, cfg.stm_coeffs))
        centers.append(min(librosa.get_duration(y=y, sr=cfg.sr) / 2.0, effective_window_s / 2.0))
    else:
        for start in range(0, len(onset_env) - win_frames + 1, hop_frames):
            stop = start + win_frames
            ac = autocorrelation_positive(onset_env[start:stop])
            vectors.append(direct_scale_transform_magnitude(ac, lag_step_s, cfg.stm_coeffs))
            centers.append(((start + stop) / 2.0) * lag_step_s)

    stm = np.asarray(vectors, dtype=np.float32).T
    stm_times = np.asarray(centers, dtype=np.float32)
    return stm, stm_times, onset_env.astype(np.float32), effective_window_s


def estimate_beats_and_downbeats(y: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray | None]:

    method = cfg.beat_tracking_method.lower()

    duration = librosa.get_duration(y=y, sr=cfg.sr)

    if method == "librosa":
        _, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=cfg.sr,
            hop_length=cfg.hop_length,
            trim=False,
        )
    
        beat_times = librosa.frames_to_time(beat_frames, sr=cfg.sr, hop_length=cfg.hop_length)

        boundaries = np.unique(np.r_[0.0, beat_times, duration])
        if len(boundaries) < 4:
            boundaries = np.arange(0.0, duration + 1.0, 1.0)
            if boundaries[-1] < duration:
                boundaries = np.r_[boundaries, duration]
        boundaries = regularize_beat_boundaries(boundaries, duration)
        return boundaries.astype(np.float32), None

    elif method == "madmom":
        if not MADMOM_AVAILABLE:
            raise ImportError("madmom not available")
        # Beats
        act_beats = bt.TCNBeatProcessor()(y)
        proc_beats = bt.BeatTrackingProcessor(fps=100)
        beat_times = proc_beats(act_beats)
        beat_boundaries = np.unique(np.r_[0.0, beat_times, duration])
        beat_boundaries = regularize_beat_boundaries(beat_boundaries, duration)
        # Downbeats
        act_downbeats = dbt.RNNDownBeatProcessor()(y)
        proc_downbeats = dbt.DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
        song_beats = proc_downbeats(act_downbeats)
        downbeat_times = [song_beats[0][0]]
        for beat in song_beats[1:]:
            if beat[1] == 1:
                downbeat_times.append(beat[0])
        downbeat_times.append(duration)
        downbeat_boundaries = np.unique(np.r_[0.0, downbeat_times, duration])
        downbeat_boundaries = regularize_beat_boundaries(downbeat_boundaries, duration)
        return beat_boundaries.astype(np.float32), downbeat_boundaries.astype(np.float32)

    elif method == "beat_this":
        if not BEAT_THIS_AVAILABLE:
            raise ImportError("beat_this not available")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, y, cfg.sr)
            file2beats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
            beats, downbeats = file2beats(f.name)
        beat_times = np.array(beats)
        downbeat_times = np.array(downbeats.tolist() + [beats[-1]])
        beat_boundaries = np.unique(np.r_[0.0, beat_times, duration])
        beat_boundaries = regularize_beat_boundaries(beat_boundaries, duration)
        downbeat_boundaries = np.unique(np.r_[0.0, downbeat_times, duration])
        downbeat_boundaries = regularize_beat_boundaries(downbeat_boundaries, duration)
        return beat_boundaries.astype(np.float32), downbeat_boundaries.astype(np.float32)

    else:
        raise ValueError(f"Unknown beat tracking method: {method}")


def regularize_beat_boundaries(boundaries: np.ndarray, duration: float) -> np.ndarray:
    """Fill large gaps left by beat tracking so beat-synchronous features stay usable."""
    boundaries = np.unique(np.r_[0.0, safe_nan_to_num(boundaries), duration]).astype(np.float32)
    intervals = np.diff(boundaries)
    reliable = intervals[(intervals >= 0.25) & (intervals <= 2.0)]
    if reliable.size == 0:
        return boundaries
    median_interval = float(np.median(reliable))
    max_gap = max(2.5 * median_interval, 2.0)
    filled = [float(boundaries[0])]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start_f = float(start)
        end_f = float(end)
        if end_f - start_f > max_gap:
            next_beat = start_f + median_interval
            while next_beat < end_f:
                filled.append(next_beat)
                next_beat += median_interval
        filled.append(end_f)
    return np.unique(np.asarray(filled, dtype=np.float32))


def make_demo_audio(path: Path, sr: int) -> None:
    """Create a tiny A/B/A synthetic audio file for smoke testing."""
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


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


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

    # Fill sections
    section_colors = plt.cm.tab10(np.linspace(0, 1, max(len(sections), 1)))

    if sections:
        for i, (start, end, label) in enumerate(sections):
            color = section_colors[i % len(section_colors)]
            axes[0].axvspan(start, end, alpha=0.1, color=color)
            # Add label text
            mid_time = (start + end) / 2
            v_pos = ((0.95* len(sections)-i)/len(sections) - 0.5)
            axes[0].text(mid_time, 2 * v_pos % 1 - 0.5, label, ha='center', va='top', fontsize=6, rotation=0, transform=axes[0].transData)

    for ax, (name, matrix) in zip(axes[1:], preview_items):
        if matrix.size:
            ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=-1, vmax=1)
        ax.set_title(f"SSM: {name}")
        ax.set_xlabel("beat interval")
        ax.set_ylabel("section" if sections else "beat interval")
        if sections:
            ax.set_yticks([]) 
        ax.tick_params(axis='both', labelsize=6)   

        # Add section boundaries
        for i, (start, end, label) in enumerate(sections):
            idx = max(0, np.searchsorted(beat_boundaries, start, side='left') - 1)
            if idx < matrix.shape[0]:
                color = section_colors[i % len(section_colors)]
                ax.axvline(idx, color=color, linestyle='--', linewidth=1)
                ax.axhline(idx, color=color, linestyle='--', linewidth=1)
                # Label on vertical axis
                ax.text(-0.5, idx, label, ha='right', va='center', fontsize=6, transform=ax.transData)

    for ax in axes[n_panels:]:
        ax.axis("off")

    for output_path in output_paths:
        save_kwargs = {"dpi": 160} if output_path.suffix.lower() == ".png" else {}
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def extract_all(audio_path: Path, out_dir: Path, cfg: FeatureConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y, _ = librosa.load(audio_path, sr=cfg.sr, mono=True)
    duration = librosa.get_duration(y=y, sr=cfg.sr)

    sections = load_sections(cfg.sections_dir, audio_path.stem)

    print(audio_path.name)
    print(cfg.sections_dir)

    mfcc, chroma, cens, frame_times = extract_mfcc_chroma_cens(y, cfg)
    orthogonal_features, orthogonal_diagnostics = extract_orthogonal_proxy_features(y, cfg, chroma, cens, frame_times)
    beat_boundaries, downbeat_boundaries = estimate_beats_and_downbeats(y, cfg)
    stm, stm_times, onset_env, stm_effective_window_s = extract_stm(y, cfg, beat_boundaries)

    beat_mfcc = aggregate_by_intervals(mfcc, frame_times, beat_boundaries)
    beat_chroma = aggregate_by_intervals(chroma, frame_times, beat_boundaries)
    beat_cens = aggregate_by_intervals(cens, frame_times, beat_boundaries)
    beat_stm = aggregate_by_intervals(stm, stm_times, beat_boundaries)
    beat_orthogonal = {
        name: aggregate_by_intervals(features, frame_times, beat_boundaries)
        for name, features in orthogonal_features.items()
    }

    ssm_map = {
        "stm": build_ssm(beat_stm),
        "mfcc": build_ssm(beat_mfcc),
        "chroma": build_ssm(beat_chroma),
        "cens": build_ssm(beat_cens),
    }
    for name, features in beat_orthogonal.items():
        ssm_map[name] = build_ssm(features)
    if sections:
        ssm_map["sections"] = build_section_ssm(sections, beat_boundaries)
    fused = np.mean([m for m in ssm_map.values() if m.size], axis=0) # TODO: CHANGE THIS HORRIBLE FUSION METHOD
    ssm_map["fused"] = fused.astype(np.float32)

    stem = audio_path.stem
    npz_path = out_dir / f"{stem}_features.npz"
    preview_paths = [out_dir / f"{stem}_preview.{fmt}" for fmt in cfg.preview_formats]
    summary_path = out_dir / f"{stem}_summary.json"

    np.savez_compressed(
        npz_path,
        audio_path=str(audio_path),
        sr=cfg.sr,
        duration=duration,
        frame_times=frame_times,
        stm=stm,
        stm_times=stm_times,
        stm_effective_window_s=stm_effective_window_s,
        onset_env=onset_env,
        mfcc=mfcc,
        chroma=chroma,
        cens=cens,
        arrangement=orthogonal_features["arrangement"],
        vocal=orthogonal_features["vocal"],
        bass=orthogonal_features["bass"],
        tonnetz=orthogonal_features["tonnetz"],
        density=orthogonal_features["density"],
        beat_boundaries=beat_boundaries,
        beat_downbeat_boundaries=downbeat_boundaries if downbeat_boundaries is not None else np.array([]),
        beat_stm=beat_stm,
        beat_mfcc=beat_mfcc,
        beat_chroma=beat_chroma,
        beat_cens=beat_cens,
        beat_arrangement=beat_orthogonal["arrangement"],
        beat_vocal=beat_orthogonal["vocal"],
        beat_bass=beat_orthogonal["bass"],
        beat_tonnetz=beat_orthogonal["tonnetz"],
        beat_density=beat_orthogonal["density"],
        ssm_stm=ssm_map["stm"],
        ssm_mfcc=ssm_map["mfcc"],
        ssm_chroma=ssm_map["chroma"],
        ssm_cens=ssm_map["cens"],
        ssm_arrangement=ssm_map["arrangement"],
        ssm_vocal=ssm_map["vocal"],
        ssm_bass=ssm_map["bass"],
        ssm_tonnetz=ssm_map["tonnetz"],
        ssm_density=ssm_map["density"],
        ssm_sections=ssm_map["sections"] if "sections" in ssm_map else np.zeros((0, 0), dtype=np.float32),
        ssm_fused=ssm_map["fused"],
    )

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
        "outputs": {
            "features_npz": str(npz_path),
            "preview_png": str(preview_paths[0]) if "png" in cfg.preview_formats else "",
            "preview_svg": str(out_dir / f"{stem}_preview.svg") if "svg" in cfg.preview_formats else "",
            "preview_files": [str(path) for path in preview_paths],
            "summary_json": str(summary_path),
        },
        "shapes": {
            "stm": list(stm.shape),
            "mfcc": list(mfcc.shape),
            "chroma": list(chroma.shape),
            "cens": list(cens.shape),
            "arrangement": list(orthogonal_features["arrangement"].shape),
            "vocal": list(orthogonal_features["vocal"].shape),
            "bass": list(orthogonal_features["bass"].shape),
            "tonnetz": list(orthogonal_features["tonnetz"].shape),
            "density": list(orthogonal_features["density"].shape),
            "beat_stm": list(beat_stm.shape),
            "beat_mfcc": list(beat_mfcc.shape),
            "beat_chroma": list(beat_chroma.shape),
            "beat_cens": list(beat_cens.shape),
            "beat_arrangement": list(beat_orthogonal["arrangement"].shape),
            "beat_vocal": list(beat_orthogonal["vocal"].shape),
            "beat_bass": list(beat_orthogonal["bass"].shape),
            "beat_tonnetz": list(beat_orthogonal["tonnetz"].shape),
            "beat_density": list(beat_orthogonal["density"].shape),
            "ssm_sections": list(ssm_map["sections"].shape) if "sections" in ssm_map else [0, 0],
            "ssm_fused": list(ssm_map["fused"].shape),
        },
        "diagnostics": {
            "beat_intervals": int(max(0, len(beat_boundaries) - 1)),
            "section_annotations_loaded": bool(sections),
        },
    }
    summary["diagnostics"].update(orthogonal_diagnostics)
    beat_intervals = np.diff(beat_boundaries)
    stm_beat_counts = count_beats_per_stm_window(
        stm_times,
        beat_boundaries,
        stm_effective_window_s,
        duration,
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path, help="Path to an audio file.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--demo", action="store_true", help="Generate and process a synthetic demo audio file.")
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument(
        "--beat-tracking-method",
        choices=["librosa", "beat_this", "madmom"],
        default=None,
    )
    parser.add_argument("--sections-dir", type=Path, default=None)
    parser.add_argument("--stm-coeffs", type=int, default=None)
    parser.add_argument("--stm-window-s", type=float, default=None)
    parser.add_argument("--stm-hop-s", type=float, default=None)
    parser.add_argument("--stm-min-beats", type=int, default=None)
    parser.add_argument(
        "--preview-features",
        nargs="+",
        default=None,
        help=f"Subset of preview SSMs to render. Options: {', '.join(PREVIEW_FEATURE_ORDER)}",
    )
    return parser.parse_args(argv)


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
