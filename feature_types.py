"""Feature extraction and SSM construction helpers."""

from __future__ import annotations

from runtime_env import ensure_local_cache

ensure_local_cache()

import librosa
import numpy as np
from scipy.signal import detrend
from sklearn.metrics.pairwise import cosine_similarity

from data_loading import FeatureConfig


def safe_nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = safe_nan_to_num(x).astype(np.float32, copy=False)
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def build_ssm(features: np.ndarray) -> np.ndarray:
    if features.size == 0 or features.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x = zscore_columns(features).T
    return cosine_similarity(x).astype(np.float32)


def build_section_ssm(
    sections: list[tuple[float, float, str]],
    beat_boundaries: np.ndarray,
) -> np.ndarray:
    n_intervals = max(0, len(beat_boundaries) - 1)
    if not sections or n_intervals == 0:
        return np.zeros((0, 0), dtype=np.float32)

    unique_labels = sorted({label for _start, _end, label in sections})
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    interval_labels = np.full(n_intervals, -1, dtype=np.int32)
    interval_centers = (beat_boundaries[:-1] + beat_boundaries[1:]) / 2.0

    for start, end, label in sections:
        mask = (interval_centers >= start) & (interval_centers < end)
        interval_labels[mask] = label_to_id[label]

    valid = interval_labels[:, None] == interval_labels[None, :]
    assigned = (interval_labels[:, None] >= 0) & (interval_labels[None, :] >= 0)
    return np.where(assigned, valid, False).astype(np.float32)


def aggregate_by_intervals(features: np.ndarray, frame_times: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
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


def extract_mfcc_chroma_cens(y: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mfcc = librosa.feature.mfcc(y=y, sr=cfg.sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    chroma = librosa.feature.chroma_cqt(y=y, sr=cfg.sr, hop_length=cfg.hop_length)
    cens = librosa.feature.chroma_cens(y=y, sr=cfg.sr, hop_length=cfg.hop_length)
    frame_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=cfg.sr, hop_length=cfg.hop_length)
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
    onset_env = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(mag, ref=np.max),
        sr=cfg.sr,
        hop_length=cfg.hop_length,
    )
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

    features = {
        "arrangement": safe_nan_to_num(
            np.vstack(
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
        ).astype(np.float32),
        "vocal": safe_nan_to_num(
            np.vstack(
                [
                    vocal_band,
                    vocal_harmonic,
                    harmonic_ratio,
                    bands["presence"],
                    contrast_mean,
                    contrast_std,
                ]
            )
        ).astype(np.float32),
        "bass": safe_nan_to_num(
            np.vstack(
                [
                    bass_chroma,
                    sub_fraction,
                    bass_fraction,
                    bass_centroid,
                    low_flux,
                ]
            )
        ).astype(np.float32),
        "tonnetz": safe_nan_to_num(
            np.vstack(
                [
                    tonnetz,
                    chroma_delta,
                    tonnetz_delta,
                    harmonic_focus,
                ]
            )
        ).astype(np.float32),
        "density": safe_nan_to_num(
            np.vstack(
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
        ).astype(np.float32),
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

def extract_ssl_features(audio_path: str, cfg: FeatureConfig, feature_names: set[str]) -> dict[str, np.ndarray]:
    requested = feature_names & {"musicfm", "muq", "matpac"}
    if not requested:
        return {}

    try:
        import os
        import sys
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("Torch and torchaudio are required for SSL feature extraction.") from exc

    ssl_root = cfg.ssl_models_root
    if str(ssl_root / "matpac" / "inference_matpac") not in sys.path:
        sys.path.insert(0, str(ssl_root / "matpac" / "inference_matpac"))
    if str(ssl_root / "MuQ" / "src") not in sys.path:
        sys.path.insert(0, str(ssl_root / "MuQ" / "src"))
    if str(ssl_root.parent) not in sys.path:
        sys.path.insert(0, str(ssl_root.parent))

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    outputs: dict[str, np.ndarray] = {}
    device = torch.device(cfg.ssl_device)

    def _resample_if_needed(target_sr: int) -> torch.Tensor:
        if sample_rate == target_sr:
            return waveform
        return torchaudio.functional.resample(waveform, sample_rate, target_sr)

    with torch.no_grad():
        if "musicfm" in requested:
            sys.path.insert(0, str(cfg.ssl_models_root))
            from musicfm.model.musicfm_25hz import MusicFM25Hz

            wav_24k = _resample_if_needed(24000).to(device)
            musicfm = MusicFM25Hz(
                is_flash=False,
                stat_path=os.fspath(cfg.musicfm_stat_path),
                model_path=os.fspath(cfg.musicfm_model_path),
            ).to(device)
            musicfm.eval()
            emb = musicfm.get_latent(wav_24k, layer_ix=7)
            outputs["musicfm"] = emb.squeeze(0).detach().cpu().numpy().T.astype(np.float32)

        if "muq" in requested:
            from muq import MuQ

            wav_24k = _resample_if_needed(24000).to(device)
            muq = MuQ.from_pretrained(cfg.muq_model_name).to(device).eval()
            emb = muq(wav_24k, output_hidden_states=True)
            outputs["muq"] = emb.last_hidden_state.squeeze(0).detach().cpu().numpy().T.astype(np.float32)

        if "matpac" in requested:
            from matpac.model import get_matpac

            wav_16k = _resample_if_needed(16000).to(device)
            matpac = get_matpac(
                checkpoint_path=os.fspath(cfg.matpac_checkpoint_path),
                pull_time_dimension=False,
                inference_type="fast",
            ).to(device).eval()
            emb, _layer_results = matpac(wav_16k)
            outputs["matpac"] = emb.squeeze(0).detach().cpu().numpy().T.astype(np.float32)

    return outputs

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

    f0_features = np.vstack([smooth_hz, smooth_midi, voiced_flag, confidence]).astype(np.float32)
    f0_contour_features = np.vstack([smooth_midi, midi_delta, confidence]).astype(np.float32)
    f0_times = librosa.frames_to_time(np.arange(f0_features.shape[1]), sr=cfg.sr, hop_length=cfg.hop_length)
    return f0_features, confidence, f0_times, f0_contour_features


def autocorrelation_positive(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2 :]
    if ac[0] > 0:
        ac = ac / ac[0]
    return ac


def direct_scale_transform_magnitude(autocorr: np.ndarray, lag_step_s: float, n_coeffs: int) -> np.ndarray:
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
    onset_env = librosa.onset.onset_strength(S=log_mel, sr=cfg.sr, hop_length=cfg.hop_length, aggregate=np.mean)
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
