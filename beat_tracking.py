"""Beat and downbeat estimation helpers."""

from __future__ import annotations

import tempfile

from runtime_env import ensure_local_cache

ensure_local_cache()

import librosa
import numpy as np
import soundfile as sf

from data_loading import FeatureConfig

try:
    import madmom as mm  # noqa: F401
    from madmom.features import beats as bt
    from madmom.features import downbeats as dbt

    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False

try:
    from beat_this.inference import File2Beats

    BEAT_THIS_AVAILABLE = True
    BEAT_THIS_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    File2Beats = None
    BEAT_THIS_AVAILABLE = False
    BEAT_THIS_IMPORT_ERROR = exc


def safe_nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def regularize_beat_boundaries(boundaries: np.ndarray, duration: float) -> np.ndarray:
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


def estimate_beats_and_downbeats(y: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray | None]:
    method = cfg.beat_tracking_method.lower()
    duration = librosa.get_duration(y=y, sr=cfg.sr)

    if method == "librosa":
        _, beat_frames = librosa.beat.beat_track(y=y, sr=cfg.sr, hop_length=cfg.hop_length, trim=False)
        beat_times = librosa.frames_to_time(beat_frames, sr=cfg.sr, hop_length=cfg.hop_length)
        boundaries = np.unique(np.r_[0.0, beat_times, duration])
        if len(boundaries) < 4:
            boundaries = np.arange(0.0, duration + 1.0, 1.0)
            if boundaries[-1] < duration:
                boundaries = np.r_[boundaries, duration]
        boundaries = regularize_beat_boundaries(boundaries, duration)
        return boundaries.astype(np.float32), None

    if method == "madmom":
        if not MADMOM_AVAILABLE:
            raise ImportError("madmom not available")
        act_beats = bt.TCNBeatProcessor()(y)
        proc_beats = bt.BeatTrackingProcessor(fps=100)
        beat_times = proc_beats(act_beats)
        beat_boundaries = np.unique(np.r_[0.0, beat_times, duration])
        beat_boundaries = regularize_beat_boundaries(beat_boundaries, duration)
        act_downbeats = dbt.RNNDownBeatProcessor()(y)
        proc_downbeats = dbt.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        song_beats = proc_downbeats(act_downbeats)
        downbeat_times = [song_beats[0][0]]
        for beat in song_beats[1:]:
            if beat[1] == 1:
                downbeat_times.append(beat[0])
        downbeat_times.append(duration)
        downbeat_boundaries = np.unique(np.r_[0.0, downbeat_times, duration])
        downbeat_boundaries = regularize_beat_boundaries(downbeat_boundaries, duration)
        return beat_boundaries.astype(np.float32), downbeat_boundaries.astype(np.float32)

    if method == "beat_this":
        if not BEAT_THIS_AVAILABLE:
            raise ImportError(
                "beat_this is not available in this environment. "
                f"Original import error: {BEAT_THIS_IMPORT_ERROR!r}"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            sf.write(handle.name, y, cfg.sr)
            file2beats = File2Beats(
                checkpoint_path=cfg.beat_tracking_model_path or "final0",
                device="cpu",
                dbn=False,
            )
            beats, downbeats = file2beats(handle.name)
        beat_times = np.asarray(beats)
        downbeat_times = np.asarray(downbeats.tolist() + [beats[-1]])
        beat_boundaries = np.unique(np.r_[0.0, beat_times, duration])
        beat_boundaries = regularize_beat_boundaries(beat_boundaries, duration)
        downbeat_boundaries = np.unique(np.r_[0.0, downbeat_times, duration])
        downbeat_boundaries = regularize_beat_boundaries(downbeat_boundaries, duration)
        return beat_boundaries.astype(np.float32), downbeat_boundaries.astype(np.float32)

    raise ValueError(f"Unknown beat tracking method: {method}")
