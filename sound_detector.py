#!/usr/bin/env python3
"""Enroll and detect loud, repeated transient sounds.

This program deliberately uses signal processing instead of a pretrained model.
An enrollment profile is made from the user's own recordings and contains a
small spectral/temporal fingerprint for the sound to detect.

Typical workflow:

    python sound_detector.py enroll samples/hit-*.wav --profile hit.json
    python sound_detector.py detect rehearsal.wav --profile hit.json
    python sound_detector.py listen --profile hit.json

The microphone mode requires PortAudio (installed by the ``sounddevice``
package). WAV/FLAC/OGG and other formats supported by ``soundfile`` can be
used as input files.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CLIP_SECONDS = 0.07
DEFAULT_PRE_SECONDS = 0.008
DEFAULT_THRESHOLD = 0.60
DEFAULT_JUMP_DB = 4.0
DEFAULT_MIN_DBFS = -40.0
DEFAULT_RISE_DB = 3.5
DEFAULT_POST_HIT_RISE_DB = 5.0
DEFAULT_MIN_GAP_MS = 50.0
DEFAULT_BLOCK_MS = 8.0
DEFAULT_DISPLAY_DBFS = 0.0
LIVE_ALIGNMENT_SECONDS = 0.025
LIVE_WARMUP_SECONDS = 0.50
POST_HIT_WINDOW_SECONDS = 0.20
PROFILE_VERSION = 2


def dbfs(rms: float) -> float:
    """Return RMS level in dBFS, with a useful floor for silence."""

    return 20.0 * math.log10(max(float(rms), 1e-7))


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio using linear interpolation.

    The detector uses short transients, where this dependency-free resampler
    is sufficient and keeps the installation small.
    """

    if source_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    new_length = max(1, int(round(len(audio) * target_rate / source_rate)))
    old_positions = np.arange(len(audio), dtype=np.float64)
    new_positions = np.linspace(0.0, len(audio) - 1, new_length)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    """Load an audio file as mono float32 audio at ``target_rate``."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Reading audio files requires soundfile. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise RuntimeError(f"Expected mono or multi-channel audio in {path}")
    if len(audio) == 0:
        raise RuntimeError(f"Audio file is empty: {path}")

    audio = audio - np.mean(audio, dtype=np.float32)
    return resample(audio, int(sample_rate), target_rate)


def frame_rms(audio: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    """Calculate RMS values for overlapping frames."""

    frame_samples = max(1, int(frame_samples))
    hop_samples = max(1, int(hop_samples))
    if len(audio) < frame_samples:
        padded = np.pad(audio, (0, frame_samples - len(audio)))
    else:
        padded = audio

    count = 1 + max(0, (len(padded) - frame_samples) // hop_samples)
    starts = np.arange(count, dtype=np.int64) * hop_samples
    required = int(starts[-1] + frame_samples)
    if required > len(padded):
        padded = np.pad(padded, (0, required - len(padded)))
    indices = starts[:, None] + np.arange(frame_samples, dtype=np.int64)[None, :]
    frames = padded[indices]
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def extract_transient(
    audio: np.ndarray,
    sample_rate: int,
    clip_seconds: float,
    pre_seconds: float,
) -> tuple[np.ndarray, int]:
    """Return a fixed-size clip around the loudest transient in a file."""

    frame_samples = max(1, round(sample_rate * 0.025))
    hop_samples = max(1, round(sample_rate * 0.010))
    levels = frame_rms(audio, frame_samples, hop_samples)
    peak_frame = int(np.argmax(levels))
    peak_sample = peak_frame * hop_samples + frame_samples // 2

    clip_samples = max(1, round(sample_rate * clip_seconds))
    start = peak_sample - round(sample_rate * pre_seconds)
    end = start + clip_samples
    source_start = max(0, start)
    source_end = min(len(audio), end)
    segment = audio[source_start:source_end]
    left_pad = max(0, -start)
    right_pad = max(0, end - len(audio))
    if left_pad or right_pad:
        segment = np.pad(segment, (left_pad, right_pad))
    if len(segment) < clip_samples:
        segment = np.pad(segment, (0, clip_samples - len(segment)))
    return segment[:clip_samples].astype(np.float32, copy=False), peak_sample


def _log_band_shapes(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Return a time-varying, relative spectrum in log-spaced bands."""

    low = 60.0
    high = min(8_000.0, float(frequencies[-1]))
    edges = np.geomspace(low, max(low * 1.01, high), 25)
    energies = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        band = power[:, (frequencies >= lower) & (frequencies < upper)]
        if band.shape[1]:
            energies.append(np.mean(band, axis=1))
        else:
            energies.append(np.full(power.shape[0], 1e-12, dtype=np.float32))
    matrix = np.stack(energies, axis=1)
    log_energies = np.log10(np.maximum(matrix, 1e-12))
    relative = log_energies - np.max(log_energies, axis=1, keepdims=True)
    # Preserve where the energy is concentrated while allowing modest changes
    # in pitch and microphone response. Each frame is in roughly [-1, 0].
    return np.clip(relative / 4.0, -1.0, 0.0).astype(np.float32)


def feature_vector(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """Create a volume-normalized fingerprint for one transient.

    The first part describes spectral shape. The second part describes the
    short attack/decay envelope. Both are normalized, so classification is
    mostly independent of how hard the sound was played. Loudness is checked
    separately by the detector.
    """

    x = np.asarray(segment, dtype=np.float32)
    x = x - np.mean(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1e-6:
        x = x / peak

    frame_samples = max(16, round(sample_rate * 0.025))
    hop_samples = max(1, round(sample_rate * 0.010))
    if len(x) < frame_samples:
        x = np.pad(x, (0, frame_samples - len(x)))
    starts = np.arange(0, max(1, len(x) - frame_samples + 1), hop_samples)
    if len(starts) == 0:
        starts = np.array([0])
    required = int(starts[-1] + frame_samples)
    if required > len(x):
        x = np.pad(x, (0, required - len(x)))
    indices = starts[:, None] + np.arange(frame_samples, dtype=np.int64)[None, :]
    frames = x[indices]
    window = np.hanning(frame_samples).astype(np.float32)
    spectra = np.abs(np.fft.rfft(frames * window[None, :], axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frame_samples, d=1.0 / sample_rate)
    spectral = _log_band_shapes(spectra, frequencies)

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    envelope = np.log10(np.maximum(rms, 1e-6))
    envelope -= np.max(envelope)
    envelope = np.clip(envelope, -6.0, 0.0) / 6.0
    target_points = 16
    old_axis = np.linspace(0.0, 1.0, len(spectral))
    new_axis = np.linspace(0.0, 1.0, target_points)
    spectral = np.stack(
        [np.interp(new_axis, old_axis, spectral[:, band]) for band in range(spectral.shape[1])],
        axis=1,
    )
    if len(envelope) == 1:
        envelope = np.repeat(envelope, target_points)
    else:
        envelope = np.interp(new_axis, np.linspace(0.0, 1.0, len(envelope)), envelope)

    # Weight the attack/decay shape enough to matter alongside the time-varying
    # spectrum. Do not L2-normalize: absolute feature distance is used for a
    # stricter nearest-template comparison.
    result = np.concatenate([spectral.ravel(), (3.0 * envelope).astype(np.float32)])
    return result.astype(np.float32)


@dataclass
class Profile:
    sample_rate: int
    clip_seconds: float
    pre_seconds: float
    templates: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    negative_templates: np.ndarray

    def to_dict(self) -> dict:
        return {
            "version": PROFILE_VERSION,
            "sample_rate": self.sample_rate,
            "clip_seconds": self.clip_seconds,
            "pre_seconds": self.pre_seconds,
            "templates": self.templates.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "negative_templates": self.negative_templates.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        if data.get("version") != PROFILE_VERSION:
            raise ValueError("Unsupported or missing profile version")
        templates = np.asarray(data["templates"], dtype=np.float32)
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        negative_data = data.get("negative_templates", [])
        negative_templates = (
            np.asarray(negative_data, dtype=np.float32)
            if negative_data
            else np.empty((0, len(mean)), dtype=np.float32)
        )
        if templates.ndim != 2 or templates.shape[0] == 0 or mean.ndim != 1 or std.ndim != 1:
            raise ValueError("Profile feature arrays have invalid shapes")
        if templates.shape[1] != len(mean) or len(mean) != len(std):
            raise ValueError("Profile feature dimensions do not match")
        if negative_templates.ndim != 2 or negative_templates.shape[1] != len(mean):
            raise ValueError("Negative profile feature dimensions do not match")
        return cls(
            sample_rate=int(data["sample_rate"]),
            clip_seconds=float(data["clip_seconds"]),
            pre_seconds=float(data["pre_seconds"]),
            templates=templates,
            mean=mean,
            std=std,
            negative_templates=negative_templates,
        )


def save_profile(profile: Profile, path: Path) -> None:
    path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_profile(path: Path) -> Profile:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Profile.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not load profile {path}: {exc}") from exc


def enroll(
    files: Iterable[Path],
    profile_path: Path,
    sample_rate: int,
    clip_seconds: float,
    pre_seconds: float,
    negative_files: Iterable[Path] = (),
) -> None:
    """Build a profile from positive and optional negative recordings."""

    templates = []
    paths = list(files)
    if not paths:
        raise RuntimeError("At least one example recording is required")

    for path in paths:
        audio = load_audio(path, sample_rate)
        segment, peak_sample = extract_transient(audio, sample_rate, clip_seconds, pre_seconds)
        templates.append(feature_vector(segment, sample_rate))
        print(
            f"Enrolled {path} (loudest transient at {peak_sample / sample_rate:.3f}s, "
            f"level {dbfs(float(np.sqrt(np.mean(segment * segment)))):.1f} dBFS)"
        )

    matrix = np.vstack(templates).astype(np.float32)
    mean = np.mean(matrix, axis=0).astype(np.float32)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm > 1e-8:
        mean /= mean_norm
    std = np.std(matrix, axis=0).astype(np.float32)
    negative_templates = []
    for path in negative_files:
        audio = load_audio(path, sample_rate)
        segment, peak_sample = extract_transient(audio, sample_rate, clip_seconds, pre_seconds)
        negative_templates.append(feature_vector(segment, sample_rate))
        print(
            f"Enrolled rejection example {path} (loudest transient at "
            f"{peak_sample / sample_rate:.3f}s)"
        )
    negative_matrix = (
        np.vstack(negative_templates).astype(np.float32)
        if negative_templates
        else np.empty((0, matrix.shape[1]), dtype=np.float32)
    )
    profile = Profile(sample_rate, clip_seconds, pre_seconds, matrix, mean, std, negative_matrix)
    save_profile(profile, profile_path)
    negative_note = f" and {len(negative_templates)} rejection example(s)" if negative_templates else ""
    print(f"Saved profile with {len(paths)} example(s){negative_note} to {profile_path}")


def add_positive_template(profile: Profile, template: np.ndarray) -> None:
    """Append a positive example and refresh the stored profile statistics."""

    matrix = np.vstack([profile.templates, template]).astype(np.float32)
    mean = np.mean(matrix, axis=0).astype(np.float32)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm > 1e-8:
        mean /= mean_norm
    profile.templates = matrix
    profile.mean = mean
    profile.std = np.std(matrix, axis=0).astype(np.float32)


def add_negative_templates(profile: Profile, templates: np.ndarray, replace: bool = False) -> None:
    """Add ambient/rejection fingerprints while keeping profile size bounded."""

    templates = np.asarray(templates, dtype=np.float32)
    if templates.ndim != 2 or templates.shape[1] != profile.templates.shape[1]:
        raise RuntimeError("Background feature dimensions do not match the profile")
    if replace or not len(profile.negative_templates):
        combined = templates
    else:
        combined = np.vstack([profile.negative_templates, templates]).astype(np.float32)
    profile.negative_templates = combined[-128:].astype(np.float32, copy=False)


def extract_background_templates(audio: np.ndarray, profile: Profile, max_examples: int) -> np.ndarray:
    """Extract the loudest short windows from an ambient recording."""

    if max_examples <= 0:
        raise RuntimeError("--examples must be greater than zero")
    clip_samples = max(1, round(profile.sample_rate * profile.clip_seconds))
    if len(audio) < clip_samples:
        audio = np.pad(audio, (0, clip_samples - len(audio)))
    hop_samples = max(1, clip_samples // 2)
    starts = np.arange(0, max(1, len(audio) - clip_samples + 1), hop_samples, dtype=np.int64)
    segments = [audio[start : start + clip_samples] for start in starts]
    levels = np.array([float(np.sqrt(np.mean(segment * segment) + 1e-12)) for segment in segments])
    selected = np.argsort(levels)[-min(max_examples, len(segments)) :]
    return np.vstack(
        [feature_vector(np.asarray(segments[index], dtype=np.float32), profile.sample_rate) for index in selected]
    ).astype(np.float32)


def calibrate_microphone(
    profile_path: Path,
    profile: Profile,
    count: int,
    record_seconds: float,
    device: Optional[str | int],
) -> None:
    """Interactively add target examples recorded through the live microphone."""

    if count <= 0:
        raise RuntimeError("--count must be greater than zero")
    if record_seconds <= 0:
        raise RuntimeError("--record-seconds must be greater than zero")
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Microphone calibration requires sounddevice. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    print(
        "Microphone calibration adds real acoustic examples to the profile. "
        "Keep the room otherwise quiet and make only the target hit when prompted."
    )
    added = 0
    try:
        while added < count:
            input(f"Example {added + 1}/{count}: press Enter, then make the hit after 'HIT NOW' appears...")
            frames = round(profile.sample_rate * record_seconds)
            print("Get ready...", flush=True)
            time.sleep(0.25)
            recording = sd.rec(
                frames,
                samplerate=profile.sample_rate,
                channels=1,
                dtype="float32",
                device=device,
            )
            print("HIT NOW", flush=True)
            sd.wait()
            audio = np.asarray(recording[:, 0], dtype=np.float32)
            levels = np.array(
                [
                    dbfs(value)
                    for value in frame_rms(
                        audio,
                        round(profile.sample_rate * 0.025),
                        round(profile.sample_rate * 0.010),
                    )
                ]
            )
            peak_level = float(np.max(levels))
            floor_level = float(np.percentile(levels, 20))
            if peak_level < -85.0 or peak_level - floor_level < 1.0:
                print(
                    f"No clear hit found (peak={peak_level:.1f} dBFS, "
                    f"rise={peak_level - floor_level:.1f} dB). Please retry."
                )
                continue
            segment, peak_sample = extract_transient(
                audio,
                profile.sample_rate,
                profile.clip_seconds,
                profile.pre_seconds,
            )
            add_positive_template(profile, feature_vector(segment, profile.sample_rate))
            save_profile(profile, profile_path)
            added += 1
            print(
                f"Added example {added}/{count}: peak at {peak_sample / profile.sample_rate:.3f}s, "
                f"level={peak_level:.1f} dBFS, rise={peak_level - floor_level:.1f} dB"
            )
    except KeyboardInterrupt:
        print(f"\nCalibration stopped after {added} example(s).")
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Could not record microphone calibration: {exc}") from exc


def calibrate_background(
    profile_path: Path,
    profile: Profile,
    seconds: float,
    examples: int,
    replace: bool,
    device: Optional[str | int],
) -> None:
    """Record ordinary room sound and store it as negative examples."""

    if seconds <= 0:
        raise RuntimeError("--seconds must be greater than zero")
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Background calibration requires sounddevice. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    print(
        "Background calibration records the normal room. Do not make the target hit "
        "during this recording."
    )
    try:
        input("Press Enter to begin...")
        print("Get ready...", flush=True)
        time.sleep(0.25)
        frames = round(profile.sample_rate * seconds)
        recording = sd.rec(
            frames,
            samplerate=profile.sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        print(f"RECORDING BACKGROUND for {seconds:.1f} seconds...", flush=True)
        sd.wait()
        audio = np.asarray(recording[:, 0], dtype=np.float32)
        templates = extract_background_templates(audio, profile, examples)
        add_negative_templates(profile, templates, replace=replace)
        save_profile(profile, profile_path)
        level = dbfs(float(np.sqrt(np.mean(audio * audio) + 1e-12)))
        print(
            f"Saved {len(templates)} background example(s) to {profile_path}; "
            f"ambient level={level:.1f} dBFS."
        )
    except KeyboardInterrupt:
        print("\nBackground calibration cancelled.")
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Could not record background calibration: {exc}") from exc


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(first, second) / denominator)


def feature_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Mean absolute distance; unlike cosine, it has a meaningful zero."""

    return float(np.mean(np.abs(first - second)))


def classify(segment: np.ndarray, profile: Profile) -> float:
    """Return a 0..1 similarity score against an enrollment profile.

    The positive score is based on the nearest enrolled example. If rejection
    examples exist, the score is also required to be closer to the positive
    class than to the rejection class.
    """

    vector = feature_vector(segment, profile.sample_rate)
    positive_distance = min(feature_distance(vector, template) for template in profile.templates)
    # A distance of zero is an exact fingerprint match. The scale is expressed
    # in feature units and intentionally makes the no-negative-example mode
    # conservative.
    positive_score = math.exp(-positive_distance / 0.28)
    if len(profile.negative_templates):
        negative_distance = min(feature_distance(vector, template) for template in profile.negative_templates)
        margin = negative_distance - positive_distance
        contrast_score = 1.0 / (1.0 + math.exp(-margin / 0.035))
        return float(np.clip(0.60 * positive_score + 0.40 * contrast_score, 0.0, 1.0))
    return float(np.clip(positive_score, 0.0, 1.0))


def format_time(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"
    return f"{int(minutes):02d}:{remainder:05.2f}"


def detect_file(
    audio_path: Path,
    profile: Profile,
    threshold: float,
    jump_db: float,
    min_dbfs: float,
    rise_db: float,
    post_hit_rise_db: float,
    min_gap_ms: float,
    verbose: bool,
) -> int:
    """Scan an audio file and print each matching transient."""

    audio = load_audio(audio_path, profile.sample_rate)
    frame_samples = max(1, round(profile.sample_rate * 0.025))
    hop_samples = max(1, round(profile.sample_rate * 0.010))
    levels = np.array([dbfs(value) for value in frame_rms(audio, frame_samples, hop_samples)])
    noise_floor = float(np.percentile(levels, 20))
    min_gap_samples = round(profile.sample_rate * min_gap_ms / 1000.0)
    last_hit = -min_gap_samples
    pre_samples = round(profile.sample_rate * profile.pre_seconds)
    clip_samples = round(profile.sample_rate * profile.clip_seconds)
    detections = 0

    for frame_number, level in enumerate(levels):
        sample = frame_number * hop_samples + frame_samples // 2
        if sample - last_hit < min_gap_samples:
            continue
        history_start = max(0, frame_number - 6)
        previous_levels = levels[history_start:frame_number]
        recent_floor = float(np.min(previous_levels)) if len(previous_levels) else noise_floor
        measured_rise = level - recent_floor
        within_post_hit_window = (
            last_hit >= 0
            and sample - last_hit < round(profile.sample_rate * POST_HIT_WINDOW_SECONDS)
        )
        required_rise = max(rise_db, post_hit_rise_db) if within_post_hit_window else rise_db
        fresh_attack = measured_rise >= required_rise
        local_end = min(len(levels), frame_number + 3)
        is_local_peak = level >= float(np.max(levels[frame_number:local_end]))
        if not fresh_attack or not is_local_peak or level < min_dbfs or level < noise_floor + jump_db:
            continue

        start = sample - pre_samples
        end = start + clip_samples
        source_start = max(0, start)
        source_end = min(len(audio), end)
        segment = audio[source_start:source_end]
        segment = np.pad(segment, (max(0, -start), max(0, end - len(audio))))
        segment = segment[:clip_samples]
        score = classify(segment, profile)
        last_hit = sample
        if score >= threshold:
            detections += 1
            print(f"[{format_time(sample / profile.sample_rate)}] DETECTED score={score:.2f} level={level:.1f} dBFS")
        elif verbose:
            print(f"[{format_time(sample / profile.sample_rate)}] loud candidate rejected score={score:.2f}")

    if detections == 0:
        print("No matching hits detected.")
    else:
        print(f"Detected {detections} matching hit(s).")
    return detections


@dataclass
class PendingCandidate:
    audio: np.ndarray
    context_start_sample: int
    trigger_sample: int
    level_db: float
    noise_floor_db: float
    rise_db: float


class LiveDetector:
    """Incremental detector used by the microphone listener."""

    def __init__(
        self,
        profile: Profile,
        threshold: float,
        jump_db: float,
        min_dbfs: float,
        verbose: bool,
        rise_db: float = DEFAULT_RISE_DB,
        post_hit_rise_db: float = DEFAULT_POST_HIT_RISE_DB,
        min_gap_ms: float = DEFAULT_MIN_GAP_MS,
    ):
        self.profile = profile
        self.threshold = threshold
        self.jump_db = jump_db
        self.min_dbfs = min_dbfs
        self.rise_db = rise_db
        self.post_hit_rise_db = post_hit_rise_db
        self.verbose = verbose
        self.pre_samples = round(profile.sample_rate * profile.pre_seconds)
        self.clip_samples = round(profile.sample_rate * profile.clip_seconds)
        self.alignment_samples = round(profile.sample_rate * LIVE_ALIGNMENT_SECONDS)
        self.context_pre_samples = self.pre_samples + self.alignment_samples
        self.context_samples = self.clip_samples + 2 * self.alignment_samples
        self.ring: list[float] = []
        self.pending_candidates: list[PendingCandidate] = []
        self.total_samples = 0
        self.noise_floor_db: Optional[float] = None
        self.min_gap_samples = round(profile.sample_rate * min_gap_ms / 1000.0)
        self.post_hit_window_samples = round(profile.sample_rate * POST_HIT_WINDOW_SECONDS)
        self.warmup_samples = round(profile.sample_rate * LIVE_WARMUP_SECONDS)
        self.last_trigger_sample = -self.min_gap_samples
        self.recent_detection_peaks: deque[int] = deque()
        self.recent_levels: deque[tuple[int, float]] = deque()
        self.noise_levels: deque[tuple[int, float]] = deque()

    def _ring_tail(self, count: int) -> np.ndarray:
        return np.asarray(self.ring[-count:], dtype=np.float32)

    def process(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if not len(block):
            return
        block_start = self.total_samples
        self.total_samples += len(block)
        self.ring.extend(block.tolist())
        max_ring = max(self.context_samples * 2, self.context_pre_samples + len(block) * 2)
        if len(self.ring) > max_ring:
            del self.ring[:-max_ring]

        block_level = dbfs(float(np.sqrt(np.mean(block * block) + 1e-12)))
        prior_noise_levels = [level for _, level in self.noise_levels]
        self.noise_floor_db = (
            float(np.percentile(prior_noise_levels, 35))
            if prior_noise_levels
            else block_level
        )
        prior_recent_levels = [level for _, level in self.recent_levels]
        recent_floor = min(prior_recent_levels) if prior_recent_levels else block_level

        still_pending: list[PendingCandidate] = []
        for candidate in self.pending_candidates:
            candidate.audio = np.concatenate([candidate.audio, block])
            if len(candidate.audio) >= self.context_samples:
                context = candidate.audio[: self.context_samples]
                segment, peak_in_context = extract_transient(
                    context,
                    self.profile.sample_rate,
                    self.profile.clip_seconds,
                    self.profile.pre_seconds,
                )
                event_sample = candidate.context_start_sample + peak_in_context
                event_seconds = event_sample / self.profile.sample_rate
                score = classify(segment, self.profile)
                if score >= self.threshold:
                    duplicate_window = round(self.profile.sample_rate * 0.040)
                    duplicate = any(
                        abs(event_sample - previous_peak) <= duplicate_window
                        for previous_peak in self.recent_detection_peaks
                    )
                    if duplicate:
                        if self.verbose:
                            print(
                                f"[{format_time(event_seconds)}] duplicate candidate suppressed "
                                f"score={score:.2f}",
                                flush=True,
                            )
                    else:
                        self.recent_detection_peaks.append(event_sample)
                        print(
                            f"[{format_time(event_seconds)}] DETECTED score={score:.2f} "
                            f"level={candidate.level_db:.1f} dBFS "
                            f"floor={candidate.noise_floor_db:.1f} rise={candidate.rise_db:.1f} dB",
                            flush=True,
                        )
                elif self.verbose:
                    print(
                        f"[{format_time(event_seconds)}] candidate rejected score={score:.2f} "
                        f"level={candidate.level_db:.1f} dBFS "
                        f"floor={candidate.noise_floor_db:.1f} rise={candidate.rise_db:.1f} dB",
                        flush=True,
                    )
            else:
                still_pending.append(candidate)
        self.pending_candidates = still_pending

        assert self.noise_floor_db is not None
        enough_volume = block_level >= self.min_dbfs
        above_noise = block_level >= self.noise_floor_db + self.jump_db
        measured_rise = block_level - recent_floor
        trigger_sample = block_start + len(block) // 2
        within_post_hit_window = (
            self.last_trigger_sample >= 0
            and trigger_sample - self.last_trigger_sample < self.post_hit_window_samples
        )
        required_rise = (
            max(self.rise_db, self.post_hit_rise_db)
            if within_post_hit_window
            else self.rise_db
        )
        fresh_attack = measured_rise >= required_rise
        cooldown_over = trigger_sample - self.last_trigger_sample >= self.min_gap_samples
        warmup_complete = self.total_samples >= self.warmup_samples
        if warmup_complete and enough_volume and above_noise and fresh_attack and cooldown_over:
            self.last_trigger_sample = trigger_sample
            initial_audio = self._ring_tail(self.context_pre_samples)
            self.pending_candidates.append(
                PendingCandidate(
                    audio=initial_audio,
                    context_start_sample=self.total_samples - len(initial_audio),
                    trigger_sample=trigger_sample,
                    level_db=block_level,
                    noise_floor_db=self.noise_floor_db,
                    rise_db=measured_rise,
                )
            )

        self.recent_levels.append((block_start, block_level))
        recent_cutoff = block_start - round(self.profile.sample_rate * 0.050)
        while self.recent_levels and self.recent_levels[0][0] < recent_cutoff:
            self.recent_levels.popleft()
        self.noise_levels.append((block_start, block_level))
        noise_cutoff = block_start - round(self.profile.sample_rate * 2.0)
        while self.noise_levels and self.noise_levels[0][0] < noise_cutoff:
            self.noise_levels.popleft()
        detection_cutoff = block_start - round(self.profile.sample_rate * 0.5)
        while self.recent_detection_peaks and self.recent_detection_peaks[0] < detection_cutoff:
            self.recent_detection_peaks.popleft()


def downsample_waveform(audio: np.ndarray, point_count: int) -> np.ndarray:
    """Reduce audio to display points while retaining short peaks."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    point_count = max(2, int(point_count))
    if len(audio) <= point_count:
        return audio
    edges = np.linspace(0, len(audio), point_count + 1, dtype=np.int64)
    result = np.empty(point_count, dtype=np.float32)
    for index in range(point_count):
        bucket = audio[edges[index] : edges[index + 1]]
        result[index] = bucket[int(np.argmax(np.abs(bucket)))] if len(bucket) else 0.0
    return result


class WaveformWindow:
    """Small Tk window that visualizes the microphone stream."""

    def __init__(
        self,
        sample_rate: int,
        seconds: float = 2.0,
        display_dbfs: float = DEFAULT_DISPLAY_DBFS,
        initial_min_dbfs: float = DEFAULT_MIN_DBFS,
    ):
        try:
            import tkinter as tk
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError("Tkinter is unavailable") from exc

        self.tk = tk
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover - display dependent
            raise RuntimeError(f"Could not open the waveform window: {exc}") from exc
        self.root.title("Live Sound Detector — Microphone Waveform")
        self.root.geometry("920x390")
        self.root.minsize(620, 280)
        self.root.configure(bg="#111820")
        if not math.isfinite(display_dbfs) or display_dbfs > 0.0:
            self.root.destroy()
            raise RuntimeError("--display-dbfs must be a finite value at or below 0")
        self.running = True
        self.display_dbfs = float(display_dbfs)
        self.display_scale = 10.0 ** (self.display_dbfs / 20.0)
        self.capacity = max(1, round(sample_rate * seconds))
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_index = 0
        self.filled = 0
        self.latest_level_db = -140.0
        self.latest_peak_db = -140.0

        self.title_label = tk.Label(
            self.root,
            text=(
                f"Live microphone — last {seconds:.1f} seconds — "
                f"fixed full-scale {self.display_dbfs:.1f} dBFS"
            ),
            bg="#111820",
            fg="#e6edf3",
            font=("TkDefaultFont", 13, "bold"),
            anchor="w",
            padx=12,
            pady=8,
        )
        self.title_label.pack(fill="x")
        self.canvas = tk.Canvas(
            self.root,
            bg="#081018",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.canvas.pack(fill="both", expand=True, padx=12)
        self.gate_frame = tk.Frame(self.root, bg="#111820")
        self.gate_frame.pack(fill="x", padx=12, pady=(8, 0))
        self.gate_label = tk.Label(
            self.gate_frame,
            text="Minimum sound level (silence gate)",
            bg="#111820",
            fg="#b8c4d4",
            anchor="w",
        )
        self.gate_label.pack(side="left")
        self.gate_var = tk.DoubleVar(value=float(initial_min_dbfs))
        self.gate_slider = tk.Scale(
            self.gate_frame,
            from_=-90,
            to=0,
            resolution=1,
            orient="horizontal",
            variable=self.gate_var,
            showvalue=True,
            length=420,
            bg="#111820",
            fg="#e6edf3",
            troughcolor="#334155",
            activebackground="#34d399",
            highlightthickness=0,
            label="dBFS",
        )
        self.gate_slider.pack(side="right", fill="x", expand=True, padx=(16, 0))
        self.status_var = tk.StringVar(value="Waiting for microphone audio...")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#111820",
            fg="#b8c4d4",
            anchor="w",
            padx=12,
            pady=10,
            font=("TkFixedFont", 10),
        )
        self.status_label.pack(fill="x")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self) -> None:
        if not self.running:
            return
        self.running = False
        try:
            self.root.quit()
        except self.tk.TclError:
            pass

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def append(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if not len(block):
            return
        self.latest_level_db = dbfs(float(np.sqrt(np.mean(block * block) + 1e-12)))
        self.latest_peak_db = 20.0 * math.log10(max(float(np.max(np.abs(block))), 1e-7))
        if len(block) >= self.capacity:
            self.buffer[:] = block[-self.capacity :]
            self.write_index = 0
            self.filled = self.capacity
            return
        first = min(len(block), self.capacity - self.write_index)
        self.buffer[self.write_index : self.write_index + first] = block[:first]
        remainder = len(block) - first
        if remainder:
            self.buffer[:remainder] = block[first:]
        self.write_index = (self.write_index + len(block)) % self.capacity
        self.filled = min(self.capacity, self.filled + len(block))

    def samples(self) -> np.ndarray:
        if self.filled < self.capacity:
            return self.buffer[: self.filled].copy()
        return np.concatenate([self.buffer[self.write_index :], self.buffer[: self.write_index]])

    def minimum_dbfs(self) -> float:
        return float(self.gate_var.get())

    def redraw(self, detector: LiveDetector) -> None:
        if not self.running:
            return
        width = max(2, self.canvas.winfo_width())
        height = max(2, self.canvas.winfo_height())
        center = height / 2.0
        self.canvas.delete("all")
        grid_color = "#203040"
        self.canvas.create_line(0, center, width, center, fill="#475569")
        self.canvas.create_line(0, height * 0.25, width, height * 0.25, fill=grid_color)
        self.canvas.create_line(0, height * 0.75, width, height * 0.75, fill=grid_color)
        for fraction in (0.25, 0.5, 0.75):
            x = width * fraction
            self.canvas.create_line(x, 0, x, height, fill=grid_color)

        audio = self.samples()
        if len(audio):
            displayed = downsample_waveform(audio, min(width, 1200))
            x_values = np.linspace(0.0, width - 1.0, len(displayed))
            y_values = center - np.clip(displayed / self.display_scale, -1.0, 1.0) * height * 0.44
            points = np.column_stack([x_values, y_values]).ravel().tolist()
            if len(points) >= 4:
                self.canvas.create_line(*points, fill="#34d399", width=1.5)

        floor = detector.noise_floor_db if detector.noise_floor_db is not None else -140.0
        gate = max(detector.min_dbfs, floor + detector.jump_db)
        self.status_var.set(
            f"level {self.latest_level_db:6.1f} dBFS   "
            f"peak {self.latest_peak_db:6.1f} dBFS   "
            f"noise floor {floor:6.1f} dBFS   "
            f"trigger gate {gate:6.1f} dBFS   "
            f"fixed full-scale {self.display_dbfs:.1f} dBFS"
        )


def listen(
    profile: Profile,
    threshold: float,
    jump_db: float,
    min_dbfs: float,
    rise_db: float,
    post_hit_rise_db: float,
    min_gap_ms: float,
    verbose: bool,
    block_ms: float,
    device: Optional[str | int],
    show_window: bool = True,
    display_dbfs: float = DEFAULT_DISPLAY_DBFS,
) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Microphone mode requires sounddevice. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    audio_queue: queue.Queue[object] = queue.Queue(maxsize=32)
    detector = LiveDetector(
        profile,
        threshold,
        jump_db,
        min_dbfs,
        verbose,
        rise_db,
        post_hit_rise_db,
        min_gap_ms,
    )
    block_size = max(64, round(profile.sample_rate * block_ms / 1000.0))
    waveform_window: Optional[WaveformWindow] = None
    if show_window:
        try:
            waveform_window = WaveformWindow(
                profile.sample_rate,
                display_dbfs=display_dbfs,
                initial_min_dbfs=min_dbfs,
            )
        except RuntimeError as exc:
            print(f"Waveform window unavailable ({exc}); continuing in terminal-only mode.", file=sys.stderr)

    def callback(indata, frames, callback_time, status):  # noqa: ANN001
        if status:
            try:
                audio_queue.put_nowait(f"audio status: {status}")
            except queue.Full:
                pass
        try:
            audio_queue.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            pass

    try:
        with sd.InputStream(
            samplerate=profile.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            device=device,
            callback=callback,
        ):
            print(
                f"Listening at {profile.sample_rate} Hz. Loudness jump={jump_db:.1f} dB, "
                f"onset rise={rise_db:.1f} dB, post-hit rise={post_hit_rise_db:.1f} dB, "
                f"match threshold={threshold:.2f}. "
                f"Calibrating room noise for {LIVE_WARMUP_SECONDS:.1f}s; press Ctrl-C to stop.",
                flush=True,
            )
            if waveform_window is None:
                while True:
                    item = audio_queue.get()
                    if isinstance(item, str):
                        print(item, file=sys.stderr, flush=True)
                    else:
                        detector.process(item)
            else:
                def poll_audio() -> None:
                    if not waveform_window or not waveform_window.running:
                        return
                    while True:
                        try:
                            item = audio_queue.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(item, str):
                            print(item, file=sys.stderr, flush=True)
                        else:
                            detector.min_dbfs = waveform_window.minimum_dbfs()
                            waveform_window.append(item)
                            detector.process(item)
                    waveform_window.redraw(detector)
                    waveform_window.root.after(16, poll_audio)

                waveform_window.root.after(0, poll_audio)
                waveform_window.root.mainloop()
                print("Waveform window closed; stopped listening.")
    except KeyboardInterrupt:
        print("\nStopped listening.")
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Could not open the microphone input: {exc}") from exc
    finally:
        if waveform_window is not None:
            waveform_window.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enroll and detect loud, user-defined rhythmic sounds.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_parser = subparsers.add_parser("enroll", help="Build a profile from example recordings")
    enroll_parser.add_argument("files", nargs="+", type=Path, help="One recording per example hit")
    enroll_parser.add_argument("--profile", type=Path, default=Path("sound_profile.json"))
    enroll_parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    enroll_parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    enroll_parser.add_argument("--pre-seconds", type=float, default=DEFAULT_PRE_SECONDS)
    enroll_parser.add_argument(
        "--negative",
        action="append",
        type=Path,
        default=[],
        help="A loud sound that should be rejected; repeat for several examples",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Add target examples recorded through the microphone",
    )
    calibrate_parser.add_argument("--profile", type=Path, default=Path("sound_profile.json"))
    calibrate_parser.add_argument("--count", type=int, default=5, help="Number of target hits to record")
    calibrate_parser.add_argument("--record-seconds", type=float, default=1.0)
    calibrate_parser.add_argument("--device", help="Input device name or numeric ID")

    background_parser = subparsers.add_parser(
        "calibrate-background",
        help="Record ordinary room sound as rejection examples",
    )
    background_parser.add_argument("--profile", type=Path, default=Path("sound_profile.json"))
    background_parser.add_argument("--seconds", type=float, default=6.0)
    background_parser.add_argument("--examples", type=int, default=24)
    background_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace previous rejection examples instead of appending",
    )
    background_parser.add_argument("--device", help="Input device name or numeric ID")

    for command, help_text in (
        ("detect", "Scan an audio file and print matching hits"),
        ("listen", "Listen to the default microphone and print matching hits"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        if command == "detect":
            command_parser.add_argument("file", type=Path)
        command_parser.add_argument("--profile", type=Path, default=Path("sound_profile.json"))
        command_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Profile similarity from 0 to 1")
        command_parser.add_argument("--jump-db", type=float, default=DEFAULT_JUMP_DB, help="Required increase above noise floor")
        command_parser.add_argument("--min-dbfs", type=float, default=DEFAULT_MIN_DBFS, help="Ignore quieter sounds")
        command_parser.add_argument("--rise-db", type=float, default=DEFAULT_RISE_DB, help="Required short-term attack rise")
        command_parser.add_argument(
            "--post-hit-rise-db",
            type=float,
            default=DEFAULT_POST_HIT_RISE_DB,
            help="Required rise for another candidate shortly after a hit",
        )
        command_parser.add_argument("--min-gap-ms", type=float, default=DEFAULT_MIN_GAP_MS, help="Minimum time between candidate hits")
        command_parser.add_argument("--verbose", action="store_true", help="Print loud candidates rejected by the profile")
        if command == "listen":
            command_parser.add_argument("--block-ms", type=float, default=DEFAULT_BLOCK_MS, help="Microphone analysis block size")
            command_parser.add_argument("--device", help="Input device name or numeric ID")
            command_parser.add_argument("--no-window", action="store_true", help="Run without the live waveform window")
            command_parser.add_argument(
                "--display-dbfs",
                type=float,
                default=DEFAULT_DISPLAY_DBFS,
                help="Fixed waveform full-scale level at or below 0 dBFS",
            )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "enroll":
            enroll(args.files, args.profile, args.sample_rate, args.clip_seconds, args.pre_seconds, args.negative)
            return 0
        profile = load_profile(args.profile)
        if args.command == "calibrate":
            device = int(args.device) if args.device and args.device.isdigit() else args.device
            calibrate_microphone(args.profile, profile, args.count, args.record_seconds, device)
            return 0
        if args.command == "calibrate-background":
            device = int(args.device) if args.device and args.device.isdigit() else args.device
            calibrate_background(
                args.profile,
                profile,
                args.seconds,
                args.examples,
                args.replace,
                device,
            )
            return 0
        if not 0.0 <= args.threshold <= 1.0:
            raise RuntimeError("--threshold must be between 0 and 1")
        if args.command == "detect":
            detect_file(
                args.file,
                profile,
                args.threshold,
                args.jump_db,
                args.min_dbfs,
                args.rise_db,
                args.post_hit_rise_db,
                args.min_gap_ms,
                args.verbose,
            )
        else:
            device = int(args.device) if args.device and args.device.isdigit() else args.device
            listen(
                profile,
                args.threshold,
                args.jump_db,
                args.min_dbfs,
                args.rise_db,
                args.post_hit_rise_db,
                args.min_gap_ms,
                args.verbose,
                args.block_ms,
                device,
                not args.no_window,
                args.display_dbfs,
            )
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
