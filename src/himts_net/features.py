from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HiMTSInputs:
    time: np.ndarray
    local_dse: np.ndarray
    entropy_contribution: np.ndarray
    sms: np.ndarray


def _frame_view(windows: np.ndarray, frame_length: int, frame_stride: int) -> np.ndarray:
    if windows.shape[1] < frame_length:
        raise ValueError(
            f"Window length {windows.shape[1]} is shorter than frame length {frame_length}"
        )
    return np.lib.stride_tricks.sliding_window_view(
        windows,
        frame_length,
        axis=1,
    )[:, ::frame_stride, :]


def _build_chunk(
    windows: np.ndarray,
    frame_length: int,
    frame_stride: int,
) -> HiMTSInputs:
    amplitude = np.abs(windows)
    time = np.stack((windows.real, windows.imag, amplitude), axis=1).astype(np.float32)

    full_spectrum = np.abs(np.fft.fft(windows, axis=1))
    full_probabilities = full_spectrum / np.maximum(
        full_spectrum.sum(axis=1, keepdims=True),
        1e-12,
    )
    entropy_contribution = -(
        full_probabilities * np.log(np.maximum(full_probabilities, 1e-12))
    )[:, None, :].astype(np.float32)

    frames = _frame_view(windows, frame_length, frame_stride)
    frame_spectrum = np.abs(np.fft.fft(frames, axis=2))
    frame_probabilities = frame_spectrum / np.maximum(
        frame_spectrum.sum(axis=2, keepdims=True),
        1e-12,
    )
    local_dse = -(
        frame_probabilities * np.log(np.maximum(frame_probabilities, 1e-12))
    ).sum(axis=2)[:, None, :].astype(np.float32)

    taper = np.hanning(frame_length)[None, None, :]
    sms = np.abs(np.fft.fft(frames * taper, axis=2)).sum(axis=1)
    sms = sms[:, None, :].astype(np.float32)
    return HiMTSInputs(
        time=time,
        local_dse=local_dse,
        entropy_contribution=entropy_contribution,
        sms=sms,
    )


def build_inputs(
    windows: np.ndarray,
    frame_length: int = 64,
    frame_stride: int = 16,
    chunk_size: int = 2048,
) -> HiMTSInputs:
    windows = np.asarray(windows)
    if windows.ndim != 2:
        raise ValueError("windows must have shape (samples, slow_time)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    sample_count, window_length = windows.shape
    frame_count = 1 + (window_length - frame_length) // frame_stride
    output = HiMTSInputs(
        time=np.empty((sample_count, 3, window_length), dtype=np.float32),
        local_dse=np.empty((sample_count, 1, frame_count), dtype=np.float32),
        entropy_contribution=np.empty((sample_count, 1, window_length), dtype=np.float32),
        sms=np.empty((sample_count, 1, frame_length), dtype=np.float32),
    )
    for start in range(0, sample_count, chunk_size):
        stop = min(sample_count, start + chunk_size)
        chunk = _build_chunk(
            np.asarray(windows[start:stop]),
            frame_length=frame_length,
            frame_stride=frame_stride,
        )
        for field in HiMTSInputs.__dataclass_fields__:
            getattr(output, field)[start:stop] = getattr(chunk, field)
    return output


def fit_normalizer(inputs: HiMTSInputs) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    normalizer = {}
    for field in HiMTSInputs.__dataclass_fields__:
        values = getattr(inputs, field)
        mean = values.mean(axis=(0, 2), keepdims=True)
        std = values.std(axis=(0, 2), keepdims=True) + 1e-6
        normalizer[field] = (mean.astype(np.float32), std.astype(np.float32))
    return normalizer


def apply_normalizer(
    inputs: HiMTSInputs,
    normalizer: dict[str, tuple[np.ndarray, np.ndarray]],
) -> HiMTSInputs:
    return HiMTSInputs(
        **{
            field: (
                (getattr(inputs, field) - normalizer[field][0]) / normalizer[field][1]
            ).astype(np.float32)
            for field in HiMTSInputs.__dataclass_fields__
        }
    )

