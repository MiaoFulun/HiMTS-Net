from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


WINDOW_LENGTH = 1024
TARGET_STRIDE = 200
CLUTTER_STRIDE = 1024
PULSE_COUNT = 131072
OBSERVATION_CELLS = tuple(range(401, 501))


@dataclass(frozen=True)
class Record:
    record_id: str
    file_name: str
    target_cells: tuple[int, ...]
    matrix_key: str = "amplitude_complex_T1"


RECORDS = (
    Record(
        "20221114140049_stare_HH",
        "20221114140049_stare_HH.mat",
        tuple(range(442, 456)),
    ),
    Record(
        "20221114220100_stare_HH",
        "20221114220100_stare_HH.mat",
        tuple(range(442, 455)),
    ),
    Record(
        "20221113190042_stare_HH",
        "20221113190042_stare_HH.mat",
        tuple(range(439, 452)),
    ),
    Record(
        "20221113230100_stare_HH",
        "20221113230100_stare_HH.mat",
        tuple(range(439, 452)),
    ),
    Record(
        "20221112205109_stare_HH",
        "20221112205109_stare_HH.mat",
        tuple(range(438, 450)),
    ),
    Record(
        "20221112180016_stare_HH",
        "20221112180016_stare_HH.mat",
        tuple(range(438, 450)),
    ),
)


def selected_records(record_ids: list[str]) -> tuple[Record, ...]:
    lookup = {record.record_id: record for record in RECORDS}
    missing = sorted(set(record_ids) - set(lookup))
    if missing:
        raise ValueError(f"Unknown SDRDSP2022 records: {', '.join(missing)}")
    return tuple(lookup[record_id] for record_id in record_ids)


def _split_ranges() -> dict[str, tuple[int, int]]:
    return {
        "train": (0, int(0.70 * PULSE_COUNT)),
        "validation": (int(0.70 * PULSE_COUNT), int(0.85 * PULSE_COUNT)),
    }


def _window_starts(split: str, stride: int, window_length: int) -> range:
    start, stop = _split_ranges()[split]
    return range(start, stop - window_length + 1, stride)


def _rows(
    records: tuple[Record, ...],
    window_length: int,
    target_stride: int,
    clutter_stride: int,
) -> dict[str, list[dict]]:
    rows = {"train": [], "validation": []}
    for record in records:
        target_set = set(record.target_cells)
        clutter_cells = [cell for cell in OBSERVATION_CELLS if cell not in target_set]
        for split in rows:
            for gate in record.target_cells:
                for start in _window_starts(split, target_stride, window_length):
                    rows[split].append(
                        {
                            "file_name": record.file_name,
                            "matrix_key": record.matrix_key,
                            "gate": gate,
                            "start": start,
                            "label": 1,
                        }
                    )
            for gate in clutter_cells:
                for start in _window_starts(split, clutter_stride, window_length):
                    rows[split].append(
                        {
                            "file_name": record.file_name,
                            "matrix_key": record.matrix_key,
                            "gate": gate,
                            "start": start,
                            "label": 0,
                        }
                    )
    return rows


def _to_complex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.iscomplexobj(values):
        return values.astype(np.complex64, copy=False)
    if values.dtype.fields and {"real", "imag"} <= set(values.dtype.fields):
        return (values["real"] + 1j * values["imag"]).astype(np.complex64)
    raise ValueError(f"Expected a complex array, got {values.dtype}")


def _materialize(
    data_dir: Path,
    rows: list[dict],
    window_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.empty((len(rows), window_length), dtype=np.complex64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.float32)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["file_name"], row["matrix_key"])].append(index)

    for (file_name, matrix_key), indices in groups.items():
        path = Path(data_dir) / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        gates = [int(rows[index]["gate"]) for index in indices]
        starts = [int(rows[index]["start"]) for index in indices]
        min_gate = min(gates)
        max_gate = max(gates)
        min_start = min(starts)
        max_stop = max(start + window_length for start in starts)
        with h5py.File(path, "r") as handle:
            if matrix_key not in handle:
                raise KeyError(f"{matrix_key} was not found in {path}")
            dataset = handle[matrix_key]
            if len(dataset.shape) != 2:
                raise ValueError(f"Expected a range-by-pulse matrix, got {dataset.shape}")
            block = _to_complex(
                dataset[min_gate - 1 : max_gate, min_start:max_stop]
            )
        for index, gate, start in zip(indices, gates, starts):
            windows[index] = block[
                gate - min_gate,
                start - min_start : start - min_start + window_length,
            ]
    return windows, labels


def prepare_sdrdsp2022_training_data(
    data_dir: Path,
    records: tuple[Record, ...],
    window_length: int = WINDOW_LENGTH,
    target_stride: int = TARGET_STRIDE,
    clutter_stride: int = CLUTTER_STRIDE,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = _rows(
        records,
        window_length=window_length,
        target_stride=target_stride,
        clutter_stride=clutter_stride,
    )
    materialized = {
        split: _materialize(data_dir, split_rows, window_length)
        for split, split_rows in rows.items()
    }
    return (
        {split: values[0] for split, values in materialized.items()},
        {split: values[1] for split, values in materialized.items()},
    )

