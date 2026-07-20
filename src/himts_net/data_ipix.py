from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PaperDataset:
    dataset_id: int
    file_name: str
    primary_target_bin: int
    affected_bins: tuple[int, ...]


PAPER_DATASETS = (
    PaperDataset(1, "19931118_023604_starea280.cdf", 8, (7, 9, 10)),
    PaperDataset(2, "19931107_135603_starea17.cdf", 9, (8, 10, 11)),
    PaperDataset(3, "19931108_220902_starea26.cdf", 7, (6, 8)),
    PaperDataset(4, "19931109_191449_starea30.cdf", 7, (6, 8)),
    PaperDataset(5, "19931109_202217_starea31.cdf", 7, (6, 8, 9)),
    PaperDataset(6, "19931110_001635_starea40.cdf", 7, (5, 6, 8)),
    PaperDataset(7, "19931111_163625_starea54.cdf", 8, (7, 9, 10)),
    PaperDataset(8, "19931118_162155_starea310.cdf", 7, (6, 8, 9)),
    PaperDataset(9, "19931118_162658_starea311.cdf", 7, (6, 8, 9)),
    PaperDataset(10, "19931118_174259_starea320.cdf", 7, (6, 8, 9)),
)


def _dataset(dataset_id: int) -> PaperDataset:
    for item in PAPER_DATASETS:
        if item.dataset_id == dataset_id:
            return item
    raise ValueError(f"Unknown IPIX dataset id {dataset_id}")


def _load_channel(path: Path, polarization: str) -> np.ndarray:
    with netcdf_file(path, "r", mmap=False) as handle:
        adc = handle.variables["adc_data"].data.copy()
    channels = {
        "HH": adc[:, 0, :, 0].astype(float) + 1j * adc[:, 0, :, 1].astype(float),
        "HV": adc[:, 0, :, 2].astype(float) + 1j * adc[:, 0, :, 3].astype(float),
        "VH": adc[:, 1, :, 2].astype(float) + 1j * adc[:, 1, :, 3].astype(float),
        "VV": adc[:, 1, :, 0].astype(float) + 1j * adc[:, 1, :, 1].astype(float),
    }
    if polarization not in channels:
        raise ValueError(f"Unknown IPIX polarization '{polarization}'")
    return channels[polarization].astype(np.complex64)


def _rows(
    item: PaperDataset,
    pulse_count: int,
    window_length: int,
    window_stride: int,
) -> list[tuple[int, int, int]]:
    excluded = {item.primary_target_bin, *item.affected_bins}
    clutter_bins = [gate for gate in range(1, 15) if gate not in excluded]
    gates = [(item.primary_target_bin, 1), *[(gate, 0) for gate in clutter_bins]]
    return [
        (gate, start, label)
        for gate, label in gates
        for start in range(0, pulse_count - window_length + 1, window_stride)
    ]


def _materialize(
    matrix: np.ndarray,
    rows: list[tuple[int, int, int]],
    selected_ids: set[int],
    window_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    chosen = [row for index, row in enumerate(rows) if index in selected_ids]
    windows = np.empty((len(chosen), window_length), dtype=np.complex64)
    labels = np.empty(len(chosen), dtype=np.float32)
    for index, (gate, start, label) in enumerate(chosen):
        windows[index] = matrix[start : start + window_length, gate - 1]
        labels[index] = label
    return windows, labels


def prepare_ipix_data(
    data_dir: Path,
    dataset_id: int,
    polarization: str,
    seed: int,
    window_length: int = 512,
    window_stride: int = 32,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    valid_splits = {"train", "validation", "test"}
    unknown_splits = set(splits) - valid_splits
    if unknown_splits:
        raise ValueError(f"Unknown IPIX splits: {', '.join(sorted(unknown_splits))}")
    item = _dataset(dataset_id)
    path = Path(data_dir) / item.file_name
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = _load_channel(path, polarization)
    rows = _rows(item, matrix.shape[0], window_length, window_stride)
    sample_ids = np.arange(len(rows))
    row_labels = np.asarray([row[2] for row in rows])
    train_pool, test_ids, train_pool_labels, _ = train_test_split(
        sample_ids,
        row_labels,
        test_size=0.30,
        stratify=row_labels,
        random_state=seed,
    )
    train_ids, validation_ids = train_test_split(
        train_pool,
        test_size=0.20,
        stratify=train_pool_labels,
        random_state=seed,
    )
    ids_by_split = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }
    materialized = {
        split: _materialize(
            matrix,
            rows,
            set(int(value) for value in ids_by_split[split]),
            window_length,
        )
        for split in splits
    }
    return (
        {split: values[0] for split, values in materialized.items()},
        {split: values[1] for split, values in materialized.items()},
    )
