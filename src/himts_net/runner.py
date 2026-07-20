from pathlib import Path

import numpy as np
import yaml

from .data_ipix import prepare_ipix_training_data
from .data_sdrdsp2022 import (
    prepare_sdrdsp2022_training_data,
    selected_records,
)
from .features import apply_normalizer, build_inputs, fit_normalizer
from .model import HiMTSNet
from .training import TrainingConfig, train_model


def _training_config(values: dict) -> TrainingConfig:
    return TrainingConfig(
        seed=int(values.get("seed", 42)),
        epochs=int(values.get("epochs", 30)),
        batch_size=int(values.get("batch_size", 256)),
        learning_rate=float(values.get("learning_rate", 1e-3)),
        hard_negative_fraction=float(values.get("hard_negative_fraction", 0.25)),
        target_pfa=float(values.get("target_pfa", 1e-3)),
        device=str(values.get("device", "cuda")),
        num_workers=int(values.get("num_workers", 0)),
    )


def _save_normalizer(
    output_dir: Path,
    normalizer: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, (mean, std) in normalizer.items():
        arrays[f"{name}_mean"] = mean
        arrays[f"{name}_std"] = std
    np.savez(output_dir / "normalizer.npz", **arrays)


def _train_one(
    windows: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    output_dir: Path,
    values: dict,
) -> None:
    chunk_size = int(values.get("feature_chunk_size", 2048))
    raw_inputs = {
        split: build_inputs(split_windows, chunk_size=chunk_size)
        for split, split_windows in windows.items()
    }
    normalizer = fit_normalizer(raw_inputs["train"])
    inputs = {
        split: apply_normalizer(split_inputs, normalizer)
        for split, split_inputs in raw_inputs.items()
    }
    _save_normalizer(output_dir, normalizer)
    model = HiMTSNet(hidden_dim=int(values.get("hidden_dim", 32)))
    train_model(
        model,
        inputs=inputs,
        labels=labels,
        output_dir=output_dir,
        config=_training_config(values),
    )


def _run_ipix(values: dict) -> None:
    data_dir = Path(values["data_dir"])
    output_root = Path(values["output_dir"])
    for dataset_id in values["dataset_ids"]:
        for polarization in values["polarizations"]:
            print(
                f"Preparing IPIX dataset {dataset_id}, polarization {polarization}",
                flush=True,
            )
            windows, labels = prepare_ipix_training_data(
                data_dir=data_dir,
                dataset_id=int(dataset_id),
                polarization=str(polarization),
                seed=int(values.get("seed", 42)),
                window_length=int(values.get("window_length", 512)),
                window_stride=int(values.get("window_stride", 32)),
            )
            _train_one(
                windows,
                labels,
                output_root / f"dataset_{dataset_id}_{polarization}",
                values,
            )


def _run_sdrdsp2022(values: dict) -> None:
    records = selected_records([str(value) for value in values["record_ids"]])
    windows, labels = prepare_sdrdsp2022_training_data(
        data_dir=Path(values["data_dir"]),
        records=records,
        window_length=int(values.get("window_length", 1024)),
        target_stride=int(values.get("target_stride", 200)),
        clutter_stride=int(values.get("clutter_stride", 1024)),
    )
    _train_one(windows, labels, Path(values["output_dir"]), values)


def run_training(config_path: Path) -> None:
    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = str(values.get("dataset", "")).lower()
    if dataset == "ipix":
        _run_ipix(values)
    elif dataset == "sdrdsp2022":
        _run_sdrdsp2022(values)
    else:
        raise ValueError("dataset must be 'ipix' or 'sdrdsp2022'")

