from pathlib import Path
import json

import numpy as np
import torch
import yaml

from .data_ipix import prepare_ipix_data
from .data_sdrdsp2022 import (
    prepare_sdrdsp2022_data,
    selected_records,
)
from .evaluation import EvaluationConfig, evaluate_model
from .features import HiMTSInputs, apply_normalizer, build_inputs, fit_normalizer
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


def _load_normalizer(
    output_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = Path(output_dir) / "normalizer.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as arrays:
        return {
            name: (
                np.asarray(arrays[f"{name}_mean"]),
                np.asarray(arrays[f"{name}_std"]),
            )
            for name in HiMTSInputs.__dataclass_fields__
        }


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


def _evaluate_one(
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path,
    values: dict,
) -> dict:
    chunk_size = int(values.get("feature_chunk_size", 2048))
    test_inputs = build_inputs(test_windows, chunk_size=chunk_size)
    normalizer = _load_normalizer(output_dir)
    test_inputs = apply_normalizer(test_inputs, normalizer)
    model = HiMTSNet(hidden_dim=int(values.get("hidden_dim", 32)))
    checkpoint_path = Path(output_dir) / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(
        model,
        inputs=test_inputs,
        labels=test_labels,
        output_dir=output_dir,
        config=EvaluationConfig(
            batch_size=int(values.get("batch_size", 256)),
            target_pfa=float(values.get("target_pfa", 1e-3)),
            device=str(values.get("device", "cuda")),
            num_workers=int(values.get("num_workers", 0)),
        ),
    )
    summary = {
        "output_dir": str(output_dir),
        **{
            name: metrics[name]
            for name in ("pd", "pfa", "auc", "threshold", "sample_count")
        },
    }
    print(json.dumps(summary), flush=True)
    return metrics


def _run_ipix(values: dict) -> None:
    data_dir = Path(values["data_dir"])
    output_root = Path(values["output_dir"])
    for dataset_id in values["dataset_ids"]:
        for polarization in values["polarizations"]:
            print(
                f"Preparing IPIX dataset {dataset_id}, polarization {polarization}",
                flush=True,
            )
            windows, labels = prepare_ipix_data(
                data_dir=data_dir,
                dataset_id=int(dataset_id),
                polarization=str(polarization),
                seed=int(values.get("seed", 42)),
                window_length=int(values.get("window_length", 512)),
                window_stride=int(values.get("window_stride", 32)),
                splits=("train", "validation"),
            )
            _train_one(
                windows,
                labels,
                output_root / f"dataset_{dataset_id}_{polarization}",
                values,
            )


def _run_sdrdsp2022(values: dict) -> None:
    records = selected_records([str(value) for value in values["record_ids"]])
    windows, labels = prepare_sdrdsp2022_data(
        data_dir=Path(values["data_dir"]),
        records=records,
        window_length=int(values.get("window_length", 1024)),
        target_stride=int(values.get("target_stride", 200)),
        clutter_stride=int(values.get("clutter_stride", 1024)),
        splits=("train", "validation"),
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


def _evaluate_ipix(values: dict) -> None:
    data_dir = Path(values["data_dir"])
    output_root = Path(values["output_dir"])
    for dataset_id in values["dataset_ids"]:
        for polarization in values["polarizations"]:
            print(
                f"Preparing IPIX test set {dataset_id}, polarization {polarization}",
                flush=True,
            )
            windows, labels = prepare_ipix_data(
                data_dir=data_dir,
                dataset_id=int(dataset_id),
                polarization=str(polarization),
                seed=int(values.get("seed", 42)),
                window_length=int(values.get("window_length", 512)),
                window_stride=int(values.get("window_stride", 32)),
                splits=("test",),
            )
            _evaluate_one(
                windows["test"],
                labels["test"],
                output_root / f"dataset_{dataset_id}_{polarization}",
                values,
            )


def _evaluate_sdrdsp2022(values: dict) -> None:
    records = selected_records([str(value) for value in values["record_ids"]])
    windows, labels = prepare_sdrdsp2022_data(
        data_dir=Path(values["data_dir"]),
        records=records,
        window_length=int(values.get("window_length", 1024)),
        target_stride=int(values.get("target_stride", 200)),
        clutter_stride=int(values.get("clutter_stride", 1024)),
        splits=("test",),
    )
    _evaluate_one(
        windows["test"],
        labels["test"],
        Path(values["output_dir"]),
        values,
    )


def run_evaluation(config_path: Path) -> None:
    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = str(values.get("dataset", "")).lower()
    if dataset == "ipix":
        _evaluate_ipix(values)
    elif dataset == "sdrdsp2022":
        _evaluate_sdrdsp2022(values)
    else:
        raise ValueError("dataset must be 'ipix' or 'sdrdsp2022'")
