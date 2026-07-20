from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from .features import HiMTSInputs
from .training import HiMTSDataset


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 256
    target_pfa: float = 1e-3
    device: str = "cuda"
    num_workers: int = 0


def calibrate_threshold(
    clutter_scores: np.ndarray,
    target_pfa: float = 1e-3,
) -> float:
    clutter_scores = np.sort(np.asarray(clutter_scores))
    if len(clutter_scores) == 0:
        raise ValueError("Test data must contain clutter samples")
    if not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must be in (0, 1)")
    allowed_false_alarms = int(np.floor(target_pfa * len(clutter_scores)))
    if allowed_false_alarms <= 0:
        return float(np.nextafter(clutter_scores[-1], np.inf))
    threshold = float(clutter_scores[-allowed_false_alarms])
    if np.mean(clutter_scores >= threshold) > target_pfa:
        threshold = float(np.nextafter(threshold, np.inf))
    return threshold


@torch.no_grad()
def _predict(
    model: nn.Module,
    inputs: HiMTSInputs,
    labels: np.ndarray,
    config: EvaluationConfig,
) -> np.ndarray:
    dataset = HiMTSDataset(inputs, labels)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": max(0, config.num_workers),
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    if config.device.startswith("cuda"):
        loader_kwargs["pin_memory"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    model.eval()
    outputs = []
    non_blocking = config.device.startswith("cuda")
    for time, local_dse, entropy_contribution, sms, _ in loader:
        logits = model(
            time=time.to(config.device, non_blocking=non_blocking),
            local_dse=local_dse.to(config.device, non_blocking=non_blocking),
            entropy_contribution=entropy_contribution.to(
                config.device,
                non_blocking=non_blocking,
            ),
            sms=sms.to(config.device, non_blocking=non_blocking),
        )
        outputs.append(torch.sigmoid(logits).detach())
    return torch.cat(outputs).cpu().numpy()


def evaluate_model(
    model: nn.Module,
    inputs: HiMTSInputs,
    labels: np.ndarray,
    output_dir: Path,
    config: EvaluationConfig,
) -> dict:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    labels = np.asarray(labels)
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        raise ValueError("Test data must contain both target and clutter samples")

    model = model.to(config.device)
    scores = _predict(model, inputs, labels, config)
    threshold = calibrate_threshold(
        scores[negatives],
        target_pfa=config.target_pfa,
    )
    predictions = scores >= threshold
    fpr, tpr, _ = roc_curve(labels, scores)
    metrics = {
        "pd": float(predictions[positives].mean()),
        "pfa": float(predictions[negatives].mean()),
        "auc": float(roc_auc_score(labels, scores)),
        "threshold": threshold,
        "threshold_source": "test_clutter_only",
        "target_pfa": config.target_pfa,
        "sample_count": int(len(labels)),
        "target_count": int(positives.sum()),
        "clutter_count": int(negatives.sum()),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics
