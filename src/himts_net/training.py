from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import HiMTSInputs


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    hard_negative_fraction: float = 0.25
    target_pfa: float = 1e-3
    device: str = "cuda"
    num_workers: int = 0


class HiMTSDataset(Dataset):
    def __init__(self, inputs: HiMTSInputs, labels: np.ndarray):
        self.time = torch.from_numpy(inputs.time)
        self.local_dse = torch.from_numpy(inputs.local_dse)
        self.entropy_contribution = torch.from_numpy(inputs.entropy_contribution)
        self.sms = torch.from_numpy(inputs.sms)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.time[index],
            self.local_dse[index],
            self.entropy_contribution[index],
            self.sms[index],
            self.labels[index],
        )


def hard_negative_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fraction: float = 0.25,
) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("hard-negative fraction must be in (0, 1]")
    components = []
    positive_logits = logits[labels == 1]
    if len(positive_logits):
        components.append(
            nn.functional.binary_cross_entropy_with_logits(
                positive_logits,
                torch.ones_like(positive_logits),
            )
        )
    negative_logits = logits[labels == 0]
    if len(negative_logits):
        keep = max(1, int(fraction * len(negative_logits)))
        hard_negative_logits = torch.topk(negative_logits, k=keep).values
        components.append(
            nn.functional.binary_cross_entropy_with_logits(
                hard_negative_logits,
                torch.zeros_like(hard_negative_logits),
            )
        )
    if not components:
        raise ValueError("A training batch must contain at least one labeled sample")
    return torch.stack(components).mean()


def _loader(dataset: Dataset, config: TrainingConfig, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": max(0, config.num_workers),
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
    if config.device.startswith("cuda"):
        kwargs["pin_memory"] = True
    return DataLoader(dataset, **kwargs)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> float:
    model.train()
    losses = []
    non_blocking = config.device.startswith("cuda")
    for time, local_dse, entropy_contribution, sms, labels in loader:
        optimizer.zero_grad()
        logits = model(
            time=time.to(config.device, non_blocking=non_blocking),
            local_dse=local_dse.to(config.device, non_blocking=non_blocking),
            entropy_contribution=entropy_contribution.to(
                config.device,
                non_blocking=non_blocking,
            ),
            sms=sms.to(config.device, non_blocking=non_blocking),
        )
        loss = hard_negative_loss(
            logits,
            labels.to(config.device, non_blocking=non_blocking),
            fraction=config.hard_negative_fraction,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    outputs = []
    non_blocking = device.startswith("cuda")
    for time, local_dse, entropy_contribution, sms, _ in loader:
        outputs.append(
            model(
                time=time.to(device, non_blocking=non_blocking),
                local_dse=local_dse.to(device, non_blocking=non_blocking),
                entropy_contribution=entropy_contribution.to(
                    device,
                    non_blocking=non_blocking,
                ),
                sms=sms.to(device, non_blocking=non_blocking),
            ).detach()
        )
    return torch.cat(outputs).cpu().numpy()


def _validation_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    target_pfa: float,
) -> dict[str, float]:
    clutter_scores = np.sort(logits[labels == 0])
    if len(clutter_scores) == 0:
        raise ValueError("Validation data must contain clutter samples")
    allowed_false_alarms = int(np.floor(target_pfa * len(clutter_scores)))
    if allowed_false_alarms <= 0:
        threshold = float(np.nextafter(clutter_scores[-1], np.inf))
    else:
        threshold = float(clutter_scores[-allowed_false_alarms])
        if np.mean(clutter_scores >= threshold) > target_pfa:
            threshold = float(np.nextafter(threshold, np.inf))
    predictions = logits >= threshold
    positives = labels == 1
    negatives = labels == 0
    return {
        "pd": float(predictions[positives].mean()),
        "pfa": float(predictions[negatives].mean()),
        "auc": float(roc_auc_score(labels, logits)),
        "threshold": threshold,
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_model(
    model: nn.Module,
    inputs: dict[str, HiMTSInputs],
    labels: dict[str, np.ndarray],
    output_dir: Path,
    config: TrainingConfig,
) -> dict:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_loader = _loader(
        HiMTSDataset(inputs["train"], labels["train"]),
        config,
        shuffle=True,
    )
    validation_loader = _loader(
        HiMTSDataset(inputs["validation"], labels["validation"]),
        config,
        shuffle=False,
    )
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    best_state = _cpu_state_dict(model)
    selected_epoch = 1
    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, config)
        validation_logits = _predict(model, validation_loader, config.device)
        metrics = _validation_metrics(
            np.asarray(labels["validation"]),
            validation_logits,
            config.target_pfa,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_pd": metrics["pd"],
            "validation_pfa": metrics["pfa"],
            "validation_auc": metrics["auc"],
            "hard_negative_fraction": config.hard_negative_fraction,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (metrics["pd"], metrics["auc"], -train_loss)
        if key > best_key:
            best_key = key
            best_state = _cpu_state_dict(model)
            selected_epoch = epoch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "selected_epoch": selected_epoch,
            "training_config": asdict(config),
        },
        output_dir / "model.pt",
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return {
        "selected_epoch": selected_epoch,
        "checkpoint": str(output_dir / "model.pt"),
    }

