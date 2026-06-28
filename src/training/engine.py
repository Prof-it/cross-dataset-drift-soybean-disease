"""Dataset-agnostic training engine.

A single training loop reused by every experiment, with the reliability features
the Colab runs depend on. ``fit`` is idempotent: if a run's checkpoint and log
already exist it returns immediately, so interrupted sessions resume cleanly.

The optimizer is built over the model's *trainable* parameters, so linear probing
needs no special path here, just freeze the backbone before calling ``fit``.

Per-epoch accuracy and macro F1 are computed with scikit-learn purely for
monitoring/logging; the authoritative metrics (ECE, per-class, bootstrap CIs) live
in :mod:`src.evaluation` and are computed from saved predictions at evaluation time.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import set_seed
from src.models.factory import create_model, reinitialize_classifier
from src.training.optim import build_optimizer, build_scheduler

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def _epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: "torch.amp.GradScaler | None" = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train for one epoch; return ``(avg_loss, y_true, y_pred)``."""
    use_amp = scaler is not None and device.type == "cuda"
    model.train()
    running_loss = 0.0
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()  # type: ignore[union-attr]
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        labels_all.append(labels.cpu().numpy())
        preds_all.append(logits.argmax(dim=1).cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)  # type: ignore[arg-type]
    return avg_loss, np.concatenate(labels_all), np.concatenate(preds_all)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate on a loader; return ``(avg_loss, y_true, y_pred)``."""
    use_amp = use_amp and device.type == "cuda"
    model.eval()
    running_loss = 0.0
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        labels_all.append(labels.cpu().numpy())
        preds_all.append(logits.argmax(dim=1).cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)  # type: ignore[arg-type]
    return avg_loss, np.concatenate(labels_all), np.concatenate(preds_all)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str | Path,
    criterion: nn.Module,
    cfg: "Config",
    device: torch.device,
    run_id: str,
) -> dict:
    """Full training loop with early stopping, plateau LR, AMP, and checkpointing.

    Idempotent: if ``save_dir/best_model.pt`` and ``training_log.json`` both exist,
    the saved log is returned without retraining (Colab-reconnect safe).
    """
    save_dir = Path(save_dir)
    checkpoint_path = save_dir / "best_model.pt"
    log_path = save_dir / "training_log.json"

    if checkpoint_path.exists() and log_path.exists():
        logger.info("[%s] checkpoint found, skipping training.", run_id)
        return json.loads(log_path.read_text())

    save_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    use_amp = cfg.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    trainable = (p for p in model.parameters() if p.requires_grad)
    optimizer = build_optimizer(trainable, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    writer = SummaryWriter(log_dir=str(cfg.paths.logs_dir / run_id))
    log: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_macro_f1": [], "val_macro_f1": [], "lr": [],
    }

    es = cfg.training.early_stopping
    best_val_loss = float("inf")
    es_counter = 0
    epoch = 0

    logger.info("[%s] training (max %d epochs)", run_id, cfg.training.max_epochs)
    for epoch in range(1, cfg.training.max_epochs + 1):
        train_loss, train_true, train_pred = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_loss, val_true, val_pred = evaluate(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        train_m = _epoch_metrics(train_true, train_pred)
        val_m = _epoch_metrics(val_true, val_pred)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["train_accuracy"].append(train_m["accuracy"])
        log["val_accuracy"].append(val_m["accuracy"])
        log["train_macro_f1"].append(train_m["macro_f1"])
        log["val_macro_f1"].append(val_m["macro_f1"])
        log["lr"].append(current_lr)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("MacroF1/train", train_m["macro_f1"], epoch)
        writer.add_scalar("MacroF1/val", val_m["macro_f1"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Reset patience only on a meaningful improvement; still checkpoint on any
        # improvement so the best weights are always retained.
        if val_loss < best_val_loss - es.min_delta:
            es_counter = 0
        else:
            es_counter += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        logger.info(
            "[%s] epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f  lr=%.2e  ES %d/%d",
            run_id, epoch, cfg.training.max_epochs, train_loss, val_loss,
            val_m["macro_f1"], current_lr, es_counter, es.patience,
        )
        if es_counter >= es.patience:
            logger.info("[%s] early stopping at epoch %d", run_id, epoch)
            break

    writer.close()
    log["best_epoch"] = int(np.argmin(log["val_loss"])) + 1
    log["best_val_loss"] = best_val_loss
    log["stopped_epoch"] = epoch
    log_path.write_text(json.dumps(log, indent=2))
    logger.info("[%s] done, best epoch %d (val_loss %.4f)", run_id, log["best_epoch"], best_val_loss)

    # Release GPU memory before the next model in the loop.
    del optimizer, scheduler, scaler
    model.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info("[%s] gpu reserved %.1f GB", run_id, torch.cuda.memory_reserved() / 1024**3)
    return log


def fit_two_stage(
    arch: str,
    pv_loaders: dict[str, DataLoader],
    target_loaders: dict[str, DataLoader],
    pv_save_dir: str | Path,
    target_save_dir: str | Path,
    pv_criterion: nn.Module,
    target_criterion: nn.Module,
    cfg: "Config",
    device: torch.device,
    pv_run_id: str,
    target_run_id: str,
    pv_seed: int,
    target_seed: int,
) -> dict:
    """Two-stage training: PlantVillage pretraining, then soybean fine-tuning.

    Stage 1 uses a fixed ``pv_seed`` and an idempotent checkpoint, so it is trained
    once per architecture and reused by every two-stage run. Stage 2 loads those
    weights under ``target_seed``, swaps the head to the soybean label space, and
    fine-tunes on the target.
    """
    num_pv = cfg.data.num_classes_plantvillage
    num_target = cfg.data.num_classes

    set_seed(pv_seed)
    model = create_model(arch, num_pv, cfg.architectures, pretrained=True)
    stage1 = fit(model, pv_loaders["train"], pv_loaders["val"], pv_save_dir,
                 pv_criterion, cfg, device, pv_run_id)

    set_seed(target_seed)
    model = create_model(arch, num_pv, cfg.architectures, pretrained=True)
    state = torch.load(Path(pv_save_dir) / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    reinitialize_classifier(model, arch, num_target, cfg.architectures)
    stage2 = fit(model, target_loaders["train"], target_loaders["val"], target_save_dir,
                 target_criterion, cfg, device, target_run_id)

    return {"stage1": stage1, "stage2": stage2}
