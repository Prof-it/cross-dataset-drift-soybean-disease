"""Tests: one-epoch training smoke test (need torch + timm + sklearn + tensorboard)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("sklearn")
pytest.importorskip("tensorboard")

from src.models.factory import create_model
from src.training.engine import fit
from src.training.losses import build_loss


def test_one_epoch_then_idempotent_resume(train_cfg, synthetic_loaders, tmp_path):
    device = torch.device("cpu")
    model = create_model("resnet50", train_cfg.data.num_classes, train_cfg.architectures, pretrained=False)
    criterion = build_loss(train_cfg, class_weights=None, device=device)
    save_dir = tmp_path / "run"

    log = fit(model, synthetic_loaders["train"], synthetic_loaders["val"],
              save_dir, criterion, train_cfg, device, "smoke")
    assert (save_dir / "best_model.pt").exists()
    assert (save_dir / "training_log.json").exists()
    assert len(log["train_loss"]) == 1  # max_epochs=1

    # Second call must short-circuit (idempotent resume) and return the same log.
    again = fit(model, synthetic_loaders["train"], synthetic_loaders["val"],
                save_dir, criterion, train_cfg, device, "smoke")
    assert again["train_loss"] == log["train_loss"]
