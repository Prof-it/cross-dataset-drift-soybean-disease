"""Tests: model construction (need torch + timm; CPU, no pretrained downloads)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.config import load_config
from src.models.factory import (
    backbone_features,
    classifier,
    classifier_parameters,
    create_model,
    freeze_backbone,
    reinitialize_classifier,
)

ARCHES = ["resnet50", "densenet201", "vit_small", "vit_base"]


def _feature_loader(n=6, batch=3):
    """Loader of random images yielding (image, label, id) like CsvImageDataset."""
    from torch.utils.data import DataLoader

    data = [(torch.randn(3, 224, 224), i % 3, f"img{i}") for i in range(n)]
    return DataLoader(data, batch_size=batch)


@pytest.mark.parametrize("arch", ARCHES)
def test_build_and_forward(arch):
    cfg = load_config()
    model = create_model(arch, num_classes=3, architectures=cfg.architectures, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)


def test_reinitialize_and_freeze():
    cfg = load_config()
    model = create_model("resnet50", num_classes=38, architectures=cfg.architectures, pretrained=False)
    reinitialize_classifier(model, "resnet50", 3, cfg.architectures)
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 3)

    freeze_backbone(model, "resnet50")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for p in classifier_parameters(model, "resnet50"))
    assert trainable == head and head > 0


@pytest.mark.parametrize("arch", ARCHES)
def test_backbone_features_shape_and_restore(arch):
    """Linear-probe feature caching: (N, feature_dim) features, head restored."""
    cfg = load_config()
    feature_dim = cfg.architectures[arch]["feature_dim"]
    model = create_model(arch, num_classes=3, architectures=cfg.architectures, pretrained=False)
    feats, labels = backbone_features(model, arch, _feature_loader(6), torch.device("cpu"))
    assert feats.shape == (6, feature_dim)
    assert labels.shape == (6,)
    # the original head must be restored (not left as Identity)
    assert isinstance(classifier(model, arch), torch.nn.Linear)


@pytest.mark.parametrize("arch", ["resnet50", "vit_small"])
def test_linear_probe_checkpoint_roundtrip(arch, tmp_path):
    """A grafted head + full state_dict must reload via the evaluator's contract."""
    cfg = load_config()
    feature_dim = cfg.architectures[arch]["feature_dim"]
    model = create_model(arch, num_classes=3, architectures=cfg.architectures, pretrained=False)
    reinitialize_classifier(model, arch, 3, cfg.architectures)

    head = torch.nn.Linear(feature_dim, 3)
    torch.nn.init.normal_(head.weight)
    classifier(model, arch).load_state_dict(head.state_dict())

    ckpt = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), ckpt)
    reloaded = create_model(arch, num_classes=3, architectures=cfg.architectures, pretrained=False)
    reloaded.load_state_dict(torch.load(ckpt, weights_only=True))  # evaluator's _load
    assert torch.allclose(classifier(reloaded, arch).weight, head.weight)
