"""Model construction, head reinitialization, and backbone freezing.

Builds the four architectures from the registry with a fresh classifier head, and
provides the operations the experiments need beyond construction: swapping the
head between label spaces (two-stage) and freezing the backbone (linear probing).

torchvision backbones (ResNet-50, DenseNet-201) load ``IMAGENET1K_V1`` weights;
timm backbones (ViT-S/16, ViT-B/16) load their ``augreg_in1k`` weights. The four
architectures keep their classifier head under different attribute names, mapped
once in ``_HEAD_ATTR`` so the rest of the code stays architecture-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models as tv_models

# Attribute holding the classifier head for each architecture.
_HEAD_ATTR: dict[str, str] = {
    "densenet201": "classifier",
    "resnet50": "fc",
    "vit_small": "head",
    "vit_base": "head",
}


def _head_attr(arch: str) -> str:
    try:
        return _HEAD_ATTR[arch]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture: {arch}") from exc


def create_model(
    arch: str,
    num_classes: int,
    architectures: Mapping[str, Any],
    pretrained: bool = True,
) -> nn.Module:
    """Build ``arch`` with a fresh linear head over ``num_classes`` classes.

    ``architectures`` is the registry from the config (``cfg.architectures``),
    keyed by architecture name with ``source`` / ``model_name`` / ``feature_dim``.
    """
    spec = architectures[arch]
    source = spec["source"]
    model_name = spec["model_name"]

    if source == "torchvision":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = getattr(tv_models, model_name)(weights=weights)
        setattr(model, _head_attr(arch), nn.Linear(spec["feature_dim"], num_classes))
    elif source == "timm":
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown source '{source}' for architecture '{arch}'")
    return model


def reinitialize_classifier(
    model: nn.Module,
    arch: str,
    num_classes: int,
    architectures: Mapping[str, Any],
) -> nn.Module:
    """Replace the classifier head (e.g. 38 -> 3 when moving PlantVillage -> soybean)."""
    feature_dim = architectures[arch]["feature_dim"]
    setattr(model, _head_attr(arch), nn.Linear(feature_dim, num_classes))
    return model


def classifier(model: nn.Module, arch: str) -> nn.Module:
    """Return the classifier head module for ``arch``."""
    return getattr(model, _head_attr(arch))


def classifier_parameters(model: nn.Module, arch: str) -> Iterator[nn.Parameter]:
    """Parameters of the classifier head (used for linear probing / head refit)."""
    return classifier(model, arch).parameters()


def freeze_backbone(model: nn.Module, arch: str) -> nn.Module:
    """Freeze every parameter except the classifier head (linear probing)."""
    for param in model.parameters():
        param.requires_grad = False
    for param in classifier_parameters(model, arch):
        param.requires_grad = True
    return model


@torch.no_grad()
def backbone_features(
    model: nn.Module,
    arch: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward the backbone once with the head replaced by ``Identity``.

    Returns ``(features, labels)`` as CPU tensors. This is the core of efficient
    linear probing: the (expensive) backbone is run a single time and its outputs
    cached, instead of recomputing it every epoch while only the head trains. The
    model's original head is restored before returning.
    """
    head_attr = _head_attr(arch)
    saved_head = getattr(model, head_attr)
    setattr(model, head_attr, nn.Identity())
    model.eval().to(device)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, target, _ in loader:
        out = model(images.to(device))
        features.append(out.detach().cpu())
        labels.append(target)
    setattr(model, head_attr, saved_head)
    return torch.cat(features), torch.cat(labels)
