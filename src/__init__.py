"""Cross-dataset robustness of soybean disease classifiers (paper code).

A small, config-driven PyTorch package shared by both papers. The reusable core
(``data``, ``models``, ``training``, ``evaluation``) is independent of any single
experiment; paper-specific pipelines live under :mod:`src.experiments`.
"""

from src.config import (
    Config,
    Data,
    Evaluation,
    Paths,
    Training,
    get_device,
    load_config,
    model_id,
    set_seed,
)

__all__ = [
    "Config",
    "Data",
    "Evaluation",
    "Paths",
    "Training",
    "get_device",
    "load_config",
    "model_id",
    "set_seed",
]
