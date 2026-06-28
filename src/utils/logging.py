"""Logging helpers.

Goal: consistent console logging and a thin per-epoch metrics logger that writes
both a JSON training log and TensorBoard scalars, so every run is inspectable.

Will provide:
- ``get_logger(name)``: a configured ``logging.Logger`` with a uniform format.
- ``TrainingLogger(log_dir, run_id)``: record per-epoch train/val loss, macro F1,
  and learning rate to TensorBoard and accumulate the JSON training log.
"""
