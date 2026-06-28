"""Experiment: bidirectional evaluation and result assembly (Paper 1).

Turns trained checkpoints into the numbers the paper reports. For each finetune
model it computes within- and cross-dataset metrics, calibration before and after
temperature scaling, and the prevalence-vs-conditional decomposition of the gap
(feedback #12). The same pass also scores the training-time ablations stored in
their own namespaces (``finetune_unweighted``, ``finetune_label_smoothing``),
tagged in the ``experiment`` column so they are directly comparable to the
baseline. It then evaluates the control-study models (ASDID-matched -> MH) and the
head-refit intervention, and writes tidy CSVs aggregated over seeds and split
seeds (the variance bands, feedback #5/#10).

Outputs (under ``paths.results_dir``):
- ``eval_results.csv``       one row per (experiment, model, seed, split, direction)
- ``eval_aggregated.csv``    mean/std/n over seeds and split seeds
- ``decomposition.csv``      prevalence/conditional split of the cross-dataset gap
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from src.config import get_device, model_id
from src.data.loaders import make_dataloaders
from src.data.splits import build_matched_subsample, build_splits
from src.evaluation.bootstrap import aggregate_over_seeds
from src.evaluation.calibration import apply_temperature, fit_temperature
from src.evaluation.decomposition import decompose_error
from src.evaluation.metrics import (
    collect_predictions,
    compute_metrics,
    expected_calibration_error,
    reliability_curve,
    softmax,
)
from src.experiments._common import run_dir, seed_pairs
from src.models.factory import create_model
from src.utils.io import ensure_dir

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

_METRIC_COLS = ["accuracy", "macro_f1", "ece", "ece_temp"]

# Fine-tune namespaces to evaluate. The baseline plus the two training-time
# interventions, each stored under its own ``run_name`` (see finetune.run). Rows
# are tagged with the namespace in the ``experiment`` column so the baseline and
# the ablations are directly comparable. Overridable via the evaluate YAML
# (``evaluate.finetune_run_names``); namespaces that were not trained are skipped.
DEFAULT_FINETUNE_RUNS = ("finetune", "finetune_unweighted", "finetune_label_smoothing")


def run(cfg: "Config") -> None:
    device = get_device()
    # Rebuild every split this pass consumes (deterministic from the raw data and
    # seed, so identical to what training used). The matched subsample is needed by
    # the control-study eval and is not produced by the plain build_splits loop.
    for split_seed in cfg.split_seeds:
        for dataset in cfg.source_datasets:
            build_splits(cfg, dataset, split_seed)
        build_matched_subsample(cfg, split_seed)

    eval_cfg = cfg.extras.get("evaluate", {})
    finetune_runs = tuple(eval_cfg.get("finetune_run_names", DEFAULT_FINETUNE_RUNS))
    rows: list[dict] = []
    decomposition: list[dict] = []
    reliability: list[dict] = []
    failures: list[str] = []
    # Resume: carry forward already-scored models from existing CSVs and only
    # compute the ones that are missing (e.g. checkpoints that were corrupt last
    # time and have since been re-trained). Off by default (full recompute).
    done: set = _load_existing(cfg, rows, decomposition, reliability) if eval_cfg.get("resume") else set()
    if done:
        logger.info("resume: %d model-runs already scored; computing only the missing ones", len(done))
    for split_seed, seed in seed_pairs(cfg):
        for arch in cfg.architectures:
            for run_name in finetune_runs:
                _eval_finetune(cfg, run_name, arch, split_seed, seed, device, rows, decomposition, reliability, failures, done)
            _eval_control(cfg, arch, split_seed, seed, device, rows, failures, done)
            _eval_head_refit(cfg, arch, split_seed, seed, device, rows, failures, done)

    if failures:
        logger.error("%d checkpoint(s) could not be loaded (corrupt/truncated or Drive read error); "
                     "delete the run dir and re-train, or re-run eval if it was transient:", len(failures))
        for path in failures:
            logger.error("  %s", path)
    _write(cfg, rows, decomposition, reliability)


# --------------------------------------------------------------------------- #
# Per-experiment passes                                                       #
# --------------------------------------------------------------------------- #
def _eval_finetune(cfg, run_name, arch, split_seed, seed, device, rows, decomposition, reliability, failures, done) -> None:
    # Skip a whole namespace silently if it was never trained (e.g. an ablation
    # the user chose not to run), rather than warning on every missing checkpoint.
    if not (cfg.paths.checkpoints_dir / run_name).exists():
        return
    for path in cfg.training_paths:
        for train_ds in cfg.source_datasets:
            mid = model_id(arch, path, train_ds)
            if (run_name, mid, split_seed, seed) in done:  # resume: already scored
                continue
            ckpt = run_dir(cfg, run_name, mid, split_seed, seed) / "best_model.pt"
            if not ckpt.exists():
                logger.warning("eval: missing %s; skipping", ckpt)
                continue
            model = _try_load(cfg, arch, cfg.data.num_classes, ckpt, device, failures)
            if model is None:
                continue
            within = cross = None
            for eval_ds in cfg.source_datasets:
                scalars, per_class_err, prevalence, bins = _evaluate(cfg, model, eval_ds, split_seed, seed, device)
                direction = "within" if eval_ds == train_ds else "cross"
                rows.append(_row(scalars, run_name, arch, path, train_ds, eval_ds, direction, seed, split_seed, mid))
                for b in bins:
                    reliability.append({
                        **b, "experiment": run_name, "arch": arch, "path": path, "train_dataset": train_ds,
                        "eval_dataset": eval_ds, "direction": direction,
                        "seed": seed, "split_seed": split_seed,
                    })
                if direction == "within":
                    within = (per_class_err, prevalence)
                else:
                    cross = (per_class_err, prevalence)
            if within and cross:
                d = decompose_error(within[0], cross[0], within[1], cross[1])
                decomposition.append({
                    **d, "experiment": run_name, "arch": arch, "path": path, "train_dataset": train_ds,
                    "seed": seed, "split_seed": split_seed,
                })


def _eval_control(cfg, arch, split_seed, seed, device, rows, failures, done) -> None:
    other = [d for d in cfg.source_datasets if d != "asdid"]
    target = other[0] if other else "mh"
    for path in cfg.training_paths:
        mid = model_id(arch, path, "asdid_matched")
        if ("control_study", mid, split_seed, seed) in done:  # resume: already scored
            continue
        ckpt = run_dir(cfg, "control_study", mid, split_seed, seed) / "best_model.pt"
        if not ckpt.exists():
            continue
        model = _try_load(cfg, arch, cfg.data.num_classes, ckpt, device, failures)
        if model is None:
            continue
        for eval_ds, direction in (("asdid_matched", "within"), (target, "cross")):
            scalars, _, _, _ = _evaluate(cfg, model, eval_ds, split_seed, seed, device)
            rows.append(_row(scalars, "control_study", arch, path, "asdid_matched",
                             eval_ds, direction, seed, split_seed, mid))


def _eval_head_refit(cfg, arch, split_seed, seed, device, rows, failures, done) -> None:
    for source in cfg.source_datasets:
        for target in (d for d in cfg.source_datasets if d != source):
            mid = f"{arch}_headrefit_from_{source}_{target}"
            if ("linear_probe", mid, split_seed, seed) in done:  # resume: already scored
                continue
            ckpt = run_dir(cfg, "linear_probe", mid, split_seed, seed) / "best_model.pt"
            if not ckpt.exists():
                continue
            model = _try_load(cfg, arch, cfg.data.num_classes, ckpt, device, failures)
            if model is None:
                continue
            scalars, _, _, _ = _evaluate(cfg, model, target, split_seed, seed, device)
            rows.append(_row(scalars, "linear_probe", arch, "headrefit", source,
                             target, "cross", seed, split_seed, mid))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_existing(cfg, rows: list, decomposition: list, reliability: list) -> set:
    """Carry forward results from a previous run for resume mode.

    Reads the existing CSVs under ``results_dir`` into the in-memory lists and
    returns the set of ``(experiment, model_id, split_seed, seed)`` already scored,
    so :func:`run` recomputes only the models that are missing (and merges).
    """
    results_dir = cfg.paths.results_dir
    done: set = set()
    eval_csv = results_dir / "eval_results.csv"
    if eval_csv.exists():
        df = pd.read_csv(eval_csv)
        rows.extend(df.to_dict("records"))
        for r in df[["experiment", "model_id", "split_seed", "seed"]].itertuples(index=False):
            done.add((str(r.experiment), str(r.model_id), int(r.split_seed), int(r.seed)))
    for name, sink in (("decomposition.csv", decomposition), ("reliability.csv", reliability)):
        path = results_dir / name
        if path.exists():
            sink.extend(pd.read_csv(path).to_dict("records"))
    return done


def _load(cfg, arch, num_classes, ckpt: Path, device) -> torch.nn.Module:
    model = create_model(arch, num_classes, cfg.architectures, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device)


def _try_load(cfg, arch, num_classes, ckpt: Path, device, failures: list) -> "torch.nn.Module | None":
    """Load a checkpoint, returning ``None`` (and recording it) if it is unreadable.

    A truncated/corrupt ``best_model.pt`` (e.g. an interrupted save) or a transient
    Drive read error should not abort the whole evaluation; skip that model and
    report it so it can be re-trained or the eval re-run.
    """
    try:
        return _load(cfg, arch, num_classes, ckpt, device)
    except Exception as exc:
        logger.error("eval: failed to load %s (%s: %s); skipping", ckpt, type(exc).__name__, exc)
        failures.append(str(ckpt))
        return None


def _evaluate(cfg, model, eval_dataset, split_seed, seed, device):
    """Evaluate on ``eval_dataset``; return (scalars, per-class error, prevalence, reliability bins)."""
    loaders = make_dataloaders(cfg, eval_dataset, split_seed, seed)
    test = collect_predictions(model, loaders["test"], device)
    val = collect_predictions(model, loaders["val"], device)
    n_bins = cfg.evaluation.ece_bins

    metrics = compute_metrics(test.y_true, test.y_pred, cfg.data.class_names)
    ece = expected_calibration_error(test.y_true, test.probs, n_bins)
    temperature = fit_temperature(val.logits, val.y_true)
    ece_temp = expected_calibration_error(
        test.y_true, softmax(apply_temperature(test.logits, temperature)), n_bins
    )
    scalars = {**metrics, "ece": ece, "ece_temp": ece_temp, "temperature": temperature}

    curve = reliability_curve(test.y_true, test.probs, n_bins)
    bins = [
        {
            "bin": i,
            "bin_accuracy": float(curve["bin_accuracies"][i]),
            "bin_confidence": float(curve["bin_confidences"][i]),
            "bin_count": int(curve["bin_counts"][i]),
        }
        for i in range(n_bins)
    ]
    k = cfg.data.num_classes
    return scalars, _per_class_error(test.y_true, test.y_pred, k), _prevalence(test.y_true, k), bins


def _per_class_error(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list[float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = []
    for c in range(num_classes):
        mask = y_true == c
        out.append(float((y_pred[mask] != c).mean()) if mask.any() else 0.0)
    return out


def _prevalence(y_true: np.ndarray, num_classes: int) -> list[float]:
    y_true = np.asarray(y_true)
    n = len(y_true)
    return [float((y_true == c).sum()) / n if n else 0.0 for c in range(num_classes)]


def _row(scalars, experiment, arch, path, train_dataset, eval_dataset, direction, seed, split_seed, mid) -> dict:
    return {
        "experiment": experiment, "arch": arch, "path": path,
        "train_dataset": train_dataset, "eval_dataset": eval_dataset, "direction": direction,
        "seed": seed, "split_seed": split_seed, "model_id": mid, **scalars,
    }


def _write(cfg, rows, decomposition, reliability) -> None:
    results_dir = ensure_dir(cfg.paths.results_dir)
    if not rows:
        logger.warning("no checkpoints evaluated; nothing written")
        return
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "eval_results.csv", index=False)
    group = ["experiment", "arch", "path", "train_dataset", "eval_dataset", "direction"]
    aggregate_over_seeds(df, group, _METRIC_COLS).to_csv(results_dir / "eval_aggregated.csv", index=False)
    if decomposition:
        pd.DataFrame(decomposition).to_csv(results_dir / "decomposition.csv", index=False)
    if reliability:
        pd.DataFrame(reliability).to_csv(results_dir / "reliability.csv", index=False)
    logger.info("wrote results to %s", results_dir)
