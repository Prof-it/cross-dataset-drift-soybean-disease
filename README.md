# cross-dataset-drift-soybean-disease

Code for two papers on the cross-dataset robustness of soybean disease
classifiers under distribution shift:

- **Paper 1 (ICSIE, agriculture):** cross-dataset transfer, a control study, and
  lightweight interventions (classifier-head refit, calibration corrections).
- **Paper 2 (ETECOM, CV/XAI):** explainability and failure analysis *(added later).*

A single, config-driven PyTorch package. The reusable core (`data`, `models`,
`training`, `evaluation`) is shared by both papers; paper-specific pipelines live
under `src/experiments/`.

## Layout

```
configs/        YAML: hyperparameters (default.yaml) and the dataset registry (datasets.yaml)
src/
  config.py     load/merge YAML into typed dataclasses; seeding; device; paths
  data/         datasets, transforms, splits (+ control-study subsample), dataloaders
  models/       model factory, classifier-head reinit, backbone freeze (linear probing)
  training/     training engine (AMP, plateau LR, early stopping, resumable checkpoints)
  evaluation/   metrics (accuracy, macro F1, ECE), calibration, bootstrap CIs
  viz/          shared figure style and per-class palette (both papers)
  experiments/
    finetune.py train the 16 models (shared by both papers)
    robustness/ Paper 1: control study, interventions, cross-dataset evaluation
  colab.py      Colab helpers (mounting, dataset caching, pre-resize, path resolution)
  utils/        I/O and logging helpers
scripts/        thin command-line entry points
notebooks/      thin Colab drivers that call the package
tests/          model build/forward, metrics sanity, one-epoch smoke test
.env.example    Colab secrets template (copy to .env; .env is gitignored)
```

## Installation

Requires **Python 3.12** (pinned in `pyproject.toml` and `.python-version`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or: pip install -r requirements.txt
```

## Data

Place the raw datasets under `data/raw/` using the folder names in
`configs/datasets.yaml` (`ASDID/`, `MH-SoyaHealthVision/`, `PlantVillage/raw/color/`).
The datasets are public: ASDID (Bevers et al., 2022), MH-SoyaHealthVision (Shinde
et al., 2025), and PlantVillage (Hughes & Salathé, 2016). Raw images are not
version-controlled. The stratified split CSVs (seed 73) and `class_weights.json`
**are** committed as reproducibility inputs; the large PlantVillage split is
regenerated on demand. On Colab the datasets are cached from Drive (see `src/colab.py`).

## Colab setup

Training runs on Google Colab. The run config (token, repo URL, Drive root) is
read from a file on your Drive, so nothing account-specific lives in the repo. The
committed `.env.example` documents the keys.

1. Create the run config on your Drive at
   `MyDrive/cross-dataset-drift-soybean-disease.env` (copy `.env.example` and fill
   it in):

   - `GITHUB_PAT`: a fine-grained, read-only token with Contents:Read on your copy
     of the repo (used to clone it on Colab).
   - `GITHUB_REPO_URL`, `GIT_BRANCH`: where and what to clone.
   - `DRIVE_ROOT`: a folder under your `MyDrive/` holding the dataset zips and the
     persisted artifacts (checkpoints / results / logs).

2. Build the dataset archives once (locally) and upload them to
   `<DRIVE_ROOT>/data/raw/`:

   ```bash
   python scripts/prepare_colab_data.py --out colab_data
   ```

   This writes slim `ASDID.zip`, `MH-SoyaHealthVision.zip`, `PlantVillage.zip`
   containing only the used classes, pre-resized to 512px (ASDID shrinks ~10×),
   each with a `.pre_resized` marker so the Colab pre-resize step is skipped.
   Upload the three zips to `<DRIVE_ROOT>/data/raw/`. This keeps the per-session
   data-prep on Colab to a quick unzip.

3. Open `notebooks/colab_runner.ipynb` (Colab web UI or the VS Code Colab plugin).
   The bootstrap cell reads the Drive config file (then env vars, then Colab
   Secrets), clones the repo, installs deps, caches and pre-resizes the datasets to
   local SSD, and resolves paths so data lives on SSD while artifacts persist on
   Drive.

Never commit `.env` or real tokens.

## Reproducing Paper 1

Install the package first (`pip install -e .`), then:

```bash
python scripts/prepare_data.py --matched                          # splits, class weights, control subsample
python scripts/run_experiment.py configs/experiments/full_finetune.yaml
python scripts/run_experiment.py configs/experiments/control_study.yaml
python scripts/run_experiment.py configs/experiments/linear_probe.yaml
python scripts/run_experiment.py configs/experiments/evaluate.yaml   # within/cross metrics, calibration, decomposition
python scripts/aggregate_results.py                               # headline transfer-gap table
python scripts/make_figures.py                                    # styled PDFs in results/figures/
```

Determinism: the seeds and the data partition are fixed, and every
`(model, seed, split)` run checkpoints independently and is skipped if its
checkpoint already exists, so interrupted (Colab) sessions resume cleanly.
Training itself runs on Colab via `notebooks/colab_runner.ipynb`, which calls this
same package code.

### Train on Colab, evaluate locally

This is the intended split: the runner notebook trains and bundles artifacts to
Drive, and evaluation runs locally. Checkpoints use the same relative layout in
both places (`checkpoints/<experiment>/<model_id>/split<S>/seed<s>/`), so after
training either copy Drive's `checkpoints/` into the repo, or point the evaluator
straight at the Drive-synced folder:

```bash
python scripts/run_experiment.py configs/experiments/evaluate.yaml \
    --checkpoints-dir "/path/to/Drive/<DRIVE_ROOT>/checkpoints"
python scripts/aggregate_results.py
python scripts/make_figures.py
```

## License

MIT. See [LICENSE](LICENSE).
