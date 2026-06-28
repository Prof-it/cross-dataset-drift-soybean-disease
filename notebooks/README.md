# Notebooks

Thin Colab drivers that call the package. No heavy logic lives here; anything
reusable belongs in `src/`.

Planned:

- `colab_runner.ipynb` — setup (mount, sparse clone, install), dataset caching and
  pre-resize, path resolution, running an experiment, and bundling artifacts to Drive.

A single parameterized runner is preferred; add per-experiment notebooks only if
that proves unwieldy.
