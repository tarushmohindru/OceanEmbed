# `notebooks/` — Exploration only

Notebooks are **not part of the pipeline**. Nothing in `src/`, `scripts/` or `tests/` may import
from or depend on a notebook, and no number in the report may come only from a notebook. If a
notebook produces something worth keeping, move the logic into `src/` and call it from there.

## Rules

- Name: `NN_short_topic.ipynb` (e.g. `01_probe_inspection.ipynb`).
- First cell: `%cd ..` to the repo root (or `os.chdir`), then load the config with
  `src.common.config.load_config("config.yaml")` and paths with `src.common.paths.Paths`. Never
  hard-code data paths.
- Use library functions from `src/` for reading and metrics; do not re-implement them.
- Clear outputs before committing (`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`)
  so diffs stay small and no large images are committed.

## Suggested notebooks (create when useful, not required for "done")

| Notebook | Purpose | When |
| --- | --- | --- |
| `01_probe_inspection.ipynb` | Open the probe sample files, look at grids, time labels, units | After `probe.py` |
| `02_raw_quicklook.ipynb` | One day of every raw product on a map; coverage per year | After downloads |
| `03_interim_checks.ipynb` | Compare raw vs regridded fields for one day; SSS gaps near coasts | After regridding |
| `04_processed_eda.ipynb` | Channel distributions, correlation of surface inputs with T at each depth, seasonal cycle per depth | After preprocessing |
| `05_error_analysis.ipynb` | Where and when the ViT-CNN fails (coast, eddies, monsoon onset); worst days | After evaluation |

The `05` analysis can feed the "limitations" section of the write-up, but any number quoted there
must be reproducible by a function in `src/evaluation`.
