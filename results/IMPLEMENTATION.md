# `results/` — Outputs contract

No code lives here. Every file is produced by code in `src/` (via `scripts/`). Never edit files
here by hand — the only exception is `results/probe/APPROVED` (an empty marker created after the
user reviews the probe report).

## Layout

```text
results/
  env_lock.txt, env_lock_conda.yml        # scripts/setup_lightning.sh
  probe/
    probe.json, probe_report.md            # src/download/probe.py
    coverage.md, coverage.csv              # src/download/check_coverage.py
    APPROVED                               # manual marker after review
    raw/                                   # probe sample files (git-ignored)
  logs/                                    # script and module logs (git-ignored)
  runs/
    <run_name>/
      config.yaml          # resolved config used for this run
      _meta.json           # git hash, GPU, versions, timestamps
      train.log
      metrics.csv          # one row per epoch (src/training/IMPLEMENTATION.md)
      summary.json         # best epoch, best val RMSE, runtime, params
      best.pt, last.pt     # checkpoints (git-ignored; best.pt of selected runs goes in the archive)
    selected.json          # {"vit_cnn": run_name, "cnn_unet": run_name}
  predictions/<model_id>/<split>.zarr      # prediction stores (git-ignored; ~130 MB each)
  metrics/
    val/, test/            # per_depth.csv, summary.csv, per_season.csv, paired_*.csv, rmse_maps.nc, daily_rmse.csv
    argo/                  # argo_profiles_*.nc, argo_matches_*.csv, argo_per_depth_*.csv, argo_summary_*.csv, argo_counts_*.csv
    linreg_alpha_selection.csv
  tables/                  # main_results_*.md/.csv, argo_results_test.md, season_results_test.md,
                           # ablations_test.md, lambda_sweep.csv, model_summary.md
  figures/
    sanity/                # from src/preprocess/sanity.py
    *.png                  # from src/evaluation/plots.py
```

## Rules

- Run names are unique; a run directory is never overwritten (train.py refuses unless `--resume`).
- Every number in `docs/` or the slides cites the file it comes from
  (e.g. "`results/tables/main_results_test.md`").
- Git: tables, figures, metrics CSVs, `summary.json`, `metrics.csv`, run `config.yaml` are
  committed; checkpoints, prediction stores, logs and probe samples are not (see `.gitignore`).
