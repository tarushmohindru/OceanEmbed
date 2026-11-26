# `docs/` — Decisions log, README, write-up, viva notes

Purpose: the human-facing deliverables. Written last (except the decisions log, which is written
throughout). Every number comes from `results/` and cites its file; until real runs exist, use
`TBD`.

## Files

```text
docs/
  decisions_log.md      # start immediately; append-only
  writeup.md            # short method / results / limitations report
  viva_notes.md         # likely examiner questions with short answers
  slides_update.md      # replacement text + figures for deck slides 8–9
README.md               # at the REPO ROOT (not in docs/): how to reproduce each stage
```

## `decisions_log.md` (append-only)

One entry per deviation, verification result or milestone:

```markdown
## 2026-11-27 — <short title>
- **Type:** Deviation | Verify-result | Milestone | Issue
- **What:** ...
- **Why:** ...
- **Evidence:** file path(s), e.g. results/probe/probe_report.md
- **Impact:** which files/config keys changed
```

Required entries: probe outcome and agreed period/streams; any stream switch inside a split; the
measured parameter counts and chosen CNN base; the selected λ for each model; test counts at each
milestone; anything in the `IMPLEMENTATION.md` files that turned out to be wrong.

## Root `README.md`

Sections: project summary (3 sentences); repository map (link every `IMPLEMENTATION.md`);
accounts needed; setup on Lightning (`scripts/setup_lightning.sh`); the runbook (probe → review →
download → coverage → preprocess → baselines → train → evaluate) with the exact commands from
`scripts/`; expected runtimes (**measured**, from logs); where outputs appear; how to run tests;
how to reproduce one figure from a checkpoint; data sources with citations (DOIs/URLs from the
draft guide's Sources section and the probe report).

## `writeup.md`

1. Problem and region (½ page).
2. Data: products, versions/streams actually used (from `provenance.csv`), period, splits, fill
   fractions per channel (from `fill_stats.csv`).
3. Method: preprocessing (regridding, masks, fill, normalization), baselines, ViT-CNN architecture
   (table from `results/tables/model_summary.md`), loss, training recipe, selection protocol.
4. Results: `main_results_test.md`, RMSE-vs-depth figure, maps, season table, ARGO table and
   scatter, paired ViT-vs-CNN CI. State plainly whether the ViT-CNN beats the CNN and whether the
   difference's CI excludes 0.
5. Ablations table.
6. Limitations (at least): one test year (state its climate context, e.g. ENSO/IOD phase, citing a
   source); GLORYS as target (model inherits its errors); ARGO not fully independent of GLORYS
   (assimilation) and representativeness error; derived inputs (OSCAR depends on SLA and winds);
   unweighted cells; same-day mapping without temporal context; any stream switch.
7. Reproducibility: git hash, config hash, env lock.

## `viva_notes.md`

Question → 3–6 sentence answer, grounded in this project's numbers (`TBD` until available).
Cover at least:

1. Why split by year (not by day or pixel)? Autocorrelation; leakage; the 15-day gap.
2. Roles of GLORYS vs ARGO. Dense training target vs sparse independent observations; GLORYS
   assimilates Argo; GLORYS-vs-ARGO as an optimistic reference.
3. Why a ViT? Global receptive field from the first layer, large-scale monsoon/basin patterns;
   why the CNN comparison isolates attention; what we found.
4. What does the gradient loss do? Penalizes wrong horizontal gradients (fronts, eddy edges); masked
   pairs avoid coastline artefacts; effect from the λ sweep.
5. Why per-depth normalization, and its side effect on the loss (variance depth-weight ablation).
6. Why is "0 m" the 0.494 m level? Why potential temperature for ARGO?
7. How are coasts and the shelf handled? Static masks, nested depth masks, no target filling.
8. Why conservative regridding? Area-averaging finer data; NaN-aware `na_thres`.
9. How do you know there is no leakage? Year split, gap, train-only statistics, leakage tests.
10. Why does the model not beat GLORYS against ARGO? It learns GLORYS.
11. Is OSCAR redundant? Derived from SLA + winds; ablation result.
12. Limitations and next steps: more years, temporal context (multi-day input), salinity output,
    uncertainty (seed ensemble spread), full domain.
13. Compute: parameter counts, epoch time on the actual GPU, total training time.

## `slides_update.md`

Replacement content for deck slides 8–9 ("expected outcomes" → real results): the main results
table (trimmed to overall + 3 bands, RMSE and skill), the RMSE-vs-depth figure, the ARGO scatter,
and three bullet findings — each bullet with its source file.

## Done when (final deliverable checklist, from the draft guide)

- [ ] Root `README.md` reproduces each stage with exact commands
- [ ] `config.yaml` and `norm_stats.json` for the final run included in the archive
- [ ] Final model checkpoint(s) in the archive (or a link)
- [ ] Results tables and all figures listed in `src/evaluation/IMPLEMENTATION.md`
- [ ] `writeup.md` (method, results, limitations)
- [ ] `slides_update.md` with real numbers for slides 8–9
- [ ] `viva_notes.md` with every question answered using real numbers
- [ ] `decisions_log.md` complete
