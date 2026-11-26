# Decisions log (append-only)

## 2026-11-26 — Copernicus Marine catalogue check of configured stream IDs
- **Type:** Verify-result
- **What:** `cmems_mod_glo_phy_myint_0.083deg_P1D-m` (GLORYS interim) and
  `cmems_obs-sl_glo_phy-ssh_myint_allsat-l4-duacs-0.125deg_P1D` (DUACS interim) do not exist in
  the catalogue. OSTIA `rep`/`nrt` and GLORYS `my` and DUACS `my` exist.
- **Why:** `copernicusmarine.describe` raised `DatasetNotFound` during the first probe.
- **Evidence:** `copernicusmarine.describe(contains=[...])` listing on 2026-11-26; first probe log.
- **Impact:** `config.yaml` DUACS second stream → `nrt`
  (`cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D`). GLORYS `myint` left in config;
  it is skipped automatically if `my` covers the whole period (decided after the probe).

## 2026-11-27 — Shortened period and training budget for a 1-day deadline
- **Type:** Deviation
- **What:** `dates.start` 2016-01-01 → 2018-01-01; `splits.train` [2016, 2020] → [2018, 2020]
  (val 2021 and test 2022 unchanged). `train.epochs` 100 → 40, `warmup_epochs` 5 → 3,
  `patience` 15 → 10. λ sweep, extra seeds and ablations are skipped unless time remains.
- **Why:** the user needs a trained model and a Streamlit front end within about one day. Raw
  download volume (GLORYS, CCMP global daily files) scales with the number of years.
- **Evidence:** user instruction, 2026-11-27.
- **Impact:** `config.yaml` (`dates`, `splits`, `train`); tests that hard-coded the 2016 period
  now build the reference 2016–2022 config explicitly (`tests/test_common_grid_time.py`) or read
  the dates from config (`tests/test_download.py`).

## 2026-11-27 — Probe STOP delegated to the agent
- **Type:** Deviation
- **What:** the user authorized the agent to make the probe-review decisions (period, streams,
  stream switches) itself and record them here instead of stopping.
- **Why:** time constraint.
- **Evidence:** user instruction, 2026-11-27.
- **Impact:** `results/probe/APPROVED` is created by the agent after the decisions are logged.
