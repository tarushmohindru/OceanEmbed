# OceanEmbed Implementation Guide (Draft v0.1)

Nov 26, 2026 · @Tarush Mohindru

## About this draft

This is a v0.1 skeleton for building OceanEmbed end to end for an ML lab evaluation. It fixes the main design decisions and leaves code-level detail to a later, deeper guide.

- **Based on:** the project deck *OceanEmbed\_Project\_Presentation\_ViT\_Final.pptx* (10 slides; Tarush Mohindru, Soumil Bhagat). The deck cites problem brief 26066, which was not available while writing this draft.
- **Audience:** the two project students, and an agent that will expand each section into a full step-by-step guide with code.
- **Scope:** a Bay of Bengal proof of concept that runs on a free Colab or Kaggle GPU. The full North Indian Ocean domain is a stretch goal.

Tags used throughout:

| Tag | Meaning |
| --- | --- |
| **Decision** | Fixed by the deck. Change only if the brief says otherwise. |
| **Default** | Recommended choice for v1. Can be changed with a reason. |
| **Verify** | Check against current product docs or the brief before relying on it. |
| **Expand** | Where the next agent should add detail, code or tests. |

## Project summary and scope

OceanEmbed takes one day of 7 surface fields and predicts ocean temperature at 15 depths, from 0 to 1000 m, at every 0.25° grid cell. A Vision Transformer encodes the surface fields and a CNN decoder outputs the 15 depth maps. The model trains on GLORYS reanalysis and is checked separately against ARGO float profiles.

| Item | v1 choice | Tag |
| --- | --- | --- |
| Region | Bay of Bengal box, 5°N–23°N, 80°E–100°E | Default (deck: "Bay of Bengal or Arabian Sea") |
| Full domain (stretch) | 5°N–30°N, 45°E–105°E | Decision |
| Grid | 0.25° × 0.25°, 72 × 80 cells for the box above | Decision |
| Time step | Daily | Decision |
| Period | 2016-01-01 to 2022-12-31 (7 years) | Default; SMAP salinity starts in 2015. Verify that every product covers it, especially OSCAR |
| Inputs (7 channels) | SST, SSS, SLA, current U, current V, wind U, wind V | Decision |
| Output (15 channels) | Temperature at 0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 400, 500, 700, 1000 m | Default. The deck says only "15 levels, 0–1000 m". Verify against the brief |
| Training target | GLORYS daily temperature (`thetao`) | Decision |
| Independent check | ARGO profiles (INCOIS or the Argo GDAC) | Decision |
| Model | ViT encoder + CNN decoder | Decision |
| Baselines | Climatology, linear regression, plain CNN encoder-decoder | Decision |
| Metrics | RMSE, bias, correlation; per depth and per season | Decision |

Using the Arabian Sea instead only changes the region box and the grid size.

## Environment and dependencies

One conda environment with Python 3.11 covers the whole project. Data download and preprocessing can run on a laptop. Training needs a single GPU; a free Colab or Kaggle T4 is enough at this data size.

```bash
conda create -n oceanembed -c conda-forge python=3.11 xarray netcdf4 dask zarr xesmf cartopy
conda activate oceanembed
pip install copernicusmarine earthaccess argopy torch timm scikit-learn matplotlib pyyaml tqdm
```

| Package | Used for |
| --- | --- |
| `copernicusmarine` | Downloading GLORYS, OSTIA and DUACS subsets |
| `earthaccess` | Downloading OSCAR and SMAP from NASA PO.DAAC |
| `argopy` | Fetching ARGO profiles with QC flags |
| `xarray`, `dask`, `netcdf4`, `zarr` | Reading files, lazy processing, storing processed tensors |
| `xesmf` | Horizontal regridding (conservative and bilinear) |
| `torch`, `timm` (optional) | Model, training loop, ready-made transformer blocks |
| `scikit-learn` | Linear regression baseline |
| `matplotlib`, `cartopy` | Maps, depth profiles, error plots |

Accounts (both free):

- **Copernicus Marine:** GLORYS, OSTIA, DUACS. Run `copernicusmarine login` once.
- **NASA Earthdata:** OSCAR and SMAP. Add credentials to `~/.netrc` or use `earthaccess.login()`.
- CCMP winds come from Remote Sensing Systems. The ARGO GDAC and INCOIS LAS need no login. Verify both access routes.

Disk: about 20–50 GB for raw downloads, which can be deleted after preprocessing, and about 1.3 GB for the processed tensors.

## Repository layout and config

Each pipeline stage is its own script, and all stages read one config file. That way each stage can be re-run on its own and the examiners can follow the flow.

```text
oceanembed/
  config.yaml              # region, dates, depths, paths, hyperparameters
  data/
    raw/<product>/         # untouched downloads (git-ignored)
    interim/<product>/     # regridded daily files
    processed/             # X.zarr, Y.zarr, masks.nc, norm_stats.json
  src/
    download/              # one script per product
    preprocess/            # regrid.py, vertical.py, mask.py, fill.py, normalize.py, build_tensors.py
    datasets.py            # PyTorch Dataset + split logic
    models/                # vit_cnn.py, cnn_unet.py
    baselines/             # climatology.py, linreg.py
    losses.py
    train.py
    evaluate.py            # GLORYS test metrics + ARGO matching
    plots.py
  notebooks/               # exploration only, not part of the pipeline
  results/                 # metrics CSVs, figures, checkpoints
```

Sketch of `config.yaml`:

```yaml
region: {lat_min: 5.0, lat_max: 23.0, lon_min: 80.0, lon_max: 100.0}
grid_res: 0.25
dates: {start: 2016-01-01, end: 2022-12-31}
splits: {train: [2016, 2020], val: [2021, 2021], test: [2022, 2022], gap_days: 15}
depths_m: [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 400, 500, 700, 1000]
inputs: [sst, sss, sla, u_cur, v_cur, u_wind, v_wind]
model: {patch: 8, dim: 256, depth: 6, heads: 8, mlp_ratio: 4, dropout: 0.1}
train: {batch: 16, lr: 3.0e-4, wd: 0.05, epochs: 100, patience: 15, grad_lambda: 0.1, seed: 42}
```

**Expand:** a full config schema with validation, and a `Makefile` or `run_all.sh` that chains the stages.

## Phase 1: Data acquisition

Download only the region box plus a 1° buffer, and only the study period, from each product. Full global files waste days of download time. Write each product unchanged to `data/raw/<product>/` and record the product version and download date in a `manifest.json`.

| Variable | Product (from deck) | Access | Native grid | Notes |
| --- | --- | --- | --- | --- |
| SST | OSTIA L4 | `copernicusmarine` | 0.05°, daily | Use the reprocessed (REP) series where it covers the period, near-real-time after. Verify the dataset ID. |
| SSS | SMAP L3/L4 (SMOS as backup) | `earthaccess` | 0.125° per the deck, daily or 8-day running mean | Has gaps near coasts and under heavy rain. Pick one SMAP product (JPL or RSS) and keep it for the whole period. Verify. |
| SLA | DUACS L4 multi-year | `copernicusmarine` | 0.25°, daily | Product family SEALEVEL\_GLO\_PHY\_L4\_MY\_008\_047. Verify. |
| Currents U, V | OSCAR v2.0 | `earthaccess` (PO.DAAC short name OSCAR\_L4\_OC\_FINAL\_V2.0) | 0.25°, daily | The "final" series may end before 2022. If so, add the interim/NRT series or shorten the period. Verify. |
| Winds U, V | CCMP v3.x (or ASCAT) | Remote Sensing Systems download | 0.25°, 6-hourly | Average the four 6-hourly fields to a daily mean. |
| Target T(z) | GLORYS12V1 reanalysis | `copernicusmarine` | 1/12°, 50 levels, daily | Download `thetao` for 0–1100 m only. Dataset ID is likely `cmems_mod_glo_phy_my_0.083deg_P1D-m`. Verify; later years may be in the interim ("myint") dataset. |
| Validation | ARGO profiles | `argopy` (GDAC) or INCOIS LAS | Point profiles | Only for the test year. Keep the QC flags. |

Example subset call (the flag names are from the Copernicus Marine toolbox; verify them against `copernicusmarine subset --help`):

```bash
copernicusmarine subset \
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable thetao \
  --minimum-longitude 79 --maximum-longitude 101 \
  --minimum-latitude 4 --maximum-latitude 24 \
  --minimum-depth 0 --maximum-depth 1100 \
  --start-datetime 2016-01-01 --end-datetime 2022-12-31 \
  --output-directory data/raw/glorys
```

Done when every product has a file for every day of the period. A `check_coverage.py` script should print any missing days for each product.

**Expand:** one download script per product with retries and resume, exact dataset IDs and versions confirmed against the current catalogues, and estimated download sizes.

## Phase 2: Preprocessing and harmonization

The deck flags this phase as the biggest risk to the project. Its output is two daily arrays on one grid: X (7 × 72 × 80) and Y (15 × 72 × 80), plus masks and normalization statistics.

1. **Define the target grid.** Cell centres run from 5.125°N to 22.875°N and from 80.125°E to 99.875°E in 0.25° steps, giving 72 × 80 cells. Save it once as `grid.nc` and regrid every product onto it.
2. **Align time.** Convert every product to daily means on a UTC calendar day. Average CCMP's four 6-hourly fields. Label each day by its date, not by a timestamp.
3. **Regrid horizontally** with `xesmf`. Use conservative regridding for products finer than 0.25° (OSTIA, SMAP, GLORYS) and bilinear for products already at 0.25°. Cache the regridding weights.
4. **Interpolate GLORYS vertically.** Interpolate linearly in depth from GLORYS's native levels to the 15 target depths. Do not extrapolate below the sea floor.
5. **Build the masks.**
   - Ocean mask: cells with valid SST and valid GLORYS temperature at 0 m.
   - Per-depth mask `M_d`: cells where GLORYS has data at depth d. On the shelf the deep levels are land, so the mask shrinks with depth.
6. **Fill input gaps.** SSS has the most gaps. Fill gaps of 3 days or less by linear interpolation in time. Fill anything left with the day-of-year climatology from the training years. Record the fill fraction for each channel. Never fill targets: missing target values stay masked.
7. **Quality control.** Apply range checks (for example SST 15–35 °C, SSS 25–38 PSU), set failures to NaN, then run the gap-fill step again.
8. **Normalize.** Compute the mean and standard deviation of each input channel and each target depth over ocean cells in the training years only. Save them to `norm_stats.json`. Statistics from all years would leak information from the test period.
9. **Write the tensors.** Store X and Y as zarr arrays of shape \[T, C, H, W\] in float32, chunked by day, along with `masks.nc` and a date index. Seven years comes to about 1.3 GB.

Sanity checks before training:

- [ ] Plot one day of each channel and confirm the coastlines line up
- [ ] Plot a GLORYS profile at a few points (the thermocline should sit around 50–150 m)
- [ ] No NaN in X after filling, and no NaN in Y inside `M_d`
- [ ] `norm_stats.json` uses training years only

**Expand:** code for each step, unit tests for grid alignment and masks, and a figure of SSS gap fraction over time.

## Phase 3: Dataset, splits and dataloaders

Split the data by year, not by random day or pixel. Neighbouring days are nearly identical, so random splits leak information between train and test (deck, slide 6).

| Split | Years | Days (approx.) | Share |
| --- | --- | --- | --- |
| Train | 2016–2020 | 1,827 | \~71% |
| Validation | 2021 | 365 | \~14% |
| Test | 2022 | 365 | \~14% |

Drop the first 15 days of the validation and test periods as a buffer, so that slow ocean memory does not carry over from the training period.

The Dataset class returns one day at a time:

- `x`: normalized inputs, shape 7 × 80 × 80. The 72 × 80 grid is padded with zeros to 80 × 80 so that 8 × 8 patches fit evenly.
- `y`: normalized targets, shape 15 × 80 × 80, padded the same way.
- `m`: per-depth mask, shape 15 × 80 × 80, set to 0 on land, padding and missing targets.
- `date`: used for seasonal breakdowns later.

Defaults:

- Batch size 16, shuffle the training set only. Everything fits in RAM, so load the arrays once.
- No flips or rotations. They break physical relationships (the Coriolis effect, coastline geometry).
- Optional extra input channels: sin/cos of day of year, latitude and longitude. Treat these as an ablation, not part of the core 7.

**Expand:** the `datasets.py` code, a leakage test that asserts no date appears in two splits, and an optional sliding-window variant that feeds the last 3 days as input.

## Phase 4: Baselines

Build the baselines before the ViT. The deck's success criterion is beating them, and they give a working evaluation pipeline early. Every baseline outputs the same 15 × H × W prediction and goes through the same `evaluate.py`.

| Baseline | How it works | Why include it |
| --- | --- | --- |
| Climatology | Per cell and depth: the mean for each day of the year over the training years, smoothed with a 31-day window. Ignores the inputs. | The bar any model must clear. Shows how much the satellite inputs actually add. |
| Linear regression | Ridge regression from the 7 surface values in a cell to the 15 depths. Default: one model per cell (5,760 small fits). Alternative: one pooled model with lat/lon features. | Tests whether a simple surface-to-depth mapping is enough, with no spatial context. |
| Plain CNN | A U-Net-style encoder-decoder with about the same parameter count as the ViT-CNN. Same data, loss and training recipe. | The main comparison. Isolates what self-attention adds over convolution. |

Expected order on RMSE: climatology > linear regression > CNN ≥ ViT-CNN. If the ViT-CNN does not beat the CNN, report that honestly; it is still a valid result.

**Expand:** code for each baseline and the CNN's layer-by-layer spec.

## Phase 5: ViT-CNN model

The default model is a small ViT encoder (about 6M parameters) with a U-Net-style CNN decoder. A convolutional skip path brings fine coastal and eddy detail back into the decoder. This matches slide 7: "positional and intermediate features guide spatial reconstruction".

&#91;embedded content: ViT-CNN architecture · 6 stages + skip path\]

Read the diagram top to bottom. The ViT encoder is the one part the plain CNN baseline lacks, so it is what the comparison actually tests.

| Part | Default | Output shape |
| --- | --- | --- |
| Input padding | Zero-pad 72 × 80 to 80 × 80 | 7 × 80 × 80 |
| Patch embedding | Conv2d(7, 256, kernel 8, stride 8) + learned 2D position embedding | 100 × 256 |
| Encoder | 6 pre-norm transformer blocks, 8 heads, MLP width 1024, dropout 0.1 | 100 × 256 |
| Reshape | Tokens back to a grid | 256 × 10 × 10 |
| Skip stem | 2–3 conv layers on the raw input, giving features at 80, 40 and 20 px | 32 / 64 / 128 channels |
| Decoder | 3 × (bilinear upsample ×2, concatenate skip, 2 × \[3×3 conv, norm, GELU\]); channels 256 → 128 → 64 → 32 | 32 × 80 × 80 |
| Head | 1×1 conv, then crop the padding | 15 × 72 × 80 |

- About 1,800 training days is very little data for a transformer. Keep the model this small, rely on dropout and weight decay, and stop early.
- `timm`'s transformer `Block` can provide the encoder blocks; writing them by hand is about 40 lines and easier to explain in the viva.
- Ablations: patch size 4 (400 tokens), the decoder without the skip stem, and encoder depth 4 vs 8.

**Expand:** `vit_cnn.py` with shape assertions after each stage, and a parameter-count printout for both the ViT-CNN and the plain CNN.

## Phase 6: Training loop and loss

Train with a masked, depth-weighted MSE plus a gradient-consistency term (deck, slide 6). Use the same recipe for the plain CNN baseline so the comparison is fair.

```latex
L = \sum_{d=1}^{15} w_d \, \frac{1}{|M_d|} \sum_{(i,j) \in M_d} \left( \hat{T}_{d,i,j} - T_{d,i,j} \right)^2 \; + \; \lambda \sum_{d=1}^{15} \frac{1}{|M_d'|} \sum_{(i,j) \in M_d'} \left( \left| \partial_x \hat{T} - \partial_x T \right| + \left| \partial_y \hat{T} - \partial_y T \right| \right)_{d,i,j}
```

- M\_d is the ocean mask at depth d. M\_d′ keeps only cells where both finite-difference neighbours are ocean, so coastlines do not create false gradients.
- Weights w\_d: targets are already normalized per depth, so start with every w\_d = 1. Try raising the weights on 50–200 m (the thermocline, usually the hardest band) as an experiment.
- λ: default 0.1. Tune on the validation set over {0, 0.05, 0.1, 0.3}. The λ = 0 run doubles as an ablation of the gradient term.

| Setting | Default |
| --- | --- |
| Optimizer | AdamW, learning rate 3e-4, weight decay 0.05 |
| Schedule | 5-epoch linear warmup, then cosine decay |
| Epochs | Up to 100; early stopping after 15 epochs without improvement in validation RMSE |
| Batch size | 16 |
| Precision | Mixed precision (AMP) on GPU |
| Reproducibility | Fixed seed; save the config and git hash with every run |
| Checkpoints | Keep the best checkpoint by validation RMSE, and the last one |
| Logging | Per-epoch train/val loss and per-depth val RMSE to CSV (TensorBoard optional) |

Optional experiment: train the model to predict the anomaly T − climatology instead of T. With fewer than 2,000 training days this often helps, because the model no longer has to learn the seasonal cycle.

**Expand:** `train.py` and `losses.py` code, expected time per epoch on a T4, and a short run on one month of data to check the pipeline before full training.

## Phase 7: Evaluation against GLORYS and ARGO

Evaluation runs in two stages: a dense comparison with withheld GLORYS for 2022, then a sparse check against real ARGO observations. Compute every metric in °C after reversing the normalization, over ocean-masked cells only.

**Metrics** (per depth, then averaged over depths):

- RMSE: typical error size; lower is better.
- Bias: mean of prediction minus truth; closer to 0 is better.
- Pearson correlation: over space and time at each depth; closer to 1 is better.
- Skill score against climatology: 1 − RMSE\_model / RMSE\_climatology. Above 0 means the model beats climatology.

**Breakdowns:**

- By depth: one RMSE-vs-depth curve per model, all on one plot.
- By season, using monsoon seasons rather than calendar quarters: winter monsoon (Dec–Feb), pre-monsoon (Mar–May), summer monsoon (Jun–Sep), post-monsoon (Oct–Nov). Verify that these definitions suit the lab.
- In space: an RMSE map at 0, 100 and 500 m.

**ARGO matching:**

1. Fetch every profile in the region during 2022. Keep only good-QC data (flags 1–2), preferring delayed-mode profiles.
2. Interpolate each profile linearly onto the 15 target depths. Skip a depth if the profile has no measurements within about 20 m of it.
3. Match each profile to its grid cell and day, and take the prediction there.
4. Compute RMSE, bias and correlation for each depth. Report the number of matched profiles.
5. Also score GLORYS itself against the same ARGO profiles. This sets the practical limit: a model trained on GLORYS is unlikely to beat GLORYS against ARGO.

**Figures for the report:**

- [ ] Results table: 4 models × (RMSE, bias, correlation, skill), overall and for 0–100 / 100–300 / 300–1000 m
- [ ] RMSE vs depth, all models
- [ ] Predicted vs GLORYS maps at 3 depths for one sample day, with the difference map
- [ ] Vertical profile comparisons at 3–4 locations
- [ ] ARGO scatter plot (predicted vs observed, coloured by depth)
- [ ] RMSE by monsoon season

**Expand:** `evaluate.py` and `plots.py` code, the ARGO QC rules in detail, and a bootstrap confidence interval on test RMSE.

## Lab evaluation deliverables and milestones

The minimum for a strong lab evaluation is a pipeline that runs end to end, all three baselines, the ViT-CNN, and the results table with figures. Everything beyond that is extra.

| # | Milestone | Done when | Rough effort |
| --- | --- | --- | --- |
| 1 | Data downloaded | `check_coverage.py` shows no missing days | 2–4 days (mostly waiting on downloads) |
| 2 | Tensors built | X/Y zarr files and masks pass the Phase 2 sanity checks | 3–5 days |
| 3 | Evaluation pipeline + climatology | Climatology scored on 2022 with all metrics | 1–2 days |
| 4 | Linear regression + CNN | Both scored by the same `evaluate.py` | 2–3 days |
| 5 | ViT-CNN | Trained, tuned on validation, scored on test | 3–5 days |
| 6 | ARGO validation | Matched profiles and metrics for all models | 2–3 days |
| 7 | Report and viva prep | Figures, results table, README, slides updated with real numbers | 2–3 days |

To hand in:

- [ ] Code repository with a README that explains how to reproduce each stage
- [ ] `config.yaml` and `norm_stats.json` for the final run
- [ ] Final model checkpoint (or a link to it)
- [ ] Results table and the figures listed in Phase 7
- [ ] Short write-up: method, results, limitations
- [ ] Updated deck slides 8–9 with real numbers in place of the expected outcomes

Stretch goals, only after the core is done: ablations (no SSS, no gradient loss, patch size 4 vs 8), the Arabian Sea or the full domain, and uncertainty estimates from an ensemble of seeds.

## Assumptions, open questions and risks

The biggest unknowns are what brief 26066 specifies (exact depths, region, required outputs) and whether every product covers 2016–2022. Resolve both before writing code.

Open questions:

- [ ] Does the brief list the 15 depth levels? If so, replace the default list.
- [ ] Is the region box fixed by the brief or the lab?
- [ ] Does the lab expect uncertainty estimates, or only point predictions?
- [ ] Should ARGO come as raw profiles (GDAC) or gridded fields (INCOIS LAS)? This draft assumes raw profiles.
- [ ] Is there a required report format, compute budget or deadline for the lab evaluation?

| Risk | Impact | Mitigation |
| --- | --- | --- |
| OSCAR "final" series ends before 2022 | Missing current inputs for the test year | Add the interim series, or shift the period (for example 2015-04 to 2021) |
| SMAP SSS gaps near coasts and in rain | Noisy salinity channel | Gap-fill rules in Phase 2; ablation without SSS |
| ViT overfits fewer than 2,000 training days | No gain over the CNN | Keep the model small, use dropout and weight decay, try anomaly targets |
| Leakage between splits | Test scores look better than they are | Year-based split, 15-day buffer, training-only normalization, leakage unit test |
| Shelf cells have no deep data | Errors or NaN in the loss at depth | Per-depth masks throughout the loss and metrics |
| GLORYS differs from ARGO | Model inherits GLORYS errors | Report GLORYS-vs-ARGO as the reference limit |
| Download time and disk space | Schedule slips | Subset by region and time; start downloads first |

## Instructions for the expanding agent

Turn each phase into a guide that a student can follow step by step, with complete, runnable code. Keep every **Decision**, keep every **Default** unless you give a reason, and resolve every **Verify** and **Expand** tag.

1. **Check first.** Confirm the current dataset IDs, versions, date coverage and access methods for OSTIA, SMAP, DUACS, OSCAR, CCMP, GLORYS and ARGO, and cite the pages you used. If a product does not cover 2016–2022, propose a new period and update the splits.
2. **Expand in pipeline order:** config, download, preprocessing, dataset, baselines, model, training, evaluation, plots. Each phase needs: purpose, inputs and outputs (files and shapes), complete code, a command to run it, expected runtime, and a "done when" check.
3. **Add tests:** grid alignment, mask shapes, no NaN inside masks, no date in two splits, model output shape, and a loss value on a tiny synthetic batch.
4. **Respect the constraints:** two students, a free single GPU (Colab or Kaggle T4, sessions of roughly 12 hours), Python 3.11, a few weeks.
5. **Add a viva section** with likely examiner questions and short answers: why split by time, GLORYS vs ARGO roles, why a ViT, what the gradient loss does, and the limitations.
6. **Do not invent results.** Leave placeholders for numbers until real runs produce them.

## Sources

- OceanEmbed project deck, *OceanEmbed\_Project\_Presentation\_ViT\_Final.pptx* (supplied by the user)
- [GLORYS reanalysis](https://doi.org/10.48670/moi-00021)
- [OSTIA SST](https://doi.org/10.48670/moi-00168)
- [DUACS sea level](https://doi.org/10.48670/moi-00145)
- [OSCAR v2.0 currents](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0)
- [Argo Program](https://argo.ucsd.edu/about/) and [INCOIS LAS](https://las.incois.gov.in)
