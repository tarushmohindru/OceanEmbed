# `data/raw/` — Raw downloads contract

Written only by `src/download/`. Read only by `src/preprocess/regrid.py` (and the GLORYS
reference scorer in `src/evaluation/argo.py`, via interim files, not raw).

## Layout

```text
data/raw/
  manifest.json                      # one record per completed file (see below)
  glorys/<stream>/glorys_<stream>_<YYYY>.nc         # thetao, native 1/12°, native depths 0–1100 m
  ostia/<stream>/ostia_<stream>_<YYYY>.nc           # analysed_sst (K), 0.05°
  duacs/<stream>/duacs_<stream>_<YYYY>.nc           # sla (m)
  oscar/<stream>/oscar_<stream>_<YYYY>.nc           # u, v (m/s); concatenated from cropped daily files
  smap/<stream>/smap_<stream>_<YYYY>.nc             # sss; concatenated from cropped daily files
  ccmp/<stream>/ccmp_<stream>_<YYYY>.nc             # uwnd, vwnd at 6-hourly times (NOT yet daily-averaged)
  argo/argo_<YYYY>.nc                               # profiles from argopy, expert mode, with QC + data mode
  _tmp/                                             # per-day global files during download; emptied after each year
```

`<stream>` is the stream `name` from `config.yaml` (e.g. `my`, `myint`, `rep`, `nrt`, `final`,
`interim`, `rss`, `v3.1`).

## Content rules

- Spatial extent: region box ± `download_buffer_deg` (default 1°), i.e. 4–24°N, 79–101°E.
  The buffer is needed so conservative regridding has full source cells at the box edges.
- Temporal extent per file: one calendar year (or the part of the year that the stream covers).
- Values unchanged: same variables, units, attributes and native grid as the provider.
  Cropping and concatenating along time are the only allowed operations.
- A year file for one stream may be partial (a stream that ends mid-year). The next stream's
  file for that year covers the rest. Overlapping dates between streams are allowed; the
  preprocessing step chooses per date by stream order.
- CCMP keeps its 4 daily times (00, 06, 12, 18 UTC); daily averaging happens in preprocessing.
- SMAP keeps its native time labels (8-day running mean centred on the label date — **Verify**).

## `manifest.json`

A JSON object keyed by relative file path. Each value:

```json
{
  "product": "glorys",
  "stream": "my",
  "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
  "dataset_version": "<from provider metadata or 'unknown'>",
  "variables": ["thetao"],
  "time_start": "2016-01-01",
  "time_end": "2016-12-31",
  "n_times": 366,
  "bbox": [79.0, 4.0, 101.0, 24.0],
  "bytes": 1234567890,
  "sha256": "<of the final file>",
  "downloaded_utc": "2026-11-27T10:00:00Z",
  "tool_versions": {"copernicusmarine": "x.y.z"}
}
```

Updated by `src.download.common.update_manifest()` with a file lock (concurrent downloaders).

## Expected sizes (rough; for planning, not for assertions)

| Product | Native | Size for 7 years, cropped |
| --- | --- | --- |
| GLORYS thetao | 1/12°, ~36 levels to 1100 m, daily | 20–25 GB uncompressed equivalent |
| OSTIA | 0.05°, daily | ~2 GB |
| DUACS | 0.125° or 0.25°, daily | < 1 GB |
| OSCAR | 0.25°, daily | < 1 GB |
| SMAP | 0.25°, daily labels | < 1 GB |
| CCMP | 0.25°, 6-hourly | ~1–2 GB |
| ARGO 2022 | profiles | < 100 MB |

## Done when

`python -m src.download.check_coverage --config config.yaml` reports, for every product,
zero missing dates in `dates.start … dates.end`, and lists which stream provides each date range.
