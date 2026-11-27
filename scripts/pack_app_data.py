"""Pack what the Streamlit app needs into one zip, to run the app on another machine.

Includes results/predictions/*/<split>.zarr, results/metrics/<split>/ and
data/processed/Y_degC.zarr cut to the days of those splits.
Extract the zip at the repo root of the other copy.

Usage: python scripts/pack_app_data.py --config config.yaml [--splits test val] [--out app_data.zip]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import xarray as xr

    from src.common.config import load_config
    from src.common.paths import Paths

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--splits", nargs="+", default=["test", "val"])
    p.add_argument("--out", default=str(ROOT / "app_data.zip"))
    a = p.parse_args()
    paths = Paths(load_config(a.config, []))
    out = Path(a.out)

    pred = paths.results_dir / "predictions"
    stores = sorted(s for sp in a.splits for s in pred.glob(f"*/{sp}.zarr"))
    if not stores:
        sys.exit("no prediction stores found")
    times = sorted({t for s in stores for t in xr.open_zarr(s)["time"].values})
    with tempfile.TemporaryDirectory(prefix="oe_pack_") as tmp:
        y_sub = Path(tmp) / "Y_degC.zarr"
        y = xr.open_zarr(paths.y_degc_zarr).sel(time=times)
        for v in y.variables.values():
            v.encoding.pop("chunks", None)
            v.encoding.pop("preferred_chunks", None)
        y.chunk({"time": 1}).to_zarr(y_sub, mode="w")
        entries = [(y_sub, Path("data/processed/Y_degC.zarr"))]
        entries += [(s, s.relative_to(ROOT)) for s in stores]
        for sp in a.splits:
            m = paths.metrics_dir / sp
            if m.exists():
                entries.append((m, m.relative_to(ROOT)))
        part = out.with_suffix(".zip.part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:  # chunks already compressed
            for src, arc in entries:
                for f in sorted(src.rglob("*")) if src.is_dir() else [src]:
                    if f.is_file():
                        name = arc / f.relative_to(src) if src.is_dir() else arc
                        z.write(f, name.as_posix())
        shutil.move(part, out)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({mb:.0f} MB, {len(times)} days, {len(stores)} stores)")


if __name__ == "__main__":
    main()
