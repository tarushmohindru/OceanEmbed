"""Every file path in the project. No other module builds paths under data/ or results/."""

from __future__ import annotations

from pathlib import Path

from src.common.config import Config

# Input channel -> interim file group.
CHANNEL_GROUP: dict[str, str] = {
    "sst": "sst",
    "sss": "sss",
    "sla": "sla",
    "u_cur": "cur",
    "v_cur": "cur",
    "u_wind": "wind",
    "v_wind": "wind",
}


def _dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


class Paths:
    """Resolve project paths from a ``Config``. Directory properties are created on access."""

    def __init__(self, cfg: Config, repo_root: Path | None = None):
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        self._data = self._resolve(cfg["paths"]["data_root"])
        self._results = self._resolve(cfg["paths"]["results_root"])

    def _resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.repo_root / p

    # ---------------- data directories ----------------
    @property
    def raw_dir(self) -> Path:
        return _dir(self._data / "raw")

    @property
    def interim_dir(self) -> Path:
        return _dir(self._data / "interim")

    @property
    def processed_dir(self) -> Path:
        return _dir(self._data / "processed")

    @property
    def results_dir(self) -> Path:
        return _dir(self._results)

    # ---------------- raw ----------------
    def raw_product_dir(self, product: str, stream: str) -> Path:
        return _dir(self.raw_dir / product / stream)

    def raw_year_file(self, product: str, stream: str, year: int) -> Path:
        """e.g. ``raw/glorys/my/glorys_my_2016.nc``."""
        return self.raw_product_dir(product, stream) / f"{product}_{stream}_{year}.nc"

    def raw_tmp_dir(self) -> Path:
        return _dir(self.raw_dir / "_tmp")

    def manifest(self) -> Path:
        return self.raw_dir / "manifest.json"

    # ---------------- interim ----------------
    def weights_file(self, product: str, method: str) -> Path:
        return _dir(self.interim_dir / "_weights") / f"{product}_{method}.nc"

    def interim_channel_file(self, group: str, year: int) -> Path:
        """``group`` in {sst, sss, sla, cur, wind}."""
        return _dir(self.interim_dir / group) / f"{group}_{year}.nc"

    def interim_glorys(self, year: int) -> Path:
        return _dir(self.interim_dir / "glorys") / f"glorys_{year}.nc"

    def interim_glorys_t15(self, year: int) -> Path:
        return _dir(self.interim_dir / "glorys_t15") / f"glorys_t15_{year}.nc"

    def interim_provenance(self, product: str, year: int) -> Path:
        return _dir(self.interim_dir / "provenance") / f"{product}_{year}.csv"

    def clim_inputs(self) -> Path:
        return self.interim_dir / "clim_inputs.nc"

    def filled_x(self, year: int) -> Path:
        return _dir(self.interim_dir / "filled") / f"X_raw_{year}.nc"

    @property
    def interim_grid(self) -> Path:
        return self.interim_dir / "grid.nc"

    # ---------------- processed ----------------
    @property
    def grid(self) -> Path:
        return self.processed_dir / "grid.nc"

    @property
    def masks(self) -> Path:
        return self.processed_dir / "masks.nc"

    @property
    def x_zarr(self) -> Path:
        return self.processed_dir / "X.zarr"

    @property
    def y_zarr(self) -> Path:
        return self.processed_dir / "Y.zarr"

    @property
    def y_degc_zarr(self) -> Path:
        return self.processed_dir / "Y_degC.zarr"

    @property
    def clim_y(self) -> Path:
        return self.processed_dir / "climatology_Y.nc"

    @property
    def norm_stats(self) -> Path:
        return self.processed_dir / "norm_stats.json"

    @property
    def provenance(self) -> Path:
        return self.processed_dir / "provenance.csv"

    @property
    def fill_stats(self) -> Path:
        return self.processed_dir / "fill_stats.csv"

    @property
    def dates_csv(self) -> Path:
        return self.processed_dir / "dates.csv"

    # ---------------- results ----------------
    def run_dir(self, run_name: str) -> Path:
        return _dir(self.results_dir / "runs" / run_name)

    def prediction_store(self, model_id: str, split: str) -> Path:
        return _dir(self.results_dir / "predictions" / model_id) / f"{split}.zarr"

    @property
    def metrics_dir(self) -> Path:
        return _dir(self.results_dir / "metrics")

    @property
    def figures_dir(self) -> Path:
        return _dir(self.results_dir / "figures")

    @property
    def tables_dir(self) -> Path:
        return _dir(self.results_dir / "tables")

    @property
    def probe_dir(self) -> Path:
        return _dir(self.results_dir / "probe")

    @property
    def logs_dir(self) -> Path:
        return _dir(self.results_dir / "logs")
