"""Target grid construction and coordinate standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from src.common.config import Config

_EARTH_RADIUS_KM = 6371.0


def regular_bounds(centres: np.ndarray) -> np.ndarray:
    """Cell edges (N+1,) for a regular 1-D grid of centres (N,).

    Raises ``ValueError`` if the spacing is not uniform (relative tolerance 1e-3, which
    tolerates float32 coordinates such as GLORYS 1/12 deg near 100E).
    """
    c = np.asarray(centres, dtype="float64")
    if c.ndim != 1 or c.size < 2:
        raise ValueError("need a 1-D array with at least 2 centres")
    d = np.diff(c)
    step = d.mean()
    if step == 0 or np.any(np.abs(d - step) > 1e-3 * abs(step)):
        raise ValueError("grid spacing is not uniform")
    mids = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - step / 2], mids, [c[-1] + step / 2]])


@dataclass(frozen=True, eq=False)
class Grid:
    """Regular lat-lon grid. ``lat`` and ``lon`` are ascending cell centres in degrees."""

    lat: np.ndarray  # (H,)
    lon: np.ndarray  # (W,)
    res: float

    @property
    def H(self) -> int:  # noqa: N802
        return int(self.lat.size)

    @property
    def W(self) -> int:  # noqa: N802
        return int(self.lon.size)

    def lat_bounds(self) -> np.ndarray:
        """(H+1,) cell edges in degrees."""
        return regular_bounds(self.lat)

    def lon_bounds(self) -> np.ndarray:
        """(W+1,) cell edges in degrees."""
        return regular_bounds(self.lon)

    def cell_area_km2(self) -> np.ndarray:
        """(H, W) spherical cell area in km^2 (R = 6371 km)."""
        lat_b = np.deg2rad(self.lat_bounds())
        dlon = np.deg2rad(self.res)
        band = _EARTH_RADIUS_KM**2 * dlon * (np.sin(lat_b[1:]) - np.sin(lat_b[:-1]))
        return np.repeat(band[:, None], self.W, axis=1)

    def index_of(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices ``(i_lat, i_lon)`` of points; both are -1 where the point is outside."""
        lat = np.asarray(lat, dtype="float64")
        lon = np.asarray(lon, dtype="float64")
        i = np.floor((lat - self.lat_bounds()[0]) / self.res + 1e-9).astype(int)
        j = np.floor((lon - self.lon_bounds()[0]) / self.res + 1e-9).astype(int)
        outside = (i < 0) | (i >= self.H) | (j < 0) | (j >= self.W) | ~np.isfinite(lat + lon)
        return np.where(outside, -1, i), np.where(outside, -1, j)

    def to_dataset(self) -> xr.Dataset:
        """Dataset with ``lat``/``lon`` coords, ``lat_bnds``, ``lon_bnds``, ``cell_area_km2``."""
        lb, ob = self.lat_bounds(), self.lon_bounds()
        return xr.Dataset(
            {
                "lat_bnds": (("lat", "bnds"), np.stack([lb[:-1], lb[1:]], axis=1)),
                "lon_bnds": (("lon", "bnds"), np.stack([ob[:-1], ob[1:]], axis=1)),
                "cell_area_km2": (("lat", "lon"), self.cell_area_km2()),
            },
            coords={"lat": self.lat, "lon": self.lon},
        )

    def padded_shape(self, multiple: int) -> tuple[int, int]:
        """(H, W) rounded up to a multiple of ``multiple``."""
        return (-(-self.H // multiple) * multiple, -(-self.W // multiple) * multiple)


def grid_from_config(cfg: Config) -> Grid:
    """Build the target grid from ``region`` and ``grid_res``."""
    r, res = cfg["region"], float(cfg["grid_res"])
    h = round((r["lat_max"] - r["lat_min"]) / res)
    w = round((r["lon_max"] - r["lon_min"]) / res)
    lat = np.round(r["lat_min"] + res / 2 + res * np.arange(h), 6)
    lon = np.round(r["lon_min"] + res / 2 + res * np.arange(w), 6)
    return Grid(lat=lat, lon=lon, res=res)


_RENAMES = {
    "latitude": "lat", "LATITUDE": "lat", "nav_lat": "lat", "Latitude": "lat",
    "longitude": "lon", "LONGITUDE": "lon", "nav_lon": "lon", "Longitude": "lon",
}  # fmt: skip


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename to ``lat``/``lon``, wrap lon to -180..180, sort ascending, squeeze surface depth."""
    rename = {
        old: new
        for old, new in _RENAMES.items()
        if (old in ds.variables or old in ds.dims) and new not in ds.variables
    }
    if rename:
        ds = ds.rename(rename)
    # 1-D lat/lon coordinates living on differently named dims (e.g. OSCAR: lat(latitude)).
    for name in ("lat", "lon"):
        if name in ds.variables and ds[name].ndim == 1 and ds[name].dims[0] != name:
            dim = ds[name].dims[0]
            ds = ds.swap_dims({dim: name})
            if dim in ds.variables:
                ds = ds.drop_vars(dim)
    if "lon" in ds.coords:
        wrapped = ((ds["lon"] + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(lon=wrapped.astype(ds["lon"].dtype))
        ds = ds.sortby("lon")
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "depth" in ds.dims and ds.sizes["depth"] == 1:
        ds = ds.squeeze("depth", drop=True)
    return ds
