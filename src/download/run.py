"""Run several downloaders in sequence (small products first so problems surface early)."""

from __future__ import annotations

import sys

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config
from src.common.logging_utils import get_logger
from src.common.paths import Paths

log = get_logger(__name__)
ORDER = ["duacs", "ostia", "oscar", "smap", "ccmp", "glorys", "argo"]


def run_product(cfg: Config, paths: Paths, product: str, years: list[int] | None, force: bool):
    """Dispatch to the downloader module for ``product`` (imported lazily)."""
    if product in ("glorys", "ostia", "duacs"):
        from src.download import copernicus

        copernicus.download_product(cfg, paths, product, years, force)
    elif product in ("oscar", "smap"):
        from src.download import earthdata

        earthdata.download_product(cfg, paths, product, years, force)
    elif product == "ccmp":
        from src.download import ccmp

        ccmp.download_product(cfg, paths, years, force)
    elif product == "argo":
        from src.download import argo_fetch

        for y in years or cfg["products"]["argo"]["fetch_years"]:
            argo_fetch.fetch_year(cfg, paths, y, force)
    else:
        raise ValueError(f"unknown product {product!r}")


def parse_products(values: list[str]) -> list[str]:
    """Accept ``all``, space- or comma-separated names; return them in ``ORDER``."""
    names = [n for v in values for n in v.split(",") if n]
    if "all" in names:
        return list(ORDER)
    unknown = sorted(set(names) - set(ORDER))
    if unknown:
        raise ValueError(f"unknown products: {unknown}")
    return [p for p in ORDER if p in names]


def main() -> None:
    parser = base_parser("Run downloaders in sequence")
    parser.add_argument("--products", nargs="+", default=["all"])
    parser.add_argument("--years", nargs="*", type=int, default=None)
    args, cfg = parse_and_load(parser)
    paths = Paths(cfg)
    failed: dict[str, str] = {}
    for product in parse_products(args.products):
        log.info("=== %s ===", product)
        try:
            run_product(cfg, paths, product, args.years, args.force)
        except Exception as e:  # noqa: BLE001 - keep going, summarize at the end
            log.exception("%s failed", product)
            failed[product] = f"{type(e).__name__}: {e}"
    if failed:
        for p, msg in failed.items():
            log.error("FAILED %s: %s", p, msg)
        sys.exit(1)
    log.info("all downloads finished")


if __name__ == "__main__":
    main()
