"""ENTRY: run directory -> prediction stores (val/test) in degC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.common.cli import base_parser
from src.common.config import load_config
from src.common.io import write_prediction_store
from src.common.logging_utils import configure_logging, get_logger
from src.common.normalization import denormalize_y, load_norm_stats
from src.common.paths import Paths
from src.dataset.arrays import load_split_numpy
from src.dataset.iterator import BatchIterator
from src.dataset.tensors import tensors_from_arrays
from src.models.factory import build_model
from src.training.checkpoint import load_checkpoint
from src.training.precision import autocast_ctx, resolve_amp

log = get_logger(__name__)


def predict_run(run_dir: Path, model_id: str, splits=("val", "test"), checkpoint: str = "best",
                data_root: str | None = None) -> list[Path]:  # fmt: skip
    """Predict with a trained run and write one store per split."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml",
                      [f"paths.data_root={data_root}"] if data_root else None)  # fmt: skip
    paths = Paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = load_checkpoint(run_dir / f"{checkpoint}.pt", device)
    model_name = st["model_name"]
    model = build_model(cfg, model_name, st["in_channels"], len(cfg["depths_m"]),
                        tuple(st["grid_hw_padded"])).to(device)  # fmt: skip
    model.load_state_dict(st["model"])
    model.eval()
    stats = load_norm_stats(paths.norm_stats, cfg)
    amp = resolve_amp(cfg, device)
    outs = []
    for split in splits:
        arrays = load_split_numpy(cfg, paths, split, load_degc=False)
        t = tensors_from_arrays(cfg, arrays, model_name, device)
        it = BatchIterator(t, int(cfg["train"]["batch"]), False, 0, device)
        preds = []
        with torch.no_grad():
            for b in it:
                with autocast_ctx(amp, device):
                    out = model(b["x"])
                pred = (out.float() + b["base"])[..., : t.H, : t.W]
                preds.append(pred.cpu().numpy())
        p = np.concatenate(preds, 0)
        degc = denormalize_y(p, stats, depth_axis=1).astype("f4")
        degc[:, ~arrays.depth_mask] = np.nan
        assert degc.shape == arrays.Y.shape
        out_path = write_prediction_store(
            degc, arrays.dates, cfg, model_id, split, paths,
            {"run_dir": str(run_dir), "checkpoint": str(run_dir / f"{checkpoint}.pt")},
        )  # fmt: skip
        log.info("%s %s -> %s", model_id, split, out_path)
        outs.append(out_path)
    return outs


def main() -> None:
    p = base_parser("Write prediction stores from a trained run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--checkpoint", default="best")
    args = p.parse_args()
    configure_logging(args.log_level)
    predict_run(Path(args.run_dir), args.model_id, args.splits, args.checkpoint)


if __name__ == "__main__":
    main()
