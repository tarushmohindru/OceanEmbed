"""ENTRY: one training run (ViT-CNN or CNN U-Net) with validation-based early stopping."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

from src.common.cli import base_parser
from src.common.config import Config, load_config
from src.common.io import write_json, write_meta
from src.common.logging_utils import configure_logging, get_logger
from src.common.normalization import load_norm_stats
from src.common.paths import Paths
from src.common.seeding import seed_everything
from src.dataset.iterator import BatchIterator
from src.dataset.splits import split_indices
from src.dataset.tensors import build_split_tensors
from src.models.factory import build_model, count_parameters
from src.training.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from src.training.losses import depth_weights, total_loss
from src.training.metrics_live import StreamingDepthRMSE
from src.training.precision import autocast_ctx, make_scaler, resolve_amp
from src.training.schedule import warmup_cosine

log = get_logger(__name__)


def param_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    """No weight decay on biases, norm weights and the position embedding."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".pos") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate_split(model, it: BatchIterator, w, lam, out_std, amp, device) -> dict:
    model.eval()
    rm = StreamingDepthRMSE(out_std)
    losses = []
    for b in it:
        with autocast_ctx(amp, device):
            out = model(b["x"])
        pred = out.float() + b["base"]
        loss, _ = total_loss(pred, b["y"], b["m"], w, lam)
        losses.append(float(loss))
        rm.update(pred, b["y"], b["m"])
    res = rm.compute()
    res["loss"] = float(sum(losses) / max(len(losses), 1))
    return res


def train(cfg: Config, model_name: str, run_dir: Path, resume: bool = False) -> dict:
    """Train one model; returns the summary dict. Writes everything into ``run_dir``."""
    paths = Paths(cfg)
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging("INFO", run_dir / "train.log")
    cfg.dump(run_dir / "config.yaml")
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    write_meta(run_dir, cfg, {"model": model_name, "gpu": gpu, "torch": torch.__version__,
                              "cuda": torch.version.cuda, "python": platform.python_version()})  # fmt: skip  # noqa: E501
    tr = cfg["train"]
    smoke = bool(tr.get("smoke", False))
    split_indices(cfg, paths)  # runs the leakage assertions
    t_train = build_split_tensors(cfg, paths, "train", model_name, device, smoke)
    t_val = build_split_tensors(cfg, paths, "val", model_name, device, smoke)
    it_train = BatchIterator(t_train, tr["batch"], True, int(cfg["seed"]), device)
    it_val = BatchIterator(t_val, tr["batch"], False, int(cfg["seed"]), device)
    stats = load_norm_stats(paths.norm_stats, cfg)
    out_std = torch.as_tensor(stats.out_std, dtype=torch.float64)
    hw = tuple(t_train.x.shape[-2:])
    cin, cout = t_train.x.shape[1], t_train.y.shape[1]
    model = build_model(cfg, model_name, cin, cout, hw).to(device)
    n_params = count_parameters(model)
    log.info("%s: %d parameters, input %s, device %s", model_name, n_params, (cin, *hw), gpu)
    opt = torch.optim.AdamW(param_groups(model, tr["wd"]), lr=tr["lr"], betas=(0.9, 0.999))
    epochs = int(tr["epochs"])
    sched = warmup_cosine(opt, len(it_train), int(tr["warmup_epochs"]), epochs, tr["lr"],
                          tr["min_lr"])  # fmt: skip
    amp = resolve_amp(cfg, device)
    scaler = make_scaler(amp)
    w = depth_weights(cfg, stats).to(device)
    lam = float(tr["grad_lambda"])
    extra = {"model_name": model_name, "in_channels": cin, "channel_names": t_train.channel_names,
             "grid_hw_padded": hw, "param_count": n_params, "grid_hw": (t_train.H, t_train.W)}  # fmt: skip  # noqa: E501

    start_epoch, best, bad = 1, math.inf, 0
    if resume and (run_dir / "last.pt").exists():
        st = load_checkpoint(run_dir / "last.pt", device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"] or {})
        restore_rng(st)
        start_epoch, best, bad = st["epoch"] + 1, st["best_val"], st.get("patience_counter", 0)
        log.info("resumed from epoch %d (best %.4f)", st["epoch"], best)

    metrics_path = run_dir / "metrics.csv"
    depth_cols = [f"val_rmse_{int(z)}" for z in cfg["depths_m"]]
    header = ["epoch", "lr", "train_loss", "train_mse", "train_grad", "val_loss", "val_rmse_mean",
              *depth_cols, "epoch_seconds"]  # fmt: skip
    if not metrics_path.exists() or not resume:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    t0 = time.time()
    best_epoch = None
    for epoch in range(start_epoch, epochs + 1):
        te = time.time()
        model.train()
        it_train.set_epoch(epoch)
        acc = {"loss": 0.0, "mse": 0.0, "grad": 0.0}
        nb = 0
        for b in it_train:
            with autocast_ctx(amp, device):
                out = model(b["x"])
            pred = out.float() + b["base"]
            loss, parts = total_loss(pred, b["y"], b["m"], w, lam)
            if not torch.isfinite(loss):
                torch.save({"batch": b, "model": model.state_dict()}, run_dir / "nan_debug.pt")
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {nb}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k in acc:
                acc[k] += parts[k]
            nb += 1
        val = evaluate_split(model, it_val, w, lam, out_std, amp, device)
        secs = time.time() - te
        row = [epoch, opt.param_groups[0]["lr"], *(acc[k] / max(nb, 1) for k in ("loss", "mse", "grad")),  # noqa: E501
               val["loss"], val["mean_rmse_degC"], *val["rmse_degC"].tolist(), secs]  # fmt: skip
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        improved = val["mean_rmse_degC"] < best - 1e-4
        if improved:
            best, bad, best_epoch = val["mean_rmse_degC"], 0, epoch
            save_checkpoint(run_dir / "best.pt", model=model, optimizer=None, scheduler=None,
                            scaler=None, epoch=epoch, best_val=best, cfg_raw=cfg.raw,
                            extra={**extra, "val_rmse_degC": val["rmse_degC"].tolist()})  # fmt: skip  # noqa: E501
        else:
            bad += 1
        save_checkpoint(run_dir / "last.pt", model=model, optimizer=opt, scheduler=sched,
                        scaler=scaler, epoch=epoch, best_val=best, cfg_raw=cfg.raw,
                        extra={**extra, "patience_counter": bad})  # fmt: skip
        log.info("epoch %d/%d train %.4f val %.4f val RMSE %.3f degC%s (%.1fs)", epoch, epochs,
                 acc["loss"] / max(nb, 1), val["loss"], val["mean_rmse_degC"],
                 " *" if improved else "", secs)  # fmt: skip
        if bad >= int(tr["patience"]):
            log.info("early stopping at epoch %d", epoch)
            break
    best_state = load_checkpoint(run_dir / "best.pt", "cpu")
    summary = {
        "model": model_name,
        "best_epoch": best_state["epoch"] if best_epoch is None else best_epoch,
        "best_val_rmse_mean": best_state["best_val"],
        "best_val_rmse_per_depth": best_state.get("val_rmse_degC"),
        "epochs_run": epoch,
        "total_seconds": time.time() - t0,
        "param_count": n_params,
        "gpu": gpu,
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def main() -> None:
    p = base_parser("Train one model")
    p.add_argument("--model", required=True, choices=["vit_cnn", "cnn_unet"])
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="run directory to resume")
    p.add_argument("--synthetic", action="store_true", help="use a synthetic processed/ dir")
    args = p.parse_args()
    configure_logging(args.log_level)
    cfg = load_config(args.config, args.set)
    if args.synthetic:
        from src.dataset.synthetic import make_synthetic_processed

        tmp = Path(tempfile.mkdtemp(prefix="oceanembed_synth_"))
        cfg = make_synthetic_processed(tmp, cfg)
        log.info("synthetic processed data in %s", tmp)
    paths = Paths(cfg)
    if args.resume:
        run_dir = Path(args.resume)
    else:
        name = args.run_name or f"{args.model}_{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = paths.results_dir / "runs" / name
        if run_dir.exists() and (run_dir / "last.pt").exists():
            sys.exit(f"{run_dir} exists; pass --resume {run_dir} to continue it")
    summary = train(cfg, args.model, run_dir, resume=bool(args.resume))
    log.info("done: %s", json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
