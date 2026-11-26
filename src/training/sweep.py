"""ENTRY: sequential experiment sets (lambda sweep, core predictions, seeds, ablations).

Finished runs (``summary.json`` present) are skipped, so re-running resumes the set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.cli import base_parser, parse_and_load
from src.common.config import Config, load_config
from src.common.io import write_json
from src.common.logging_utils import get_logger
from src.common.paths import Paths

log = get_logger(__name__)
MODELS = ("cnn_unet", "vit_cnn")
LAMBDAS = (0.0, 0.05, 0.1, 0.3)
ABLATIONS = {
    "patch4": ["model.patch=4"],
    "noskip": ["model.use_skip_stem=false"],
    "depth4": ["model.depth=4"],
    "depth8": ["model.depth=8"],
    "anomaly": ["train.target_mode=anomaly"],
    "w_variance": ["train.depth_weights=variance"],
    "w_thermocline": ["train.depth_weights=thermocline"],
    "landmask": ["extra_inputs.land_mask=true"],
    "no_sss": ["train.drop_inputs=[sss]"],
    "no_cur": ["train.drop_inputs=[u_cur,v_cur]"],
}


def run_one(cfg_path: Path, model: str, name: str, overrides: list[str]) -> dict:
    from src.training.train import train

    cfg = load_config(cfg_path, overrides)
    run_dir = Paths(cfg).results_dir / "runs" / name
    if (run_dir / "summary.json").exists():
        log.info("%s finished, skipping", name)
        return json.loads((run_dir / "summary.json").read_text())
    return train(cfg, model, run_dir, resume=(run_dir / "last.pt").exists())


def lambdas_for(cfg: Config) -> tuple[float, ...]:
    return tuple(cfg.get("train.sweep_lambdas", LAMBDAS))


def main() -> None:
    p = base_parser("Run an experiment set")
    p.add_argument("--set-name", required=True, choices=["lambda", "core", "seeds", "ablations"])
    args, cfg = parse_and_load(p)
    paths = Paths(cfg)
    runs_dir = paths.results_dir / "runs"
    sel_path = runs_dir / "selected.json"
    if args.set_name == "lambda":
        rows = []
        for m in MODELS:
            for lam in lambdas_for(cfg):
                name = f"{m}_lambda{lam:g}"
                s = run_one(cfg.path, m, name, [*args.set, f"train.grad_lambda={lam}"])
                rows.append({"model": m, "lambda": lam, "run": name,
                             "val_rmse_mean": s["best_val_rmse_mean"]})  # fmt: skip
        df = pd.DataFrame(rows)
        df.to_csv(paths.tables_dir / "lambda_sweep.csv", index=False)
        selected = {m: df[df.model == m].sort_values("val_rmse_mean").iloc[0]["run"]
                    for m in MODELS}  # fmt: skip
        write_json(selected, sel_path)
        log.info("selected: %s", selected)
    elif args.set_name == "core":
        from src.training.predict import predict_run

        selected = json.loads(sel_path.read_text())
        for m, run in selected.items():
            predict_run(runs_dir / run, m)
    elif args.set_name == "seeds":
        from src.training.predict import predict_run

        selected = json.loads(sel_path.read_text())
        run_cfg = load_config(runs_dir / selected["vit_cnn"] / "config.yaml")
        for k in (1, 2, 3, 4):
            name = f"vit_cnn_seed{k}"
            ov = [f"seed={k}", f"train.grad_lambda={run_cfg['train']['grad_lambda']}"]
            run_one(cfg.path, "vit_cnn", name, [*args.set, *ov])
            predict_run(runs_dir / name, name)
    else:
        from src.training.predict import predict_run

        selected = json.loads(sel_path.read_text())
        run_cfg = load_config(runs_dir / selected["vit_cnn"] / "config.yaml")
        lam = f"train.grad_lambda={run_cfg['train']['grad_lambda']}"
        for tag, ov in ABLATIONS.items():
            name = f"vit_cnn_{tag}"
            run_one(cfg.path, "vit_cnn", name, [*args.set, lam, *ov])
            predict_run(runs_dir / name, name)


if __name__ == "__main__":
    main()
