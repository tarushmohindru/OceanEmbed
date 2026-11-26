"""Print shapes and parameter counts of both models; write results/tables/model_summary.md."""

from __future__ import annotations

import sys

import torch

from src.common.cli import base_parser, parse_and_load
from src.common.grid import grid_from_config
from src.common.paths import Paths
from src.dataset.tensors import pad_multiple_for
from src.models.factory import build_model, count_parameters


def main() -> None:
    _, cfg = parse_and_load(base_parser("Model summary"))
    grid = grid_from_config(cfg)
    cin = len(cfg["inputs"]) - len(cfg["train"].get("drop_inputs", []))
    cout = len(cfg["depths_m"])
    lines = ["# Model summary", ""]
    counts = {}
    for name in ("vit_cnn", "cnn_unet"):
        mult = pad_multiple_for(cfg, name)
        hw = (-(-grid.H // mult) * mult, -(-grid.W // mult) * mult)
        model = build_model(cfg, name, cin, cout, hw).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, cin, *hw))
        n = count_parameters(model)
        counts[name] = n
        lines += [f"## {name}", "", f"- input (2, {cin}, {hw[0]}, {hw[1]}) -> output "
                  f"{tuple(out.shape)}", f"- parameters: {n:,}", "", "| module | params |",
                  "|---|---|"]  # fmt: skip
        for child_name, child in model.named_children():
            lines.append(f"| {child_name} | {count_parameters(child):,} |")
        if name == "cnn_unet":
            lines.append(f"\nCNN base channels: {model.enc0[0][0].out_channels}")
        lines.append("")
    lines.append(
        f"Parameter ratio cnn_unet / vit_cnn: {counts['cnn_unet'] / counts['vit_cnn']:.3f}"
    )
    text = "\n".join(lines) + "\n"
    out_path = Paths(cfg).tables_dir / "model_summary.md"
    out_path.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
