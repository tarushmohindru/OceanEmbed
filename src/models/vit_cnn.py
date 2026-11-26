"""ViT-CNN: transformer encoder over patch tokens + convolutional skip stem and decoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.models.blocks import DecoderStage, PatchEmbed, SkipStem, TransformerBlock, init_head


class ViTCNN(nn.Module):
    """(B, C_in, Hp, Wp) -> (B, D, Hp, Wp) in normalized units (the caller adds ``base``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_hw: tuple[int, int],
        patch: int = 8,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        stem_base_channels: int = 32,
        use_skip_stem: bool = True,
    ):
        super().__init__()
        levels = int(math.log2(patch))
        assert 2**levels == patch and patch >= 2, "patch must be a power of two >= 2"
        self.in_channels, self.out_channels = in_channels, out_channels
        self.grid_hw, self.patch, self.dim = tuple(grid_hw), patch, dim
        self.use_skip_stem = use_skip_stem
        self.embed = PatchEmbed(in_channels, dim, patch, grid_hw)
        self.embed_drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        skip_ch = [stem_base_channels * 2**i for i in range(levels)]
        self.stem = SkipStem(in_channels, stem_base_channels, levels) if use_skip_stem else None
        dec_out = list(reversed(skip_ch))  # [c_{L-1}, ..., c_0]
        stages, prev = [], dim
        for i, cout in enumerate(dec_out):
            cskip = skip_ch[levels - 1 - i] if use_skip_stem else 0
            stages.append(DecoderStage(prev, cskip, cout))
            prev = cout
        self.decoder = nn.ModuleList(stages)
        self.head = nn.Conv2d(prev, out_channels, kernel_size=1)
        self.apply(self._init)
        init_head(self.head)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        hp, wp = self.grid_hw
        assert x.shape == (b, self.in_channels, hp, wp), (
            f"expected input (B, {self.in_channels}, {hp}, {wp}), got {tuple(x.shape)}"
        )
        th, tw = hp // self.patch, wp // self.patch
        t = self.embed_drop(self.embed(x))
        assert t.shape == (b, th * tw, self.dim), t.shape
        for blk in self.encoder:
            t = blk(t)
        t = self.norm(t)
        z = t.transpose(1, 2).reshape(b, self.dim, th, tw)
        skips = self.stem(x) if self.stem is not None else None
        for i, stage in enumerate(self.decoder):
            skip = skips[len(self.decoder) - 1 - i] if skips is not None else None
            z = stage(z, skip)
        out = self.head(z)
        assert out.shape == (b, self.out_channels, hp, wp), out.shape
        return out

    def no_weight_decay_names(self) -> set[str]:
        return {"embed.pos"}


def _stage_shapes(model: ViTCNN, x: torch.Tensor) -> list[tuple[str, tuple]]:
    """Per-stage output shapes for the summary printout."""
    out = []
    t = model.embed(x)
    out.append(("patch_embed", tuple(t.shape)))
    for blk in model.encoder:
        t = blk(t)
    out.append(("encoder", tuple(model.norm(t).shape)))
    return out
