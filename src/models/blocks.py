"""Building blocks shared by ViTCNN and CNNUNet (pure torch.nn, written by hand)."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class ConvNormAct(nn.Sequential):
    """Conv3x3 (no bias) -> GroupNorm -> GELU."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        g = min(8, cout)
        assert cout % g == 0, f"GroupNorm needs channels ({cout}) divisible by {g}"
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.GELU(),
        )


class TransformerBlock(nn.Module):
    """Pre-norm block: x + attn(LN(x)), then x + mlp(LN(x)). (B, N, dim) -> (B, N, dim)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads, self.dropout = heads, dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(mlp_ratio * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.drop = nn.Dropout(dropout)

    def attn(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, d_head)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(o.transpose(1, 2).reshape(b, n, d))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Conv(patch, stride patch) -> tokens (B, N, dim) + learned position embedding."""

    def __init__(self, cin: int, dim: int, patch: int, grid_hw: tuple[int, int]):
        super().__init__()
        hp, wp = grid_hw
        assert hp % patch == 0 and wp % patch == 0, f"grid {grid_hw} not divisible by {patch}"
        self.grid_hw, self.patch = (hp, wp), patch
        self.tokens_hw = (hp // patch, wp // patch)
        self.proj = nn.Conv2d(cin, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.tokens_hw[0] * self.tokens_hw[1], dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        assert tuple(x.shape[-2:]) == self.grid_hw, (
            f"input grid {tuple(x.shape[-2:])} != model grid {self.grid_hw} (pad the input)"
        )
        t = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        return t + self.pos


class SkipStem(nn.Module):
    """Convolutional skip features at full, 1/2, ..., 1/2^(L-1) resolution."""

    def __init__(self, cin: int, base: int, levels: int):
        super().__init__()
        self.channels = [base * 2**i for i in range(levels)]
        stages = []
        prev = cin
        for i, c in enumerate(self.channels):
            stages.append(nn.Sequential(ConvNormAct(prev, c, stride=1 if i == 0 else 2),
                                        ConvNormAct(c, c)))  # fmt: skip
            prev = c
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        return feats


class DecoderStage(nn.Module):
    """Upsample x2 (bilinear) -> concat skip -> 2 x ConvNormAct."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.cskip = cskip
        self.conv = nn.Sequential(ConvNormAct(cin + cskip, cout), ConvNormAct(cout, cout))

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            assert skip.shape[-2:] == x.shape[-2:], (skip.shape, x.shape)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def init_head(head: nn.Conv2d) -> None:
    """Small initial outputs: head bias 0 and weights scaled by 0.1."""
    with torch.no_grad():
        head.weight.mul_(0.1)
        head.bias.zero_()
