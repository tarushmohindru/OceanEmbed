# `src/models/` — ViT-CNN and the plain CNN baseline

Purpose: the two neural networks. Both map `x` (B, C_in, Hp, Wp) → `out` (B, D, Hp, Wp) in
normalized units (the caller adds `base`, see `src/dataset/IMPLEMENTATION.md`). Pure
`torch.nn`; no disk access, no config file reading (they receive plain arguments).

Write the transformer blocks by hand (no `timm`): ~40 lines, and every line must be explainable
in the viva.

## Files

```text
src/models/
  __init__.py      # re-export: build_model, count_parameters
  blocks.py        # ConvNormAct, TransformerBlock, PatchEmbed, SkipStem, DecoderStage
  vit_cnn.py       # ViTCNN
  cnn_unet.py      # CNNUNet
  factory.py       # build_model(cfg, in_channels, out_channels) + param-count matching
  summary.py       # ENTRY: prints layer-by-layer shapes and parameter counts for both models
```

---

## `blocks.py`

```python
class ConvNormAct(nn.Sequential):
    # Conv2d(cin, cout, k=3, stride=s, padding=1, bias=False) -> GroupNorm(g, cout) -> GELU
    # g = min(8, cout) and cout % g == 0 (assert)

class TransformerBlock(nn.Module):
    # pre-norm:  x = x + drop(attn(LN(x)));  x = x + drop(mlp(LN(x)))
    # attn: qkv = Linear(dim, 3*dim); split heads; F.scaled_dot_product_attention(q, k, v,
    #       dropout_p=dropout if training else 0); proj = Linear(dim, dim)
    # mlp: Linear(dim, mlp_ratio*dim) -> GELU -> Dropout -> Linear(mlp_ratio*dim, dim)
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float)
    def forward(self, x: Tensor) -> Tensor          # (B, N, dim) -> (B, N, dim)

class PatchEmbed(nn.Module):
    # Conv2d(cin, dim, kernel=patch, stride=patch); output flattened to (B, N, dim), N = (Hp/p)*(Wp/p)
    # learned position embedding nn.Parameter(1, N_max, dim), trunc_normal_(std=0.02).
    # N is fixed at construction from (Hp, Wp): the model is built for one padded grid size
    # and asserts the input matches.

class SkipStem(nn.Module):
    # levels L = log2(patch). Level 0: ConvNormAct(cin, c0) -> ConvNormAct(c0, c0) at full res.
    # Level i>0: ConvNormAct(c_{i-1}, c_i, stride 2) -> ConvNormAct(c_i, c_i).  c_i = base * 2**i.
    # forward returns [f0 (full res), f1 (1/2), ..., f_{L-1} (1/2^{L-1})]

class DecoderStage(nn.Module):
    # Upsample(x2, bilinear, align_corners=False) -> concat(skip) -> ConvNormAct(cin+cskip, cout)
    # -> ConvNormAct(cout, cout).  cskip = 0 when no skip is given.
```

## `vit_cnn.py`

```python
class ViTCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, grid_hw: tuple[int, int],
                 patch: int = 8, dim: int = 256, depth: int = 6, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 stem_base_channels: int = 32, use_skip_stem: bool = True)
    def forward(self, x: Tensor) -> Tensor   # (B, C_in, Hp, Wp) -> (B, D, Hp, Wp)
```

Default architecture (patch 8, Hp = Wp = 80, C_in = 7, D = 15). Assert every shape in `forward`
(use `assert t.shape == (...)`, with a message).

| # | Stage | Operation | Output shape |
| --- | --- | --- | --- |
| 0 | input | (already padded) | B × 7 × 80 × 80 |
| 1 | patch embed | Conv2d(7, 256, k=8, s=8) → flatten → + pos-emb → Dropout | B × 100 × 256 |
| 2 | encoder | 6 × TransformerBlock(256, 8 heads, mlp 1024, dropout 0.1) → LayerNorm | B × 100 × 256 |
| 3 | reshape | tokens → grid | B × 256 × 10 × 10 |
| 4 | skip stem | f0: 32 @ 80×80, f1: 64 @ 40×40, f2: 128 @ 20×20 | — |
| 5 | decoder 1 | up ×2 → cat f2 → 384 → 128 → 128 | B × 128 × 20 × 20 |
| 6 | decoder 2 | up ×2 → cat f1 → 192 → 64 → 64 | B × 64 × 40 × 40 |
| 7 | decoder 3 | up ×2 → cat f0 → 96 → 32 → 32 | B × 32 × 80 × 80 |
| 8 | head | Conv2d(32, 15, k=1) | B × 15 × 80 × 80 |

General rule for any power-of-two patch p: L = log2(p) decoder stages; decoder output channels
are the skip channels in reverse (`[c_{L-1}, …, c_0]`); stage i concatenates `f_{L-1-i}`.
Patch 4 → 400 tokens, L = 2, skips 32 @ 80, 64 @ 40, decoder 256 → 64 → 32.
`use_skip_stem=False` (ablation): decoder stages receive no skip (cskip = 0), channels unchanged.

Initialization: Linear → `trunc_normal_(std=0.02)`, bias 0; LayerNorm weight 1 bias 0; convs →
PyTorch default (kaiming uniform); **head bias = 0 and head weight scaled by 0.1** so initial
predictions are near the base (climatology in anomaly mode, mean in absolute mode).

Expected parameter count for defaults ≈ 5.9 M (encoder ≈ 4.9 M, skip stem ≈ 0.3 M, decoder ≈ 0.8 M; hand estimate). The
`summary.py` printout must show the real number; record it in `docs/decisions_log.md`.

## `cnn_unet.py`

The main comparison baseline: same input, output, loss, training recipe; convolution instead of
self-attention. Must have a parameter count within ±10 % of `ViTCNN` with the same config.

```python
class CNNUNet(nn.Module):
    N_DOWN = 3   # total downsampling factor 8 -> pad multiple 8 (src/dataset uses this)
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 48)
    def forward(self, x: Tensor) -> Tensor     # (B, C_in, Hp, Wp) -> (B, D, Hp, Wp)
```

Architecture with base `b` (channels `b, 2b, 4b, 8b`):

| Stage | Operation | Shape (80 × 80 input) |
| --- | --- | --- |
| enc0 | ConvNormAct(C_in, b) → ConvNormAct(b, b) | b × 80 × 80 |
| enc1 | ConvNormAct(b, 2b, s=2) → ConvNormAct(2b, 2b) | 2b × 40 × 40 |
| enc2 | ConvNormAct(2b, 4b, s=2) → ConvNormAct(4b, 4b) | 4b × 20 × 20 |
| enc3 | ConvNormAct(4b, 8b, s=2) → ConvNormAct(8b, 8b) | 8b × 10 × 10 |
| bottleneck | 1 × ConvNormAct(8b, 8b) → Dropout2d(0.1) | 8b × 10 × 10 |
| dec2 | DecoderStage(8b, skip enc2 4b → 4b) | 4b × 20 × 20 |
| dec1 | DecoderStage(4b, skip enc1 2b → 2b) | 2b × 40 × 40 |
| dec0 | DecoderStage(2b, skip enc0 b → b) | b × 80 × 80 |
| head | Conv2d(b, D, 1), same init as ViTCNN head | D × 80 × 80 |

Hand estimate: b = 48 gives ≈ 5.7 M parameters (within ±10 % of the ViT-CNN's ≈ 5.9 M);
b = 40 gives ≈ 3.9 M. GroupNorm needs channels divisible by 8, so candidates are multiples of 8.

## `factory.py`

```python
def build_model(cfg, model_name: str, in_channels: int, out_channels: int,
                grid_hw_padded: tuple[int, int]) -> nn.Module
def count_parameters(model: nn.Module) -> int   # trainable only
def match_cnn_base_channels(cfg, in_channels, out_channels, grid_hw_padded,
                            candidates=(32, 40, 48, 56, 64, 72, 80)) -> int
    # builds ViTCNN and each CNNUNet candidate, returns the base whose count is closest to ViTCNN;
    # raises if none within ±10 %. Used when cfg.cnn.base_channels == "auto".
```

Model names: `vit_cnn`, `cnn_unet`. Unknown name → `ValueError`.

## `summary.py` (ENTRY)

`python -m src.models.summary --config config.yaml` prints for both models: per-stage output
shapes for a dummy batch (2, C_in, Hp, Wp), parameter counts per stage and total, the chosen CNN
base, and the ratio. Writes `results/tables/model_summary.md`. Runs on CPU in seconds.

## Tests (`tests/test_models.py`, CPU)

- Output shape (2, 15, 80, 80) for both models with defaults; also patch 4 for ViTCNN; also
  `use_skip_stem=False`.
- Wrong input size → assertion error with a clear message.
- `match_cnn_base_channels` returns a base within ±10 %.
- Gradient flows: after one backward pass on a random batch, every parameter has a non-None,
  finite gradient.
- Overfit check: each model drives MSE on one fixed random batch (B=2) down by > 90 % within 200
  Adam steps (lr 1e-3) on CPU. Mark `slow` if it takes > 30 s.
- Initial outputs are small: |out| mean < 0.5 at init.

## Done when

`pytest tests -k models` passes and `python -m src.models.summary` prints both parameter counts.
