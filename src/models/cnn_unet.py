"""Plain CNN U-Net baseline with the same input/output contract as ViTCNN."""

from __future__ import annotations

from torch import Tensor, nn

from src.models.blocks import ConvNormAct, DecoderStage, init_head


class CNNUNet(nn.Module):
    N_DOWN = 3  # total downsampling factor 8 -> pad multiple 8

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 48):
        super().__init__()
        b = base_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        self.enc0 = nn.Sequential(ConvNormAct(in_channels, b), ConvNormAct(b, b))
        self.enc1 = nn.Sequential(ConvNormAct(b, 2 * b, 2), ConvNormAct(2 * b, 2 * b))
        self.enc2 = nn.Sequential(ConvNormAct(2 * b, 4 * b, 2), ConvNormAct(4 * b, 4 * b))
        self.enc3 = nn.Sequential(ConvNormAct(4 * b, 8 * b, 2), ConvNormAct(8 * b, 8 * b))
        self.bottleneck = nn.Sequential(ConvNormAct(8 * b, 8 * b), nn.Dropout2d(0.1))
        self.dec2 = DecoderStage(8 * b, 4 * b, 4 * b)
        self.dec1 = DecoderStage(4 * b, 2 * b, 2 * b)
        self.dec0 = DecoderStage(2 * b, b, b)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)
        init_head(self.head)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        m = 2**self.N_DOWN
        assert c == self.in_channels, f"expected {self.in_channels} input channels, got {c}"
        assert h % m == 0 and w % m == 0, f"input {h}x{w} must be a multiple of {m} (pad it)"
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.bottleneck(self.enc3(e2))
        z = self.dec0(self.dec1(self.dec2(e3, e2), e1), e0)
        out = self.head(z)
        assert out.shape == (b, self.out_channels, h, w), out.shape
        return out
