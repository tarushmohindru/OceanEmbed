"""Batch iterator over split tensors held in memory (GPU-resident when possible)."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from src.dataset.tensors import SplitTensors


class BatchIterator:
    def __init__(
        self,
        t: SplitTensors,
        batch_size: int,
        shuffle: bool,
        seed: int,
        device: torch.device,
        drop_last: bool = False,
    ):
        self.t, self.batch_size, self.shuffle = t, batch_size, shuffle
        self.seed, self.device, self.drop_last = seed, device, drop_last
        self.epoch = 0
        self.n = t.x.shape[0]

    def __len__(self) -> int:
        full, rest = divmod(self.n, self.batch_size)
        return full + (0 if self.drop_last or rest == 0 else 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def order(self) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.n)
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.n, generator=g)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        idx_all = self.order()
        for b in range(len(self)):
            idx = idx_all[b * self.batch_size : (b + 1) * self.batch_size]
            batch = {}
            for name in ("x", "y", "m", "base"):
                src = getattr(self.t, name)
                sel = src.index_select(0, idx.to(src.device))
                batch[name] = sel.to(self.device, non_blocking=True)
            batch["t_index"] = self.t.t_index[idx]
            yield batch
