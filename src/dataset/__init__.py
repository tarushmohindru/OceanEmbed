"""Splits, in-memory data and batching. Torch-dependent names are imported lazily."""

from src.dataset.arrays import load_split_numpy
from src.dataset.splits import split_indices

__all__ = ["BatchIterator", "SplitTensors", "build_split_tensors", "load_split_numpy",
           "split_indices"]  # fmt: skip


def __getattr__(name):  # keep `import src.dataset` torch-free for baselines/evaluation
    if name in ("SplitTensors", "build_split_tensors"):
        from src.dataset import tensors

        return getattr(tensors, name)
    if name == "BatchIterator":
        from src.dataset.iterator import BatchIterator

        return BatchIterator
    raise AttributeError(name)
