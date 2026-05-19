"""
dataset.py — Pre-generated episode dataset for TabICL-free training.

Each episode file (episode_NNNNNN.pt) is a dict produced by generate_dataset.py:
    E_all      : (B, T, d_max*d_model_base)  — TabICL embeddings, input to JointReadoutLayer
    P          : int                          — number of context (train) rows
    d          : int                          — actual target dimension (outputs sliced to [:d])
    X_train    : (B, P, p)
    Y_train    : (B, P, d)
    X_test     : (B, N, p)
    Y_test     : (B, N, d)
    oracle_mu  : (B, N, d)
    oracle_D   : (B, N, d)
    oracle_V   : (B, N, d, r_data)
    p, n_train, n_test : int metadata

Usage::

    from dataset import make_episode_loader
    loader = make_episode_loader("./data/episodes", shuffle=True, num_workers=2)
    for episode in loader:
        E_all  = episode["E_all"].to(device)   # (B, T, d_max*d_model)
        P      = episode["P"]
        d      = episode["d"]
        Y_test = episode["Y_test"].to(device)  # (B, N, d)
        ...
"""

from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import DataLoader, Dataset


class EpisodeDataset(Dataset):
    """Maps episode index → loaded dict from a pre-generated .pt file."""

    def __init__(self, dataset_dir: str | None = None, files: list[str] | None = None) -> None:
        if files is not None:
            self.files = files
        else:
            self.files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No episode_*.pt files found")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], weights_only=True)


def _collate_episode(batch: list[dict]) -> dict:
    # Each item in batch is already a full episode dict (with B datasets inside).
    # batch_size=1 in the DataLoader, so just unwrap and return as-is.
    assert len(batch) == 1, "EpisodeDataset expects batch_size=1 in DataLoader"
    return batch[0]


def split_episode_files(dataset_dir: str, val_n_episodes: int) -> tuple[list[str], list[str]]:
    """Split episode files into (train_files, val_files).

    Holds out the last val_n_episodes files for validation so they are never
    seen during training, ensuring no distribution shift.
    """
    files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.pt")))
    if not files:
        raise FileNotFoundError(f"No episode_*.pt files found in {dataset_dir!r}")
    n_val = min(val_n_episodes, len(files) - 1)
    return files[:-n_val], files[-n_val:]


def make_episode_loader(
    dataset_dir: str | None = None,
    shuffle: bool = True,
    num_workers: int = 2,
    files: list[str] | None = None,
) -> DataLoader:
    """Return a DataLoader that yields one pre-generated episode dict per iteration."""
    dataset = EpisodeDataset(dataset_dir=dataset_dir, files=files)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_episode,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def infinite_episode_iter(loader: DataLoader):
    """Yield episodes from loader indefinitely, re-shuffling after each full pass."""
    while True:
        yield from loader
