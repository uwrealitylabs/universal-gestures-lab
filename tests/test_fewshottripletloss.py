"""Legacy triplet trainer (src/fewshottripletloss.py). Its network uses BatchNorm1d, which
cannot train on a batch of one, and the number of triplets per epoch is not a multiple of
the batch size."""

import random

import torch
from torch.utils.data import DataLoader

from src import fewshottripletloss as legacy


def triplet_loader(n_samples):
    """Loader built as in main(): every sample of a two-label set anchors one triplet."""
    X = torch.randn(n_samples, legacy.input_dim)
    Y = torch.tensor([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    dataset = legacy.TripletGestureDataset(X, Y)
    assert len(dataset) == n_samples
    return DataLoader(dataset, batch_size=legacy.batch_size, shuffle=True)


def train_one_epoch(loader):
    model = legacy.SiameseNetwork(legacy.input_dim, legacy.embedding_dim)
    before = [p.detach().clone() for p in model.parameters()]
    trained = legacy.train_epoch(model, loader, legacy.TripletLoss(),
                                 torch.optim.Adam(model.parameters(), lr=1e-3))
    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    return trained, changed


def test_final_batch_of_one_is_skipped_not_fatal():
    torch.manual_seed(0)
    random.seed(0)
    trained, changed = train_one_epoch(triplet_loader(legacy.batch_size + 1))  # batches of 32 and 1
    assert trained == 1
    assert changed


def test_every_batch_of_two_or_more_is_trained():
    torch.manual_seed(0)
    random.seed(0)
    trained, changed = train_one_epoch(triplet_loader(2 * legacy.batch_size + 2))  # 32, 32, 2
    assert trained == 3
    assert changed
