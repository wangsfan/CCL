"""PTC: four subsets (PTC_MM/MR/FM/FR) as domains, features padded to common dim."""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform


class PadNodeFeatures(BaseTransform):
    """Pad / truncate node features to target_dim."""

    def __init__(self, target_dim):
        self.target_dim = target_dim

    def __call__(self, data):
        cur = data.x.size(1)
        if cur < self.target_dim:
            pad = torch.zeros((data.x.size(0), self.target_dim - cur))
            data.x = torch.cat([data.x, pad], dim=1)
        elif cur > self.target_dim:
            data.x = data.x[:, :self.target_dim]
        return data


PTC_SUBSETS = ["PTC_MM", "PTC_MR", "PTC_FM", "PTC_FR"]


def load_ptc_datasets(root="data/PTC"):
    """Load 4 PTC subsets; pad all node features to the same dim."""
    raw = {name: TUDataset(root=root, name=name) for name in PTC_SUBSETS}
    max_dim = max(ds[0].x.shape[1] for ds in raw.values() if len(ds) > 0)
    transform = PadNodeFeatures(max_dim)
    processed = {name: [transform(g) for g in ds] for name, ds in raw.items()}
    return processed, max_dim


def prepare_data_loader(dataset, train_ratio, batch_size, seed, shuffle_train=True):
    size = len(dataset)
    train_size = int(train_ratio * size)
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, size - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=shuffle_train, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader