"""COX2_MD -> COX2: pad source/target features to a common dim."""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform


class PadNodeFeatures(BaseTransform):
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


def load_cox2_datasets(source_name="COX2_MD", target_name="COX2",
                       root="data/TUDataset"):
    """Load source and target datasets; pad node features to max dim."""
    src_raw = TUDataset(root=root, name=source_name)
    tgt_raw = TUDataset(root=root, name=target_name)

    dim_s = src_raw[0].x.shape[1]
    dim_t = tgt_raw[0].x.shape[1]
    max_dim = max(dim_s, dim_t)
    print(f"[COX2] source dim={dim_s}, target dim={dim_t} -> pad to {max_dim}")

    transform = PadNodeFeatures(max_dim)
    src = [transform(g) for g in src_raw]
    tgt = [transform(g) for g in tgt_raw]
    return src, tgt, max_dim


def prepare_data_loader(dataset, train_ratio, batch_size, seed,
                        shuffle_train=True):
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