"""BZR_MD -> BZR: cross-dataset with different feature dims."""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


def load_dataset(dataset_name, train_ratio, batch_size, seed,
                 root="data/TUDataset"):
    """Load a dataset and split into train/test."""
    dataset = TUDataset(root=root, name=dataset_name)
    in_dim = dataset.num_node_features
    n = len(dataset)
    train_size = int(train_ratio * n)
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, n - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, in_dim