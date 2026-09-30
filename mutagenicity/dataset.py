"""Mutagenicity: split into M0/M1/M2/M3 by edge density."""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


def calculate_edge_density(graph):
    n, e = graph.num_nodes, graph.num_edges
    max_e = n * (n - 1) / 2 if n > 1 else 1.0
    return e / max_e


def get_m_subset(full_dataset, subset_name):
    """Sort by edge density, then split into 4 equal parts."""
    pairs = [(calculate_edge_density(g), g) for g in full_dataset]
    pairs.sort(key=lambda x: x[0])
    q = len(pairs) // 4
    ranges = {
        "M0": (0, q),
        "M1": (q, 2 * q),
        "M2": (2 * q, 3 * q),
        "M3": (3 * q, None),
    }
    if subset_name not in ranges:
        raise ValueError("subset_name must be M0/M1/M2/M3")
    lo, hi = ranges[subset_name]
    return [g for _, g in pairs[lo:hi]]


def get_full_dataset(root="data/TUDataset", name="Mutagenicity"):
    return TUDataset(root=root, name=name)


def load_single_domain_data(full_dataset, domain_name, args):
    """Split into train/test and return DataLoaders + node feature dim."""
    dataset = get_m_subset(full_dataset, domain_name)
    size = len(dataset)
    train_size = int(args.source_train_ratio * size)
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, size - train_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, dataset[0].x.shape[1]