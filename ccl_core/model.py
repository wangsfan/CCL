"""
CCL main model.

Supports two modes:
  - Same source/target feature dims   (Mutagenicity, PTC, COX2)
  - Different source/target dims      (PROTEINS -> DD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class MLP(nn.Module):
    """Plain MLP used inside GINConv and as projection head."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(self.bns[i](layer(x)))
        return self.layers[-1](x)


class GINBranch(nn.Module):
    """
    Single GIN encoder branch.

    If source/target feature dims are equal: a single feat_map is used.
    Otherwise, separate source_feat_map / target_feat_map are used
    and selected by `is_source`.
    """

    def __init__(self, in_dim_source, hidden_dim, z_shared_dim, num_layers,
                 in_dim_target=None):
        super().__init__()
        if in_dim_target is None:
            in_dim_target = in_dim_source

        if in_dim_source == in_dim_target:
            self.feat_map = nn.Linear(in_dim_source, hidden_dim)
            self.source_feat_map = None
            self.target_feat_map = None
        else:
            self.feat_map = None
            self.source_feat_map = nn.Linear(in_dim_source, hidden_dim)
            self.target_feat_map = nn.Linear(in_dim_target, hidden_dim)

        self.gin_layers = nn.ModuleList(
            [GINConv(MLP(hidden_dim, hidden_dim, hidden_dim), train_eps=True)
             for _ in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_dim, z_shared_dim)

    def forward(self, x, edge_index, batch, is_source=True):
        if self.feat_map is not None:
            x = self.feat_map(x)
        else:
            x = self.source_feat_map(x) if is_source else self.target_feat_map(x)

        x = F.relu(x)
        for layer in self.gin_layers:
            x = F.relu(layer(x, edge_index))

        emb = global_mean_pool(x, batch)     # paper: average pooling
        return F.normalize(self.fc(emb), dim=-1)


class CoCo_DoubleGIN(nn.Module):
    """
    CCL: dual GIN encoders + causal/spurious decoupling + classifier.

    Same-dim case:       CoCo_DoubleGIN(in_dim, ...)
    Cross-dataset case:  CoCo_DoubleGIN(in_dim_source, ..., in_dim_target=in_dim_target)
    """

    def __init__(self, in_dim_source, hidden_dim, z_shared_dim,
                 z_c_dim, z_s_dim, num_layers, num_classes, in_dim_target=None):
        super().__init__()
        self.branch1 = GINBranch(in_dim_source, hidden_dim, z_shared_dim,
                                 num_layers, in_dim_target)
        self.branch2 = GINBranch(in_dim_source, hidden_dim, z_shared_dim,
                                 num_layers, in_dim_target)

        self.proj_c = nn.Sequential(
            nn.Linear(z_shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_c_dim),
        )
        self.proj_s = nn.Sequential(
            nn.Linear(z_shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_s_dim),
        )
        self.generator = nn.Sequential(
            nn.Linear(z_c_dim + z_s_dim, z_shared_dim),
            nn.ReLU(),
            nn.Linear(z_shared_dim, z_shared_dim),
        )
        self.aux_s_classifier = nn.Linear(z_s_dim, num_classes)
        self.classifier = nn.Linear(z_c_dim, num_classes)

    def forward(self, x, edge_index, batch, is_source=True):
        z1 = self.branch1(x, edge_index, batch, is_source)
        z2 = self.branch2(x, edge_index, batch, is_source)
        zc1 = F.normalize(self.proj_c(z1), dim=-1)
        zs1 = F.normalize(self.proj_s(z1), dim=-1)
        zc2 = F.normalize(self.proj_c(z2), dim=-1)
        zs2 = F.normalize(self.proj_s(z2), dim=-1)
        logits = self.classifier(zc1)
        return z1, z2, zc1, zs1, zc2, zs2, logits