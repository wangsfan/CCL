"""Hyper-parameters, random seed, and pseudo-label utilities."""

import random
import numpy as np
import torch
import torch.nn.functional as F


class Args:
    """Global hyper-parameters."""

    def __init__(self):
        # Model
        self.hidden_dim = 64
        self.output_dim = 64         # dim of z1, z2
        self.num_layers = 2
        self.z_c_dim = 64            # causal embedding dim
        self.z_s_dim = 64            # spurious embedding dim

        # Training
        self.batch_size = 32
        self.lr = 1e-4
        self.epochs = 100
        self.warmup_epochs = 10
        self.seed = 42

        # Temperatures
        self.tau = 0.5               # tau1: contrastive temperature
        self.tau_pseudo = 0.95       # tau2: pseudo-label confidence threshold

        # Loss weights (paper)
        self.alpha = 0.1             # for L_MI^2
        self.beta = 0.01             # for (L_CDC + L_MI^1 + alpha * L_MI^2)
        self.gamma = 0.003           # for (L_recont + L_inter)

        # Data split
        self.source_train_ratio = 0.8
        self.target_train_ratio = 0.8

        self.num_classes = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_adaptive_pseudo_labels(logits_t, tau_pseudo=0.95):
    """
    Adaptive confidence-threshold pseudo-labeling.

    Returns:
        pseudo_labels: [B] hard labels
        conf_mask:     [B] bool, True = keep (confidence above threshold)
    """
    probs_t = F.softmax(logits_t, dim=1)
    conf_scores, pseudo_labels = torch.max(probs_t, dim=1)
    num_classes = probs_t.size(1)

    # Per-class max confidence
    class_max_conf = torch.zeros(num_classes, device=logits_t.device)
    for c in range(num_classes):
        mask = pseudo_labels == c
        if mask.any():
            class_max_conf[c] = conf_scores[mask].max()

    tau_c = class_max_conf * tau_pseudo
    conf_mask = conf_scores > tau_c[pseudo_labels]
    return pseudo_labels, conf_mask