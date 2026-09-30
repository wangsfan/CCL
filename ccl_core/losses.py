"""All CCL losses."""

import random
import torch
import torch.nn.functional as F


def cross_branch_contrastive_loss(z1, z2, temperature=0.5):
    """
    L_CBC: InfoNCE across the two branches.
    Positive pair: (z1_i, z2_i). Negatives: all j != i.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    sim = torch.matmul(z1, z2.T) / temperature      # [B, B]
    labels = torch.arange(B, device=sim.device)
    return F.cross_entropy(sim, labels)


def cross_domain_contrastive_loss(c_s, c_t, y_s, pseudo_labels_t, temperature=0.5):
    """
    L_CDC: pull cross-domain causal embeddings with the same label together.
    """
    c_s = F.normalize(c_s, dim=1)
    c_t = F.normalize(c_t, dim=1)
    B_t = c_t.size(0)
    if B_t == 0:
        return torch.tensor(0.0, device=c_t.device)

    sim = torch.matmul(c_t, c_s.T) / temperature    # [B_t, B_s]
    loss = 0.0
    for i in range(B_t):
        pos_mask = (y_s == pseudo_labels_t[i])
        if pos_mask.sum() == 0:
            continue
        pos_sim = sim[i, pos_mask].exp().sum()
        denom = sim[i].exp().sum()
        loss = loss - torch.log(pos_sim / denom)
    return loss / B_t


def compute_mi_losses(z_s, z_shared, labels, model, temperature=0.5, alpha=0.1):
    """
    Returns (L_MI1, L_MI2).

    L_MI1: minimize I(c'; y) via variational upper bound + shuffled labels.
    L_MI2: maximize I(c'; z1) via InfoNCE (minimizing this = maximizing MI).
    """
    # ----- L_MI1 -----
    logits_aux = model.aux_s_classifier(z_s)
    ce_joint = F.cross_entropy(logits_aux, labels)
    shuffled_labels = labels[torch.randperm(len(labels), device=labels.device)]
    ce_marginal = F.cross_entropy(logits_aux, shuffled_labels)
    L_MI1 = ce_joint - ce_marginal

    # ----- L_MI2 -----
    z_s_norm = F.normalize(z_s, dim=1)
    z_shared_norm = F.normalize(z_shared, dim=1)
    sim = torch.matmul(z_s_norm, z_shared_norm.T) / temperature
    B = sim.size(0)
    labels_diag = torch.arange(B, device=sim.device)
    L_MI2 = F.cross_entropy(sim, labels_diag)

    return L_MI1, L_MI2


def compute_intervention_loss(z_c, z_s, z_shared, domain_mask, model):
    """
    Returns (L_recont, L_inter).

    L_recont: source-only reconstruction of z1 from (c, c').
    L_inter:  source reconstruction when c' is replaced by a target c'.
    """
    z_comb = torch.cat([z_c, z_s], dim=1)
    z_recon = model.generator(z_comb)
    L_re = F.mse_loss(z_recon[domain_mask], z_shared[domain_mask])

    L_inter = torch.tensor(0.0, device=z_c.device)
    if domain_mask.any() and (~domain_mask).any():
        idx_s = torch.where(domain_mask)[0]
        idx_t = torch.where(~domain_mask)[0]
        accum = 0.0
        for i in idx_s:
            j = random.choice(idx_t.tolist())
            z_c_i = z_c[i].unsqueeze(0)
            z_s_j = z_s[j].unsqueeze(0)
            z_comb_swap = torch.cat([z_c_i, z_s_j], dim=1)
            z_recon_swap = model.generator(z_comb_swap)
            accum = accum + F.mse_loss(z_recon_swap, z_shared[i].unsqueeze(0))
        L_inter = accum / len(idx_s)

    return L_re, L_inter