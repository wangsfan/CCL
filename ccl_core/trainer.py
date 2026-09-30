"""Dataset-agnostic training / test loop."""

import torch
import torch.nn.functional as F
from itertools import zip_longest
from sklearn.metrics import accuracy_score

from .losses import (
    cross_branch_contrastive_loss,
    cross_domain_contrastive_loss,
    compute_mi_losses,
    compute_intervention_loss,
)
from .utils import generate_adaptive_pseudo_labels


def train(source_loader, target_loader, model, opt, args, epoch):
    model.train()
    total_loss = 0.0

    for s_data, t_data in zip_longest(source_loader, target_loader):
        opt.zero_grad()
        L_sup = L_CBC = L_CDC = L_MI1 = L_MI2 = L_recont = L_inter = 0.0
        pl = None

        # 1. Source forward + supervised loss
        if s_data is not None:
            s_data = s_data.to(args.device)
            z1_s, z2_s, zc_s, zs_s, _, _, logits_s = model(
                s_data.x, s_data.edge_index, s_data.batch, is_source=True
            )
            L_sup = F.cross_entropy(logits_s, s_data.y)

        # 2. Target forward + pseudo-label CE (after warmup)
        if t_data is not None:
            t_data = t_data.to(args.device)
            z1_t, z2_t, zc_t, zs_t, _, _, logits_t = model(
                t_data.x, t_data.edge_index, t_data.batch, is_source=False
            )
            if epoch >= args.warmup_epochs:
                pl, cm = generate_adaptive_pseudo_labels(logits_t, args.tau_pseudo)
                if cm.any():
                    L_sup = L_sup + F.cross_entropy(logits_t[cm], pl[cm])

        # 3. Cross-branch contrastive loss
        if s_data is not None and t_data is not None:
            z1_all = torch.cat([z1_s, z1_t])
            z2_all = torch.cat([z2_s, z2_t])
            L_CBC = cross_branch_contrastive_loss(z1_all, z2_all, args.tau)
        elif s_data is not None:
            L_CBC = cross_branch_contrastive_loss(z1_s, z2_s, args.tau)
        elif t_data is not None:
            L_CBC = cross_branch_contrastive_loss(z1_t, z2_t, args.tau)

        # 4. Decoupling losses (after warmup, requires both domains)
        if epoch >= args.warmup_epochs and s_data is not None and t_data is not None:
            L_CDC = cross_domain_contrastive_loss(zc_s, zc_t, s_data.y, pl, args.tau)

            zc_all = torch.cat([zc_s, zc_t])
            zs_all = torch.cat([zs_s, zs_t])
            z1_all = torch.cat([z1_s, z1_t])
            labels_all = torch.cat([s_data.y, pl])

            L_MI1, L_MI2 = compute_mi_losses(
                zs_all, z1_all, labels_all, model, args.tau, args.alpha
            )

            domain_mask = torch.cat([
                torch.ones(len(zc_s), dtype=torch.bool),
                torch.zeros(len(zc_t), dtype=torch.bool),
            ]).to(args.device)
            L_recont, L_inter = compute_intervention_loss(
                zc_all, zs_all, z1_all, domain_mask, model
            )

        # 5. Total loss
        if epoch >= args.warmup_epochs:
            loss = (L_sup + L_CBC
                    + args.beta * (L_CDC + L_MI1 + args.alpha * L_MI2)
                    + args.gamma * (L_recont + L_inter))
        else:
            loss = L_sup + L_CBC

        loss.backward()
        opt.step()
        total_loss += loss.item()

    return total_loss / max(len(source_loader), len(target_loader))


def test(loader, model, args, is_source=True):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(args.device)
            _, _, _, _, _, _, logits = model(
                d.x, d.edge_index, d.batch, is_source=is_source
            )
            preds.append(torch.argmax(logits, dim=1))
            trues.append(d.y)
    return accuracy_score(
        torch.cat(trues).cpu().numpy(),
        torch.cat(preds).cpu().numpy(),
    )