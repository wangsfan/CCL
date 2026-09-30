"""Train CCL on Mutagenicity (12 M_i -> M_j pairs)."""

import os
import sys
from itertools import product

import torch

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ccl_core import CoCo_DoubleGIN, train, test, set_seed, Args
from mutagenicity.dataset import get_full_dataset, load_single_domain_data


def main():
    args = Args()
    set_seed(args.seed)

    full_dataset = get_full_dataset(name="Mutagenicity")
    domains = ["M0", "M1", "M2", "M3"]
    results = {}

    os.makedirs("models", exist_ok=True)

    for source, target in product(domains, domains):
        if source == target:
            continue
        print(f"\n{'=' * 50}\n {source} -> {target}\n{'=' * 50}")

        s_train, s_test, in_dim = load_single_domain_data(full_dataset, source, args)
        t_train, t_test, _ = load_single_domain_data(full_dataset, target, args)

        model = CoCo_DoubleGIN(
            in_dim, args.hidden_dim, args.output_dim,
            args.z_c_dim, args.z_s_dim, args.num_layers, args.num_classes,
        ).to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_t_acc = 0.0
        for epoch in range(args.epochs):
            loss = train(s_train, t_train, model, opt, args, epoch)
            s_acc = test(s_test, model, args)
            t_acc = test(t_test, model, args)
            if t_acc > best_t_acc:
                best_t_acc = t_acc
                torch.save(model.state_dict(),
                           f"models/best_{source}_to_{target}.pth")
            print(f"Epoch {epoch + 1:02d} | Loss {loss:.3f} | "
                  f"{source} {s_acc:.3f} | {target} {t_acc:.3f} | "
                  f"Best {best_t_acc:.3f}")

        results[f"{source}->{target}"] = best_t_acc

    print("\n" + "=" * 60)
    print("Mutagenicity: all 12 pairs done.")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k} : {v:.4f}")


if __name__ == "__main__":
    main()