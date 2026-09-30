"""Train CCL on PTC (12 subset-to-subset pairs)."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ccl_core import CoCo_DoubleGIN, train, test, set_seed, Args
from ptc.dataset import PTC_SUBSETS, load_ptc_datasets, prepare_data_loader


def main():
    args = Args()
    print("Loading PTC subsets...")
    processed, max_dim = load_ptc_datasets()
    print(f"Common feature dim: {max_dim}")

    results = {}
    os.makedirs("models", exist_ok=True)

    for source_name in PTC_SUBSETS:
        for target_name in PTC_SUBSETS:
            if source_name == target_name:
                continue
            print(f"\n{'=' * 50}\n {source_name} -> {target_name}\n{'=' * 50}")
            set_seed(args.seed)

            s_train, s_test = prepare_data_loader(
                processed[source_name], args.source_train_ratio,
                args.batch_size, args.seed, shuffle_train=True)
            t_train, t_test = prepare_data_loader(
                processed[target_name], args.target_train_ratio,
                args.batch_size, args.seed, shuffle_train=True)

            model = CoCo_DoubleGIN(
                max_dim, args.hidden_dim, args.output_dim,
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
                               f"models/best_{source_name}_to_{target_name}.pth")
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1:03d} | Loss {loss:.3f} | "
                          f"{source_name} {s_acc:.3f} | "
                          f"{target_name} {t_acc:.3f} | Best {best_t_acc:.3f}")

            results[(source_name, target_name)] = best_t_acc
            print(f"Done {source_name}->{target_name} | Best {best_t_acc:.4f}")

    print("\n" + "=" * 70)
    print("PTC results (best target test accuracy)")
    print("=" * 70)
    for (src, tgt), acc in results.items():
        print(f"{src:8s} -> {tgt:8s} : {acc:.4f}")


if __name__ == "__main__":
    main()