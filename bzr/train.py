"""Train CCL on BZR_MD -> BZR (cross-dataset, different feature dims)."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ccl_core import CoCo_DoubleGIN, train, test, set_seed, Args
from bzr.dataset import load_dataset


def main():
    args = Args()
    set_seed(args.seed)

    source_name, target_name = "BZR", "BZR_MD"

    print(f"Loading {source_name} as source...")
    s_train, s_test, in_dim_source = load_dataset(
        source_name, args.source_train_ratio, args.batch_size, args.seed)
    print(f"Source feature dim: {in_dim_source}")

    print(f"Loading {target_name} as target...")
    t_train, t_test, in_dim_target = load_dataset(
        target_name, args.target_train_ratio, args.batch_size, args.seed)
    print(f"Target feature dim: {in_dim_target}")

    # Different dims -> model uses separate source/target feat_map
    model = CoCo_DoubleGIN(
        in_dim_source=in_dim_source,
        hidden_dim=args.hidden_dim,
        z_shared_dim=args.output_dim,
        z_c_dim=args.z_c_dim,
        z_s_dim=args.z_s_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        in_dim_target=in_dim_target,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("models", exist_ok=True)
    best_s_acc = 0.0
    best_t_acc = 0.0

    for epoch in range(args.epochs):
        loss = train(s_train, t_train, model, opt, args, epoch)

        s_acc = test(s_test, model, args, is_source=True)
        t_acc = test(t_test, model, args, is_source=False)

        if s_acc > best_s_acc:
            best_s_acc = s_acc
        if t_acc > best_t_acc:
            best_t_acc = t_acc
            torch.save(model.state_dict(),
                       f"models/best_{source_name}_to_{target_name}.pth")

        print(f"Epoch {epoch + 1:02d} | Loss {loss:.4f} | "
              f"{source_name} {s_acc:.4f} (best {best_s_acc:.4f}) | "
              f"{target_name} {t_acc:.4f} (best {best_t_acc:.4f})")

    print(f"\nDone. {source_name} -> {target_name}")
    print(f"Best source acc: {best_s_acc:.4f}")
    print(f"Best target acc: {best_t_acc:.4f}")


if __name__ == "__main__":
    main()