"""Train CCL on COX2_MD -> COX2."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ccl_core import CoCo_DoubleGIN, train, test, set_seed, Args
from cox2.dataset import load_cox2_datasets, prepare_data_loader


def main():
    args = Args()
    set_seed(args.seed)

    source_ds, target_ds, in_dim = load_cox2_datasets(
        source_name="COX2_MD", target_name="COX2"
    )

    s_train, s_test = prepare_data_loader(
        source_ds, args.source_train_ratio, args.batch_size, args.seed)
    t_train, t_test = prepare_data_loader(
        target_ds, args.target_train_ratio, args.batch_size, args.seed)

    print(f"Source train/test: {len(s_train.dataset)}/{len(s_test.dataset)}")
    print(f"Target train/test: {len(t_train.dataset)}/{len(t_test.dataset)}")

    model = CoCo_DoubleGIN(
        in_dim, args.hidden_dim, args.output_dim,
        args.z_c_dim, args.z_s_dim, args.num_layers, args.num_classes,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("models", exist_ok=True)
    best_s_acc = 0.0
    best_t_acc = 0.0

    for epoch in range(args.epochs):
        loss = train(s_train, t_train, model, opt, args, epoch)
        s_acc = test(s_test, model, args)
        t_acc = test(t_test, model, args)

        if s_acc > best_s_acc:
            best_s_acc = s_acc
        if t_acc > best_t_acc:
            best_t_acc = t_acc
            torch.save(model.state_dict(), "models/best_cox2.pth")

        print(f"Epoch {epoch + 1:03d} | Loss {loss:.4f} | "
              f"Source {s_acc:.4f} (best {best_s_acc:.4f}) | "
              f"Target {t_acc:.4f} (best {best_t_acc:.4f})")

    print("\nDone.")
    print(f"Best source acc: {best_s_acc:.4f}")
    print(f"Best target acc: {best_t_acc:.4f}")


if __name__ == "__main__":
    main()