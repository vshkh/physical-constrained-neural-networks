# main.py
import argparse
import torch

from data import get_mnist_loaders
from model import TinyNet
from train import train_one_epoch, evaluate
from utils import set_seed, device_auto

def parse_args():
    p = argparse.ArgumentParser(description="Step-0: Real vs Complex TinyNet on MNIST")
    p.add_argument("--epochs", type=int, default=10, help="number of training epochs")
    p.add_argument("--batch_size", type=int, default=128, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--width", type=int, default=256, help="hidden width of TinyNet")
    p.add_argument("--use_complex", action="store_true", help="enable complex-valued pipeline")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--save", type=str, default="", help="optional path to save final model .pt")
    return p.parse_args()

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_auto()

    # Data
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size, num_workers=2)

    # Model

    # Manual complex flag:
    use_complex = False

    model = TinyNet(use_complex, width=args.width).to(device)
    first_param = next(model.parameters())
    print(f"Mode: {'COMPLEX' if use_complex else 'REAL'} "
          f"| Param dtype: {first_param.dtype} "
          f"| Params: {count_params(model):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train/Eval loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:02d} | "
              f"train_loss {train_loss:.4f} | "
              f"test_loss {test_loss:.4f} | "
              f"test_acc {test_acc:.3f}")

    # Optional save
    if args.save:
        torch.save({
            "model_state": model.state_dict(),
            "args": vars(args),
        }, args.save)
        print(f"Saved model to {args.save}")

if __name__ == "__main__":
    main()
