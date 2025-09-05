# graph.py
"""
Experiment runner + plotting for the Physical Constraints NN.
- Sweeps: noise sigma, quant bit-depth, drift rate/mode, real vs complex
- Outputs: CSV logs + PNG plots
Requires your existing modules: TinyNet, DriftController, train/evaluate, data.get_mnist_loaders, utils.
"""

import os, csv, time, math, json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import torch
import matplotlib.pyplot as plt

from model import TinyNet                         # uses complex activation + ADC + noise  :contentReference[oaicite:6]{index=6}
from constraints.drift import DriftController     # drift controller with epoch/batch + multiplicative  :contentReference[oaicite:7]{index=7}
from train import train_one_epoch, evaluate       # training/eval loops (batch-drift hook supported)  :contentReference[oaicite:8]{index=8}
from data import get_loaders, get_class_names, infer_in_dim
from utils import set_seed, device_auto
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -----------------------
# Config structures
# -----------------------
@dataclass
class TrainConfig:
    # data
    dataset: str = "mnist"

    # core knobs
    use_complex: bool = True
    mode_noise: str = "off"             # "off"|"add"|"mul"|"both"
    noise_sigma_add: float = 0.0
    noise_sigma_mult: float = 0.0
    noise_apply_in_eval: bool = False
    noise_sigma_phase: float = 0.0

    mode_quant: str = "off"             # "off"|"act"|"both"
    act_bits: int = 8
    adc_in_range: Tuple[float, float] = (0.0, 1.0)
    adc_out_range: Tuple[float, float] = (-5.0, 5.0)
    adc_apply_in_eval: bool = True

    mode_drift: str = "off"             # "off"|"epoch"|"batch"
    drift_eta: float = 0.0
    drift_multiplicative: bool = False
    drift_bias: bool = True

    # training
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    width: int = 256
    seed: int = 42

    # bookkeeping
    tag: str = ""                        # appended into run folder/file names
    out_dir: str = "runs"               # base folder for logs/figures

# -----------------------
# Runner
# -----------------------
def run_experiment(cfg: TrainConfig) -> str:
    """
    Trains/evals one configuration, logs CSV (epoch, losses, acc) and returns the CSV path.
    """
    set_seed(cfg.seed)
    device = device_auto()

    os.makedirs(cfg.out_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_id}_{cfg.tag}" if cfg.tag else run_id
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Data
    train_loader, test_loader = get_loaders(cfg.dataset, batch_size=cfg.batch_size, num_workers=2)
    class_names = get_class_names(cfg.dataset)
    in_dim = infer_in_dim(cfg.dataset)
    num_classes = len(class_names)

    # Model
    model = TinyNet(
        in_dim=in_dim,
        num_classes=num_classes,
        mode_noise=cfg.mode_noise,
        mode_quant=cfg.mode_quant,
        noise_sigma_add=cfg.noise_sigma_add,
        noise_sigma_mult=cfg.noise_sigma_mult,
        noise_apply_in_eval=cfg.noise_apply_in_eval,
        noise_sigma_phase=cfg.noise_sigma_phase,
        act_bits=cfg.act_bits,
        adc_in_range=cfg.adc_in_range,
        adc_out_range=cfg.adc_out_range,
        adc_apply_in_eval=cfg.adc_apply_in_eval,
        use_complex=cfg.use_complex,
        width=cfg.width,
    ).to(device)  # TinyNet uses Noise + ADC according to flags  :contentReference[oaicite:9]{index=9}

    # Drift
    drift = DriftController(
        eta=cfg.drift_eta,
        mode=cfg.mode_drift,
        drift_bias=cfg.drift_bias,
        multiplicative=cfg.drift_multiplicative,
    )
    drift.attach(model)  # caches LinearRC targets  :contentReference[oaicite:10]{index=10}

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # CSV logger
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "test_loss", "test_acc",
            "use_complex", "mode_noise", "noise_sigma_add", "noise_sigma_mult", "noise_apply_in_eval",
            "mode_quant", "act_bits", "adc_apply_in_eval",
            "mode_drift", "drift_eta", "drift_multiplicative", "drift_bias",
            "seed", "width", "lr", "batch_size"
        ])

        best_acc = 0.0
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optim, device, drift_controller=drift)  # batch drift  :contentReference[oaicite:11]{index=11}
            drift.step_epoch(model)  # epoch drift  :contentReference[oaicite:12]{index=12}
            test_loss, test_acc = evaluate(model, test_loader, device)

            if test_acc > best_acc:
                best_acc = test_acc
                cm_path = os.path.join(run_dir, f"best_confusion_matrix_epoch{epoch}.png")
                plot_confusion_matrix(model, test_loader, device, cm_path, title=f"Best Accuracy ({best_acc:.4f}) at Epoch {epoch}", class_names=class_names)

            writer.writerow([
                epoch, train_loss, test_loss, test_acc,
                cfg.use_complex, cfg.mode_noise, cfg.noise_sigma_add, cfg.noise_sigma_mult, cfg.noise_apply_in_eval,
                cfg.mode_quant, cfg.act_bits, cfg.adc_apply_in_eval,
                cfg.mode_drift, cfg.drift_eta, cfg.drift_multiplicative, cfg.drift_bias,
                cfg.seed, cfg.width, cfg.lr, cfg.batch_size
            ])
            f.flush()

    # Basic learning curve plots
    plot_learning_curves(csv_path, run_dir, title=cfg.tag or run_name)

    return csv_path

# -----------------------
# Plot helpers
# -----------------------
def plot_confusion_matrix(model, loader, device, path, title="", class_names: List[str] = None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names or range(10), yticklabels=class_names or range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title or "Confusion Matrix")
    plt.savefig(path, dpi=160)
    plt.close()

def _read_csv(csv_path: str) -> Dict[str, List[float]]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {c: df[c].tolist() for c in df.columns}

def plot_learning_curves(csv_path: str, out_dir: str, title: str = ""):
    os.makedirs(out_dir, exist_ok=True)
    data = _read_csv(csv_path)
    epochs = data["epoch"]

    # Loss vs epoch
    plt.figure()
    plt.plot(epochs, data["train_loss"], label="train_loss")
    plt.plot(epochs, data["test_loss"], label="test_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss vs Epoch {title}"); plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_vs_epoch.png"), dpi=160)
    plt.close()

    # Acc vs epoch
    plt.figure()
    plt.plot(epochs, data["test_acc"], label="test_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"Accuracy vs Epoch {title}"); plt.legend()
    plt.savefig(os.path.join(out_dir, "acc_vs_epoch.png"), dpi=160)
    plt.close()

def plot_accuracy_vs_param(results: List[Tuple[float, float]], xlabel: str, out_path: str, title: str = ""):
    """
    results: list of (xparam, best_acc)
    """
    xs, ys = zip(*sorted(results, key=lambda t: t[0]))
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel); plt.ylabel("Best Test Accuracy"); plt.title(title)
    plt.savefig(out_path, dpi=160)
    plt.close()

# -----------------------
# Sweeps
# -----------------------
def sweep_noise_sigma(base: TrainConfig, sigmas: List[float]) -> List[Tuple[float, float]]:
    """
    Accuracy vs noise sigma (additive). Uses current ADC/drift settings.
    """
    results = []
    for s in sigmas:
        cfg = base
        cfg = dataclasses_replace(cfg, noise_sigma_add=s, mode_noise="add")
        cfg.tag = f"noise_sigma_{s:g}"
        csv_path = run_experiment(cfg)
        best_acc = max(_read_csv(csv_path)["test_acc"])
        results.append((s, best_acc))
    return results

def sweep_bits(base: TrainConfig, bits_list: List[int]) -> List[Tuple[int, float]]:
    """
    Accuracy vs quant bits (ADC in/out already set via mode_quant).
    """
    results = []
    for b in bits_list:
        cfg = base
        cfg = dataclasses_replace(cfg, act_bits=b, mode_quant="both", adc_apply_in_eval=True)
        cfg.tag = f"bits_{b}"
        csv_path = run_experiment(cfg)
        best_acc = max(_read_csv(csv_path)["test_acc"])
        results.append((b, best_acc))
    return results

def sweep_drift_eta(base: TrainConfig, etas: List[float], mode: str = "epoch") -> List[Tuple[float, float]]:
    """
    Accuracy vs drift rate for chosen mode ("epoch" or "batch").
    """
    results = []
    for e in etas:
        cfg = base
        cfg = dataclasses_replace(cfg, mode_drift=mode, drift_eta=e)
        cfg.tag = f"drift_{mode}_eta_{e:g}"
        csv_path = run_experiment(cfg)
        best_acc = max(_read_csv(csv_path)["test_acc"])
        results.append((e, best_acc))
    return results

def compare_complex_vs_real(base: TrainConfig) -> List[Tuple[str, float]]:
    """
    Best accuracy for complex vs real under the same constraints.
    """
    res = []
    for flag in [True, False]:
        cfg = base
        cfg = dataclasses_replace(cfg, use_complex=flag)
        cfg.tag = f"{'complex' if flag else 'real'}"
        csv_path = run_experiment(cfg)
        best_acc = max(_read_csv(csv_path)["test_acc"])
        res.append(("complex" if flag else "real", best_acc))
    return res

# -----------------------
# Utility for dataclasses immutability-ish replace
# -----------------------
def dataclasses_replace(cfg: TrainConfig, **kwargs) -> TrainConfig:
    d = asdict(cfg)
    d.update(kwargs)
    return TrainConfig(**d)

# -----------------------
# Script entry: example grids
# -----------------------
if __name__ == "__main__":
    # Base config aligned with your current main.py defaults
    base = TrainConfig(
        dataset="mnist",
        use_complex=True,
        mode_noise="add",
        noise_sigma_add=1e-3,
        noise_sigma_mult=0.0,
        noise_apply_in_eval=True,         # hardware-style noise  :contentReference[oaicite:13]{index=13}
        mode_quant="off",
        act_bits=8,
        adc_apply_in_eval=False,           # hardware-style ADC     :contentReference[oaicite:14]{index=14}
        adc_in_range=(0.0, 1.0),
        adc_out_range=(-5.0, 5.0),
        mode_drift="off",
        drift_eta=0.0,
        drift_multiplicative=False,
        drift_bias=True,
        epochs=20,
        batch_size=128,
        lr=1e-3,
        width=256,
        seed=42,
        out_dir="runs"
    )

    # 1) Learning curve baseline
    base.tag = "baseline_learning_curve"
    csv_path = run_experiment(base)

    # 2) Accuracy vs noise sigma
    sigmas = [0.0, 3e-4, 1e-3, 3e-3, 1e-2]
    noise_results = sweep_noise_sigma(base, sigmas)
    plot_accuracy_vs_param(
        noise_results, xlabel="Noise σ (additive)", 
        out_path=os.path.join(base.out_dir, "acc_vs_noise_sigma.png"),
        title="Accuracy vs Noise σ"
    )

    # 3) Accuracy vs quant bits
    bits_list = [4, 8]
    bits_results = sweep_bits(base, bits_list)
    plot_accuracy_vs_param(
        bits_results, xlabel="ADC bits", 
        out_path=os.path.join(base.out_dir, "acc_vs_bits.png"),
        title="Accuracy vs Bit-Depth"
    )

    # 4) Drift sweeps (epoch mode)
    etas = [0.0, 1e-5, 5e-5, 1e-4]
    drift_epoch_results = sweep_drift_eta(base, etas, mode="epoch")
    plot_accuracy_vs_param(
        drift_epoch_results, xlabel="Drift η (epoch mode)", 
        out_path=os.path.join(base.out_dir, "acc_vs_drift_epoch.png"),
        title="Accuracy vs Drift (epoch)"
    )

    # 5) Drift sweeps (batch mode) — note: same η values; interpret as per-step σ
    drift_batch_results = sweep_drift_eta(base, etas, mode="batch")
    plot_accuracy_vs_param(
        drift_batch_results, xlabel="Drift η (batch mode)", 
        out_path=os.path.join(base.out_dir, "acc_vs_drift_batch.png"),
        title="Accuracy vs Drift (batch)"
    )

    # 6) Complex vs Real
    cvsr = compare_complex_vs_real(base)
    # simple bar plot
    labels, vals = zip(*cvsr)
    plt.figure()
    plt.bar(labels, vals)
    plt.ylabel("Best Test Accuracy")
    plt.title("Complex vs Real")
    plt.savefig(os.path.join(base.out_dir, "acc_complex_vs_real.png"), dpi=160)
    plt.close()
