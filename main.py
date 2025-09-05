# main.py

import torch
from data import get_mnist_loaders, infer_in_dim
from model import TinyNet
from train import train_one_epoch, evaluate
from utils import set_seed, device_auto
from constraints.drift import DriftController

# =========================
# Manual "flags" (edit me)
# =========================
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
WIDTH = 256
SEED = 42
DATASET = "mnist" #  mnist | fashion | cifar10
SAVE_PATH = ""   # e.g., "tinynet_mnist.pt" or leave ""

# Core modes
USE_COMPLEX = True                   # complex pipeline toggle
MODE_NOISE = "add"                   # "off" | "add" | "mul" | "both"
MODE_QUANT = "both"                  # "off" | "act" | "both"  (ADC at input/output)
MODE_DRIFT = "epoch"                 # "off" | "epoch" | "batch"

# Noise params
NOISE_SIGMA_ADD = 1e-3              # std-dev for additive noise
NOISE_SIGMA_MULT = 0.0              # std-dev for multiplicative noise
NOISE_APPLY_IN_EVAL = True         # True => apply noise during eval (hardware-mode)
NOISE_SIGMA_PHASE    = 0.0


# ADC (activation quantization) params
ACT_BITS = 8
ADC_APPLY_IN_EVAL = True           # True => apply ADC during eval (hardware-mode)
ADC_IN_RANGE = (0.0, 1.0)           # MNIST after ToTensor() is in [0,1]
ADC_OUT_RANGE = (-5.0, 5.0)         # roomy logit range; adjust later if desired

# DRIFT (slow miscalibration)
DRIFT_ETA  = 1e-5       # start tiny; we’ll tune this after first run
DRIFT_BIAS = True
DRIFT_MULTIPLICATIVE = False

# =========================
# Helpers
# =========================
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def main():
    # Repro & device
    set_seed(SEED)
    device = device_auto()

    # Data
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE, num_workers=2)
    in_dim = infer_in_dim(DATASET)
    # Model
    model = TinyNet(
        mode_noise=MODE_NOISE,
        mode_quant=MODE_QUANT,
        noise_sigma_add=NOISE_SIGMA_ADD,
        noise_sigma_mult=NOISE_SIGMA_MULT,
        noise_apply_in_eval=NOISE_APPLY_IN_EVAL,
        noise_sigma_phase=NOISE_SIGMA_PHASE,
        act_bits=ACT_BITS,
        adc_in_range=ADC_IN_RANGE,
        adc_out_range=ADC_OUT_RANGE,
        adc_apply_in_eval=ADC_APPLY_IN_EVAL,
        use_complex=USE_COMPLEX,
        width=WIDTH,
    ).to(device)

    drift = DriftController(
        eta=DRIFT_ETA, mode=MODE_DRIFT, 
        multiplicative=DRIFT_MULTIPLICATIVE, 
        bias=DRIFT_BIAS
    )
    drift.attach(model)

    first_param = next(model.parameters())
    print(
        f"Mode: {'COMPLEX' if USE_COMPLEX else 'REAL'} "
        f"| Param dtype: {first_param.dtype} "
        f"| Params: {count_params(model):,} "
        f"| Noise: {MODE_NOISE}(add={NOISE_SIGMA_ADD}, mul={NOISE_SIGMA_MULT}) "
        f"| Quant: {MODE_QUANT}(A{ACT_BITS}) "
        f"| Drift: {MODE_DRIFT}(eta={DRIFT_ETA}, mul={DRIFT_MULTIPLICATIVE}) "
        f"| ADC_Eval={'Y' if ADC_APPLY_IN_EVAL else 'N'} "
        f"| Noise_Eval={'Y' if NOISE_APPLY_IN_EVAL else 'N'}"
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train/Eval loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, drift_controller=drift)
        drift.step_epoch(model)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"test_loss {test_loss:.4f} | "
            f"test_acc {test_acc:.3f}"
        )

    # Optional save
    if SAVE_PATH:
        torch.save(
            {"model_state": model.state_dict(),
             "config": {
                 "EPOCHS": EPOCHS,
                 "BATCH_SIZE": BATCH_SIZE,
                 "LR": LR,
                 "WIDTH": WIDTH,
                 "SEED": SEED,
                 "USE_COMPLEX": USE_COMPLEX,
                 "MODE_NOISE": MODE_NOISE,
                 "NOISE_SIGMA_ADD": NOISE_SIGMA_ADD,
                 "NOISE_SIGMA_MULT": NOISE_SIGMA_MULT,
                 "MODE_QUANT": MODE_QUANT,
                 "ACT_BITS": ACT_BITS,
                 "ADC_APPLY_IN_EVAL": ADC_APPLY_IN_EVAL,
                 "ADC_IN_RANGE": ADC_IN_RANGE,
                 "ADC_OUT_RANGE": ADC_OUT_RANGE,
             }},
            SAVE_PATH
        )
        print(f"Saved model to {SAVE_PATH}")

if __name__ == "__main__":
    main()
