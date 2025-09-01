# main.py
import torch

from data import get_mnist_loaders
from model import TinyNet
from train import train_one_epoch, evaluate
from utils import set_seed, device_auto

# =========================
# Manual "flags" (edit me)
# =========================
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
WIDTH = 256
SEED = 42
SAVE_PATH = ""   # e.g., "tinynet_mnist.pt" or leave ""

# Core modes
USE_COMPLEX = False                 # complex pipeline toggle
MODE_NOISE = "off"                  # "off" | "add" | "mul" | "addmul"
MODE_QUANT = "act"                  # "off" | "act" | "both"  (ADC at input/output)

# Noise params
NOISE_SIGMA_ADD = 0.0               # std-dev for additive noise
NOISE_SIGMA_MULT = 0.0              # std-dev for multiplicative noise

# ADC (activation quantization) params
ACT_BITS = 8
ADC_APPLY_IN_EVAL = False           # True => apply ADC during eval (hardware-mode)
ADC_IN_RANGE = (0.0, 1.0)           # MNIST after ToTensor() is in [0,1]
ADC_OUT_RANGE = (0.0, 16.0)         # roomy logit range; adjust later if desired

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

    # Model
    model = TinyNet(
        mode_noise=MODE_NOISE,
        mode_quant=MODE_QUANT,
        noise_sigma_add=NOISE_SIGMA_ADD,
        noise_sigma_mult=NOISE_SIGMA_MULT,
        act_bits=ACT_BITS,
        adc_in_range=ADC_IN_RANGE,
        adc_out_range=ADC_OUT_RANGE,
        adc_apply_in_eval=ADC_APPLY_IN_EVAL,
        use_complex=USE_COMPLEX,
        width=WIDTH,
    ).to(device)

    first_param = next(model.parameters())
    print(
        f"Mode: {'COMPLEX' if USE_COMPLEX else 'REAL'} "
        f"| Param dtype: {first_param.dtype} "
        f"| Params: {count_params(model):,} "
        f"| Noise: {MODE_NOISE}(add={NOISE_SIGMA_ADD}, mul={NOISE_SIGMA_MULT}) "
        f"| Quant: {MODE_QUANT}(A{ACT_BITS}, Eval={'Y' if ADC_APPLY_IN_EVAL else 'N'})"
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train/Eval loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
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
