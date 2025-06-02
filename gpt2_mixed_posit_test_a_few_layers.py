import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel
from qtorch_plus.quant import posit_quantize

# Load GPT-2 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT2LMHeadModel.from_pretrained('gpt2').eval().to(device)

# Posit parameters
nsize = 4
log2_scales = np.arange(-5, 6)
sweep_scales = [2.0 ** x for x in log2_scales]
epsilon = 1e-8

def compute_sqnr(original, quantized):
    signal_power = torch.sum(original ** 2)
    noise_power = torch.sum((original - quantized) ** 2) + epsilon
    return 10 * torch.log10(signal_power / noise_power)

def quantize_with_energy_rms(w, nsize):
    w_np = w.detach().cpu().numpy().flatten()
    abs_w = np.abs(w_np) + epsilon
    energy = w_np ** 2
    log2_abs = np.log2(abs_w)
    log2_energy = 2 * log2_abs

    # Energy-weighted mean of log2(w^2)
    mean_log2_energy = np.sum(log2_energy * energy) / (np.sum(energy) + epsilon)
    mean_log2_abs = mean_log2_energy / 2
    floored_log2 = int(np.floor(mean_log2_abs))
    scale = 2 ** (-floored_log2)

    w_scaled = w * scale
    quant = posit_quantize(w_scaled, nsize=nsize, es=0, scale=1.0)
    quant_rescaled = quant / scale
    sqnr = compute_sqnr(w, quant_rescaled)

    return sqnr.item(), scale

def quantize_with_sweep(w, nsize):
    best_sqnr = -float("inf")
    best_scale = None
    for log2_s in log2_scales:
        scale = 2.0 ** log2_s
        w_scaled = w * scale
        quant = posit_quantize(w_scaled, nsize=nsize, es=0, scale=1.0)
        quant_rescaled = quant / scale
        sqnr = compute_sqnr(w, quant_rescaled)
        if sqnr > best_sqnr:
            best_sqnr = sqnr
            best_scale = scale
    return best_sqnr, best_scale

# Process first few layers
num_layers_to_test = 5
count = 0

print(f"{'Layer':<40} | {'RMS SQNR':>10} | {'Sweep SQNR':>12}")
print("-" * 70)

for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        weights = module.weight.data.detach().cpu().float()
        
        rms_sqnr, rms_scale = quantize_with_energy_rms(weights, nsize)
        sweep_sqnr, sweep_scale = quantize_with_sweep(weights, nsize)

        print(f"{name:<40} | {rms_sqnr:10.2f} | {sweep_sqnr:12.2f}")

        count += 1
        if count >= num_layers_to_test:
            break
