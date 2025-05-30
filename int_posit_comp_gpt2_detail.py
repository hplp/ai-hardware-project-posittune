import torch
import numpy as np
from transformers import GPT2Model
from qtorch_plus.quant import posit_quantize

# Target config
BIT_WIDTH = 4
ES_BITS = 0
EPSILON = 1e-8
layer_idx = 2
component = "attn.c_proj"

# INTn quantization
def intn_quantize_asymmetric(tensor, nbits):
    int_min = -2 ** (nbits - 1)
    int_max = 2 ** (nbits - 1) - 1
    x_min, x_max = tensor.min(), tensor.max()
    scale = (x_max - x_min) / (int_max - int_min)
    zero_point = torch.round(-x_min / scale) + int_min
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, int_min, int_max)
    dequantized = (quantized - zero_point) * scale
    return dequantized

# Posit quantization
def posit_quantize_energy_scaled_floor(tensor, nsize, es, rounding="nearest"):
    weight = tensor.detach().cpu().numpy().flatten()
    abs_weights = np.abs(weight) + EPSILON
    log2_abs = np.log2(abs_weights)
    log2_energy = 2 * log2_abs
    energy = weight ** 2
    mean_log2_energy = np.sum(log2_energy * energy) / np.sum(energy)
    mean_log2_abs = mean_log2_energy / 2
    floored_log2 = int(np.floor(mean_log2_abs))
    scale = 2 ** (-floored_log2)
    quantized = posit_quantize(tensor, nsize=nsize, es=es, scale=scale, rounding=rounding)
    return quantized, scale

# Load model and get weight tensor
model = GPT2Model.from_pretrained("gpt2")
tensor = eval(f"model.h[{layer_idx}].{component}.weight").detach().cpu().to(torch.float32)

# Compute SQNRs
signal_power = torch.sum(tensor ** 2)
quant_posit, scale_posit = posit_quantize_energy_scaled_floor(tensor, nsize=BIT_WIDTH, es=ES_BITS)
noise_posit = torch.sum((tensor - quant_posit) ** 2) + EPSILON
sqnr_posit = 10 * torch.log10(signal_power / noise_posit)

quant_intn = intn_quantize_asymmetric(tensor, nbits=BIT_WIDTH)
noise_intn = torch.sum((tensor - quant_intn) ** 2) + EPSILON
sqnr_intn = 10 * torch.log10(signal_power / noise_intn)

print(f"[Layer h.{layer_idx}.{component}]")
print(f"  Posit SQNR: {sqnr_posit.item():.2f} dB (scale={scale_posit})")
print(f"  INT{BIT_WIDTH} SQNR: {sqnr_intn.item():.2f} dB")
