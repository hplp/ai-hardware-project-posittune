import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model
from qtorch_plus.quant import posit_quantize
from tqdm import tqdm

# -------------------- Adjustable Parameters --------------------
BIT_WIDTH = 4   # Set this to any bit width (e.g., 4, 5, 6, ...)
ES_BITS = 0     # Posit exponent bits (keep 0 for Posit(n,0))
NUM_LAYERS = 20
# components = ["attn.c_attn", "attn.c_proj"]
components = [
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    "ln_1",
    "ln_2"
]


# -------------------- INTn Quantization --------------------
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

# -------------------- Posit Quantization --------------------
def posit_quantize_energy_scaled_floor(tensor, nsize, es, rounding="nearest"):
    epsilon = 1e-8
    weight = tensor.detach().cpu().numpy().flatten()
    abs_weights = np.abs(weight) + epsilon
    log2_abs = np.log2(abs_weights)
    log2_energy = 2 * log2_abs
    energy = weight ** 2

    mean_log2_energy = np.sum(log2_energy * energy) / np.sum(energy)
    mean_log2_abs = mean_log2_energy / 2
    floored_log2 = int(np.floor(mean_log2_abs))
    scale = 2 ** (-floored_log2)

    quantized = posit_quantize(tensor, nsize=nsize, es=es, scale=scale, rounding=rounding)
    return quantized, scale

# -------------------- Main Loop --------------------
model = GPT2Model.from_pretrained("gpt2")

layer_labels = []
posit_sqnr = []
intn_sqnr = []

for i, block in tqdm(enumerate(model.h[:NUM_LAYERS]), total=NUM_LAYERS, desc="Quantizing Layers"):
    for comp in components:
        submodule = eval(f"block.{comp}")
        weight = submodule.weight.detach().cpu().numpy().flatten()
        tensor = torch.tensor(weight, dtype=torch.float32)
        signal_power = torch.sum(tensor ** 2)
        epsilon = 1e-8

        # --- Posit Quantization ---
        quant_posit, scale = posit_quantize_energy_scaled_floor(tensor, nsize=BIT_WIDTH, es=ES_BITS)
        noise_posit = torch.sum((tensor - quant_posit) ** 2) + epsilon
        sqnr_posit = 10 * torch.log10(signal_power / noise_posit)
        posit_sqnr.append(sqnr_posit.item())

        # --- INTn Quantization ---
        quant_intn = intn_quantize_asymmetric(tensor, nbits=BIT_WIDTH)
        noise_intn = torch.sum((tensor - quant_intn) ** 2) + epsilon
        sqnr_intn = 10 * torch.log10(signal_power / noise_intn)
        intn_sqnr.append(sqnr_intn.item())

        layer_labels.append(f"L{i}.{comp}")

# -------------------- Plot --------------------
x = np.arange(len(layer_labels))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, posit_sqnr, width, label=f'Posit({BIT_WIDTH},{ES_BITS})', color='royalblue')
plt.bar(x + width/2, intn_sqnr, width, label=f'INT{BIT_WIDTH}', color='orange')
plt.ylabel('SQNR (dB)')
plt.title(f'SQNR per GPT-2 Component (First {NUM_LAYERS} Layers, Bit Width = {BIT_WIDTH})')
plt.xticks(x, layer_labels, rotation=45, ha='right', fontsize=9)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
