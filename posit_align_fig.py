import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2Model

# Posit config
n = 8
es = 0

# Define log2 regions
edges = np.arange(-14, 10)
fraction_bits = []
for i in range(len(edges) - 1):
    left = edges[i]
    regime_bits = abs(int(np.floor(left))) if left < 0 else abs(int(np.floor(left))) + 1
    frac_bits = max(n - 1 - regime_bits - es, 0)
    fraction_bits.append(frac_bits)
x = edges[:-1] + 0.5

# Load GPT2 weights
model = GPT2Model.from_pretrained("gpt2-large")
attn_weights = model.h[0].attn.c_attn.weight.detach().cpu().numpy().flatten()

# Compute log2(|weights|) and energy-weighted mean
epsilon = 1e-8
abs_weights = np.abs(attn_weights) + epsilon
log2_weights = np.log2(abs_weights)
energy = attn_weights ** 2
log2_energy = 2 * log2_weights
mean_log2_energy = np.sum(log2_energy * energy) / np.sum(energy)
mean_log2_abs = mean_log2_energy / 2
floored_scale = int(round(mean_log2_abs))
scale_val = 2 ** (-floored_scale)

# Plot
fig, ax1 = plt.subplots(figsize=(12, 5))

# Plot histogram of log2(|weights|) on left y-axis
ax1.hist(log2_weights, bins=100, color='orange', alpha=0.5, edgecolor='black')
ax1.set_xlabel("log₂(value)")
ax1.set_ylabel("Weight Frequency", color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.set_xlim(-15, 8)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.axvline(x=mean_log2_abs, color='red', linestyle='--', linewidth=2)
ax1.legend([
    f"Weighted Mean ≈ {mean_log2_abs:.2f}\nscale = 2^–round({mean_log2_abs:.2f}) = 2^{-floored_scale}"
], loc='upper left')

# Plot fraction bits on right y-axis
ax2 = ax1.twinx()
ax2.step(x, fraction_bits, where='mid', color='blue', linewidth=2, label="Posit(8,0) fraction bits")
ax2.axvline(x=0, color='blue', linestyle=':', linewidth=2, label="Posit Peak Accuracy @ log₂(1)=0")
ax2.set_ylabel("Available Fraction Bits", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim(0, n - 1)
ax2.legend(loc='upper right')

plt.title("log₂(|w|) of GPT2-Large Attention Layer 0 (c_attn) vs. Posit(8,0) Fraction Bits")
plt.tight_layout()
plt.show()
