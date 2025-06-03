import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import transformers
import transformers.modeling_utils as modeling_utils
import json

transformers.logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

layer_stats = {}
epsilon = 1e-8

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, modeling_utils.Conv1D, nn.Embedding)):
        print(f"Processing layer: {name}")
        weights = module.weight.data.detach().float().cpu().numpy().flatten()

        abs_weights = np.abs(weights)
        mask = abs_weights > epsilon

        if np.sum(mask) == 0:
            layer_stats[name] = {
                "mean_log2_abs_weight": None,
                "median_log2_abs_weight": None,
                "energy_weighted_mean_log2_abs_weight": None
            }
            continue

        # Use masked weights
        w = weights[mask]
        abs_w = abs_weights[mask]
        log2_abs = np.log2(abs_w)
        log2_energy = 2 * log2_abs
        energy = w ** 2

        mean_log2 = np.mean(log2_abs)
        median_log2 = np.median(log2_abs)
        mean_log2_energy = np.sum(log2_energy * energy) / np.sum(energy)
        energy_weighted_mean_log2_abs = mean_log2_energy / 2

        layer_stats[name] = {
            "mean_log2_abs_weight": float(mean_log2),
            "median_log2_abs_weight": float(median_log2),
            "energy_weighted_mean_log2_abs_weight": float(energy_weighted_mean_log2_abs)
        }

# Save to file
with open("layer_weight_stats.json", "w") as f:
    json.dump(layer_stats, f, indent=2)

print("Layer stats saved to 'layer_weight_stats.json'.")
