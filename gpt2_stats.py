import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import transformers
import transformers.modeling_utils as modeling_utils
import json

transformers.logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

layer_stats = {}

def safe_log2(x, eps=1e-8):
    x = x.abs()
    x = x[x > eps]  # avoid log(0)
    return torch.log2(x)

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, modeling_utils.Conv1D, nn.Embedding)):
        print(f"Processing layer: {name}")
        weights = module.weight.data.detach().float().cpu().flatten()

        mean_weight = weights.mean().item()
        median_weight = weights.median().item()

        log2_vals = safe_log2(weights)
        if len(log2_vals) > 0:
            log2_mean = log2_vals.mean().item()
        else:
            log2_mean = None

        layer_stats[name] = {
            "mean_weight": mean_weight,
            "median_weight": median_weight,
            "mean_log2_abs_weight": log2_mean
        }

# Save to file
with open("layer_weight_stats.json", "w") as f:
    json.dump(layer_stats, f, indent=2)

print("Layer stats saved to 'layer_weight_stats.json'.")
