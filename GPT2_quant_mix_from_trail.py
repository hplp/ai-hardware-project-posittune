import torch
import numpy as np
import torch.nn as nn
from transformers import set_seed, GPT2LMHeadModel, GPT2TokenizerFast
from qtorch_plus.quant import posit_quantize, float_quantize
from datasets import load_dataset
from tqdm import tqdm
import transformers
import transformers.modeling_utils as modeling_utils

transformers.logging.set_verbosity_error()

# -------------------- Load Model --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
model = model.to(device)

# -------------------- Quant Settings --------------------
nsize = 4
es = 0
epsilon = 1e-8
log2_scales = np.arange(-10, 10)  # sweep from 2^-10 to 2^9
sweep_scales = [2.0 ** x for x in log2_scales]

def linear_activation(input):
    return float_quantize(input, exp=4, man=3, rounding="nearest")

def forward_pre_hook_linear(m, input):
    return (linear_activation(input[0]),)

# -------------------- Quantization with SQNR Sweep --------------------
layer_count = 0
op_count = 0

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, modeling_utils.Conv1D)):
        layer_count += 1
        print(f"Processing layer: {name}")

        weights = module.weight.data.detach().float().cpu()
        signal_power = torch.sum(weights ** 2)
        max_val = weights.abs().max()
        norm_weights = weights / (max_val + epsilon)

        best_sqnr = -float("inf")
        best_scale = None
        best_log2_scale = None

        for i, scale in enumerate(sweep_scales):
            quantized = posit_quantize(norm_weights, nsize=nsize, es=es, scale=scale)
            quantized_rescaled = quantized * max_val
            noise_power = torch.sum((weights - quantized_rescaled) ** 2) + epsilon
            sqnr = 10 * torch.log10(signal_power / noise_power)

            if sqnr > best_sqnr:
                best_sqnr = sqnr
                best_scale = scale
                best_log2_scale = log2_scales[i]

        quantized_weights = posit_quantize(norm_weights, nsize=nsize, es=es, scale=best_scale)
        module.weight.data = (quantized_weights * max_val).to(module.weight.data.device)

        print(f"Best log2(scale) = {best_log2_scale}, SQNR = {best_sqnr:.2f} dB")
        module.register_forward_pre_hook(forward_pre_hook_linear)

        if isinstance(module, modeling_utils.Conv1D):
            op_count += module.weight.shape[0] * module.weight.shape[1]
        else:
            op_count += module.in_features * module.out_features

    elif isinstance(module, nn.Embedding):
        print(f"Processing embedding layer: {name}")
        weights = module.weight.data.detach().float().cpu()
        signal_power = torch.sum(weights ** 2)
        max_val = weights.abs().max()
        norm_weights = weights / (max_val + epsilon)

        best_sqnr = -float("inf")
        best_scale = None
        best_log2_scale = None

        for i, scale in enumerate(sweep_scales):
            quantized = posit_quantize(norm_weights, nsize=nsize, es=es, scale=scale)
            quantized_rescaled = quantized * max_val
            noise_power = torch.sum((weights - quantized_rescaled) ** 2) + epsilon
            sqnr = 10 * torch.log10(signal_power / noise_power)

            if sqnr > best_sqnr:
                best_sqnr = sqnr
                best_scale = scale
                best_log2_scale = log2_scales[i]

        quantized_weights = posit_quantize(norm_weights, nsize=nsize, es=es, scale=best_scale)
        module.weight.data = (quantized_weights * max_val).to(module.weight.data.device)
        print(f"Best log2(scale) = {best_log2_scale}, SQNR = {best_sqnr:.2f} dB")

print("Total layers processed:", layer_count)
print("MAC operation count:", op_count)

# -------------------- Generation and Perplexity --------------------
def generate_text_direct(model, tokenizer):
    model.eval()
    prompts = [
        "Machine learning is the study of",
        "In the 19th century, the invention",
        "A robot was created",
        "One day I will"
    ]
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        outputs = model.generate(
            input_ids, max_length=50, num_return_sequences=3, do_sample=True
        )
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))
            print("-" * 35)

print("\nGenerated text after Posit quantization:\n")
generate_text_direct(model, tokenizer)

# Perplexity Evaluation
test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

max_length = model.config.n_positions
stride = 1024
lls = []
input_size = encodings.input_ids.size(1)
print("Input size:", input_size)

for i in tqdm(range(0, input_size, stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, input_size)
    trg_len = end_loc - i

    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len

    lls.append(log_likelihood)

if lls:
    ppl = torch.exp(torch.stack(lls).sum() / input_size)
    print("\nPerplexity after Posit quantization:", ppl.item())
else:
    print("No log likelihoods calculated.")
