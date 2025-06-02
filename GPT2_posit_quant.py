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

# -------------------- Quantization Settings --------------------
nsize = 4  # posit bitwidth
es = 0     # posit exponent bits
epsilon = 1e-12

# Optional hook for quantized activation (not used here)
def linear_activation(input):
    # return input
    return float_quantize(input,exp=4, man=3, rounding="nearest")

def forward_pre_hook_linear(m, input):
    return (linear_activation(input[0]),)

# -------------------- Quantization Loop --------------------
layer_count = 0
op_count = 0

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, modeling_utils.Conv1D)):
        layer_count += 1
        print(f"Processing layer: {name}")

        weights_flattened = module.weight.data.cpu().numpy().flatten()
        abs_weights = np.abs(weights_flattened) + epsilon
        log2_abs = np.log2(abs_weights)
        log2_energy = 2 * log2_abs
        energy = weights_flattened ** 2
        mean_log2_energy = np.sum(log2_energy * energy) / np.sum(energy)
        floored_log2 = int(np.floor(mean_log2_energy / 2))
        scale = 2 ** (-floored_log2)

        quantized_weights = posit_quantize(
            module.weight.data.float(), nsize=nsize, es=es, scale=scale
        )
        module.weight.data = quantized_weights
        print(f"Quantized {name}.weight with scale 2^-{floored_log2} = {scale:.2e}")

        module.register_forward_pre_hook(forward_pre_hook_linear)

        if isinstance(module, modeling_utils.Conv1D):
            op_count += module.weight.shape[0] * module.weight.shape[1]
        else:
            op_count += module.in_features * module.out_features

    elif isinstance(module, nn.Embedding):
        print(f"Processing embedding layer: {name}")
        weights_flattened = module.weight.data.cpu().numpy().flatten()
        abs_weights = np.abs(weights_flattened) + epsilon
        log2_abs = np.log2(abs_weights)
        log2_energy = 2 * log2_abs
        energy = weights_flattened ** 2
        mean_log2_energy = np.sum(log2_energy * energy) / np.sum(energy)
        floored_log2 = int(np.floor(mean_log2_energy / 2))
        scale = 2 ** (-floored_log2)

        quantized_weights = posit_quantize(
            module.weight.data.float(), nsize=nsize, es=es, scale=scale
        )
        module.weight.data = quantized_weights
        print(f"Quantized embedding {name}.weight with scale 2^-{floored_log2} = {scale:.2e}")

print("Total layers processed:", layer_count)
print("MAC operation count:", op_count)

# -------------------- Direct Text Generation --------------------
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
        # set_seed(42)
        outputs = model.generate(
            input_ids, max_length=50, num_return_sequences=3, do_sample=True
        )
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))
            print("-" * 35)

print("\nGenerated text after Posit quantization:\n")
generate_text_direct(model, tokenizer)

# -------------------- Perplexity Evaluation --------------------
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
