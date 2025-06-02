import torch
import numpy as np
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from qtorch_plus.quant import posit_quantize, float_quantize
from datasets import load_dataset
from tqdm import tqdm
import transformers
import transformers.modeling_utils as modeling_utils

transformers.logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

base_nsize = 3
es = 0
epsilon = 1e-8
log2_scales = np.arange(-5,5)
sweep_scales = [2.0 ** x for x in log2_scales]
sqnr_threshold = 15.0

def linear_activation(input):
    return float_quantize(input, exp=4, man=3, rounding="nearest")

def forward_pre_hook_linear(m, input):
    return (linear_activation(input[0]),)

# def find_best_sqnr(weights, nsize):
#     signal_power = torch.sum(weights ** 2)
#     max_val = weights.abs().max()
#     norm_weights = weights / (max_val + epsilon)
    
#     best_sqnr = -float("inf")
#     best_scale = None
#     best_log2_scale = None
#     best_es = None

#     for es_candidate in [0, 1]:
#         for i, scale in enumerate(sweep_scales):
#             quantized = posit_quantize(norm_weights, nsize=nsize, es=es_candidate, scale=scale)
#             quantized_rescaled = quantized * max_val
#             noise_power = torch.sum((weights - quantized_rescaled) ** 2) + epsilon
#             sqnr = 10 * torch.log10(signal_power / noise_power)
#             if sqnr > best_sqnr:
#                 best_sqnr = sqnr
#                 best_scale = scale
#                 best_log2_scale = log2_scales[i]
#                 best_es = es_candidate

#     return best_sqnr, best_scale, best_log2_scale, best_es, max_val

def find_best_sqnr(weights, nsize, mode='sweep'):
    signal_power = torch.sum(weights ** 2)
    best_sqnr = -float("inf")
    best_scale = None
    best_log2_scale = None
    best_es = None

    if mode == 'rms':
        rms = torch.sqrt(torch.mean(weights ** 2))
        for es_candidate in [0, 1]:
            scale = 1.0 / (rms + epsilon)
            norm_weights = weights * scale
            quantized = posit_quantize(norm_weights, nsize=nsize, es=es_candidate, scale=1.0)
            quantized_rescaled = quantized / scale
            noise_power = torch.sum((weights - quantized_rescaled) ** 2) + epsilon
            sqnr = 10 * torch.log10(signal_power / noise_power)
            if sqnr > best_sqnr:
                best_sqnr = sqnr
                best_scale = scale
                best_log2_scale = np.log2(scale.item())
                best_es = es_candidate
        return best_sqnr, best_scale, best_log2_scale, best_es, 1.0  # already scaled, so max_val is 1

    else:
        max_val = weights.abs().max()
        norm_weights = weights / (max_val + epsilon)
        for es_candidate in [0, 1]:
            for i, scale in enumerate(sweep_scales):
                quantized = posit_quantize(norm_weights, nsize=nsize, es=es_candidate, scale=scale)
                quantized_rescaled = quantized * max_val
                noise_power = torch.sum((weights - quantized_rescaled) ** 2) + epsilon
                sqnr = 10 * torch.log10(signal_power / noise_power)
                if sqnr > best_sqnr:
                    best_sqnr = sqnr
                    best_scale = scale
                    best_log2_scale = log2_scales[i]
                    best_es = es_candidate
        return best_sqnr, best_scale, best_log2_scale, best_es, max_val

    


layer_count = 0
op_count = 0

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, modeling_utils.Conv1D)):
        layer_count += 1
        print(f"Processing layer: {name}")
        weights = module.weight.data.detach().float().cpu()

        sqnr, scale, log2_scale, best_es, max_val = find_best_sqnr(weights, base_nsize)
        final_nsize = base_nsize
        # if sqnr < sqnr_threshold:
        #     print(f"  SQNR = {sqnr:.2f} < {sqnr_threshold} dB → using nsize = {base_nsize + 1}")
        #     final_nsize = base_nsize + 1
        #     sqnr, scale, log2_scale, max_val = find_best_sqnr(weights, final_nsize)
        while sqnr < sqnr_threshold:
            final_nsize += 1
            sqnr, scale, log2_scale, best_es, max_val = find_best_sqnr(weights, final_nsize)
        print(f"  Final SQNR = {sqnr:.2f} dB, log2(scale) = {log2_scale}, nsize = {final_nsize}, es = {best_es}")


        norm_weights = weights / (max_val + epsilon)
        quantized = posit_quantize(norm_weights, nsize=final_nsize, es=best_es, scale=scale)
        module.weight.data = (quantized * max_val).to(module.weight.data.device)
        module.register_forward_pre_hook(forward_pre_hook_linear)

        # if isinstance(module, modeling_utils.Conv1D):
        #     op_count += module.weight.shape[0] * module.weight.shape[1]
        # else:
        #     op_count += module.in_features * module.out_features

    # elif isinstance(module, nn.Embedding):
    #     print(f"Processing embedding layer: {name}")
    #     weights = module.weight.data.detach().float().cpu()
    #     sqnr, scale, log2_scale, max_val = find_best_sqnr(weights, base_nsize)
    #     norm_weights = weights / (max_val + epsilon)
    #     while sqnr < sqnr_threshold:
    #         final_nsize = base_nsize + 1
    #         sqnr, scale, log2_scale, max_val = find_best_sqnr(weights, final_nsize)
    #     quantized = posit_quantize(norm_weights, nsize=final_nsize, es=es, scale=scale)
    #     module.weight.data = (quantized * max_val).to(module.weight.data.device)
    #     print(f"  Embedding SQNR = {sqnr:.2f} dB, log2(scale) = {log2_scale}, nsize = {final_nsize}")
    elif isinstance(module, nn.Embedding):
        print(f"Processing embedding layer: {name}")
        weights = module.weight.data.detach().float().cpu()

        final_nsize = base_nsize
        sqnr, scale, log2_scale, best_es, max_val = find_best_sqnr(weights, final_nsize)
        while sqnr < sqnr_threshold:
            final_nsize += 1
            sqnr, scale, log2_scale, best_es, max_val = find_best_sqnr(weights, final_nsize)

        norm_weights = weights / (max_val + epsilon)
        quantized = posit_quantize(norm_weights, nsize=final_nsize, es=best_es, scale=scale)
        module.weight.data = (quantized * max_val).to(module.weight.data.device)

        print(f"  Embedding SQNR = {sqnr:.2f} dB, log2(scale) = {log2_scale}, nsize = {final_nsize}, es = {best_es}")


print("Total layers processed:", layer_count)
print("MAC operation count:", op_count)

# -------------------- Generate Text --------------------
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
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=3, do_sample=True)
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))
            print("-" * 35)

print("\nGenerated text after Posit quantization:\n")
generate_text_direct(model, tokenizer)

# -------------------- Compute Perplexity --------------------
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
    print("\nPerplexity after quantization:", ppl.item())
else:
    print("No log likelihoods calculated.")


# result: Generated text after Posit quantization:

# Machine learning is the study of solving problems by using machine learning techniques using machine learning techniques and statistical models. This can be used to predict outcomes from large data sets that cannot be analyzed by human intervention. It provides a data mining approach that can help predict
# -----------------------------------
# Machine learning is the study of the way that some systems or applications work, so that they can better be designed so that they perform better. A good example of this is the ability to better classify and/or classify correctly.

# For example,
# -----------------------------------
# Machine learning is the study of patterns and how they arise at the very core of each piece of software (such as how the computer uses a keyboard, or how the software in a computer works or what the computer is, or what the computer has been
# -----------------------------------
# In the 19th century, the invention of the phonograph gave new ideas to the way in which music for the deaf could be played. But from the end of the 20th century until the end of the 20th century, there was little innovation
# -----------------------------------
# In the 19th century, the invention of the automobile came about, resulting in an increase in the size of the globe, which was then 2.7 times as large as it was at the turn of the century.

# The human population of
# -----------------------------------
# In the 19th century, the invention of the rifle changed all that. Most firearms were single-shot rifles or were capable of firing more powerful bullets and cartridges. The first fully automatic rifles became effective at that time, and most modern rifles are still
# -----------------------------------
# A robot was created which is able to "see" through a cloth.

# Researchers from Newcastle University set out to create a small, cheap and user-friendly robot that could sense and react to its surroundings.

# The results of its research
# -----------------------------------
# A robot was created. It was small in size, but still had great features. It had great features.

# ROSES

# A robot was created. It was small in size but still had great features. It had great features.
# -----------------------------------
# A robot was created that's capable of doing everything a human can! The most interesting thing about it is the fact that it isn't a machine." https://www.reddit.com/r/politics/comments/5zf6zv
# -----------------------------------
# One day I will meet him, in a future, where all of his friends have met his friends," he added.
# -----------------------------------
# One day I will say to you, 'You're a star.' You're going to be a star. You're not going to be able to contain a mouse and a drop of oil.' It could have been a really big, you know,
# -----------------------------------
# One day I will take my place on the stage and I will play and I will play until I die. We'll be together forever.' These are not the words of a child."

# But, he added, he "would never take that
# -----------------------------------
# Input size: 287644
# 100%|███████████████████████████████████████████████████████████████████| 281/281 [01:27<00:00,  3.21it/s]

# Perplexity after quantization: 22.303539276123047