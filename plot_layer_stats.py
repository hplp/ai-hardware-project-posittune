import json
import matplotlib.pyplot as plt

# Load saved stats
with open("posit_quantization_log.json", "r") as f:
    quant_log = json.load(f)

with open("layer_weight_stats.json", "r") as f:
    weight_stats = json.load(f)

# Ensure the layers are in the same order
layer_names = sorted(set(quant_log.keys()) | set(weight_stats.keys()))

# Collect stats
scales = []
sqnrs = []
means = []
medians = []
log2_means = []

for name in layer_names:
    q = quant_log.get(name, {})
    s = weight_stats.get(name, {})

    scales.append(q.get("log2_scale", None))
    sqnrs.append(q.get("sqnr", None))
    means.append(s.get("mean_weight", None))
    medians.append(s.get("median_weight", None))
    log2_means.append(s.get("mean_log2_abs_weight", None))

# Plotting
def plot_stat(y, title, ylabel):
    plt.figure(figsize=(12, 4))
    plt.plot(y, marker='o')
    plt.title(title)
    plt.xlabel("Layer index")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_stat(scales, "Log2(Scale) per Layer", "log2(scale)")
plot_stat(sqnrs, "SQNR per Layer", "SQNR (dB)")
plot_stat(means, "Mean Weight per Layer", "Mean(weight)")
plot_stat(medians, "Median Weight per Layer", "Median(weight)")
plot_stat(log2_means, "Mean log2(|Weight|) per Layer", "Mean log2(|w|)")
