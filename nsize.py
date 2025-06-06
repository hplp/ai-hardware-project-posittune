import json

# Load the JSON file
with open("posit_quantization_log.json", "r") as f:
    data = json.load(f)

# Extract all nsize values
nsizes = [layer_info["nsize"] for layer_info in data.values() if "nsize" in layer_info]

# Compute average nsize
average_nsize = sum(nsizes) / len(nsizes)

print(f"Number of layers: {len(nsizes)}")
print(f"Average nsize: {average_nsize:.4f}")
