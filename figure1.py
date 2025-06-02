import numpy as np
import matplotlib.pyplot as plt

# Generate normally distributed weights
weights = np.random.normal(loc=0.0, scale=1.0, size=10000)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(weights, bins=100, density=False, alpha=0.75, color='steelblue', edgecolor='black')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Simulated Weight Distribution in a Layer (Gaussian)')
plt.grid(True)
plt.tight_layout()
plt.show()
