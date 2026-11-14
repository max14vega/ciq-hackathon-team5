from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# Folder for our experiment outputs
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Generate reproducible simulated predictions
rng = np.random.default_rng(42)

height, width = 256, 256
y, x = np.mgrid[:height, :width]

# Simulated ink-like pattern
ink = (
    np.sin(x / 12) * np.cos(y / 18) > 0.45
).astype(float)

# Three simulated model predictions
predictions = np.stack([
    np.clip(ink + rng.normal(0, 0.10, ink.shape), 0, 1),
    np.clip(ink + rng.normal(0, 0.15, ink.shape), 0, 1),
    np.clip(ink + rng.normal(0, 0.20, ink.shape), 0, 1),
])

# Calculate average prediction and disagreement
mean_prediction = predictions.mean(axis=0)
uncertainty = predictions.std(axis=0)

# Save TIFF files
tifffile.imwrite(RESULTS / "mean_prediction.tif",
                 mean_prediction.astype(np.float32))
tifffile.imwrite(RESULTS / "uncertainty.tif",
                 uncertainty.astype(np.float32))

# Create comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(predictions[0], cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Baseline Prediction")

axes[1].imshow(mean_prediction, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Average Prediction")

im = axes[2].imshow(uncertainty, cmap="inferno")
axes[2].set_title("Prediction Disagreement")

for ax in axes:
    ax.axis("off")

fig.colorbar(im, ax=axes[2], shrink=0.75)
fig.tight_layout()
fig.savefig(RESULTS / "comparison.png", dpi=180)
plt.close(fig)

print("Experiment complete!")
print("Saved results/comparison.png")
print("Saved results/mean_prediction.tif")
print("Saved results/uncertainty.tif")
print(f"Mean disagreement: {uncertainty.mean():.4f}")
