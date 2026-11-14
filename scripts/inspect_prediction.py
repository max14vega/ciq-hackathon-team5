
import numpy as np
import tifffile

path = "predictions/w043_step-075000.tif"
image = tifffile.imread(path)

print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum:", image.min())
print("Maximum:", image.max())
print("Mean:", image.mean())
print("Standard deviation:", image.std())
