import matplotlib.pyplot as plt
import cv2
from skimage import filters, measure
import numpy as np

# Load the image
new_nucleus_image_path = "D:\\Thesis\\Data extract\\4_32_12_1000_CLL_4.png"
new_nucleus_image = cv2.imread(new_nucleus_image_path)

# Convert to grayscale
new_nucleus_gray = cv2.cvtColor(new_nucleus_image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
new_nucleus_threshold_value = filters.threshold_otsu(new_nucleus_gray)
new_nucleus_binary = new_nucleus_gray > new_nucleus_threshold_value

# Compute texture feature: Shannon Entropy (as a measure of information density)
new_nucleus_entropy = measure.shannon_entropy(new_nucleus_image)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(new_nucleus_gray, cmap='gray')
axes[0].set_title("Grayscale Image")
axes[0].axis("off")

axes[1].imshow(new_nucleus_binary, cmap='gray')
axes[1].set_title("Binary Image (Otsu Threshold)")
axes[1].axis("off")

axes[2].hist(new_nucleus_gray.ravel(), bins=256, color='blue', alpha=0.7)
axes[2].set_title("Pixel Intensity Distribution")
axes[2].set_xlabel("Pixel Intensity")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

print("Shannon Entropy (Texture Measure):", new_nucleus_entropy)
