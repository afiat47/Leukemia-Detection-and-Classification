import cv2
import numpy as np

from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from math import pi

from skimage.feature import canny
# Read image in grayscale
img = cv2.imread('3_27_3_1000_CML_5.png', cv2.IMREAD_GRAYSCALE)

# 1. Noise removal with median filtering
denoised = cv2.medianBlur(img, ksize=5)

# 2. Contrast enhancement with CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(denoised)

# 3. Background correction using morphological opening (top-hat)
# Create a large structuring element (disk-shaped kernel) to approximate background
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
# Subtract background and normalize the result
corrected = cv2.normalize(enhanced - background, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite('preprocessed.png', corrected)


# Assume `corrected` is the preprocessed image from above (numpy array)
# Global thresholding using Otsu's method
global_thresh = threshold_otsu(corrected)
binary_global = corrected > global_thresh

# Adaptive (local) thresholding (e.g., 51x51 neighborhood, with a slight offset)
local_thresh = threshold_local(corrected, block_size=51, offset=10)
binary_local = corrected > local_thresh

# Remove very small regions (noise) from the binary mask
binary_clean = remove_small_objects(binary_global, min_size=100)


# Label connected components (nuclei) in the binary mask
label_img = label(binary_clean)

features = []  # will hold feature vectors for each nucleus
for region in regionprops(label_img):
    area = region.area
    perimeter = region.perimeter
    # Basic shape features
    aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length>0 else 0
    eccentricity = region.eccentricity
    circularity = (4 * pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    solidity = region.solidity               # area / convex_area
    convex_perimeter = region.convex_image.sum()  # perimeter of convex hull can be approximated from convex_image
    convexity = convex_perimeter / perimeter if perimeter > 0 else 0

    features.append([area, perimeter, aspect_ratio, eccentricity, circularity, solidity, convexity])

print(features)




# Apply Canny edge detector to the preprocessed image (or nucleus patch)
edges = canny(corrected, sigma=1.0, low_threshold=50, high_threshold=150)

# Use edges to compute an edge-based feature, e.g., edge density in each nucleus
edge_density_features = []
for region in regionprops(label_img):
    # Count edge pixels within the nucleus bounding box
    minr, minc, maxr, maxc = region.bbox
    nucleus_mask = (label_img[minr:maxr, minc:maxc] == region.label)
    nucleus_edges = edges[minr:maxr, minc:maxc]
    # Edge density = fraction of nucleus area that are edge pixels
    edge_pixels = np.logical_and(nucleus_edges, nucleus_mask).sum()
    edge_density = edge_pixels / region.area
    edge_density_features.append(edge_density)

print(edge_density_features)
