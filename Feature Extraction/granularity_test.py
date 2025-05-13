import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from scipy import ndimage as ndi
from skimage import feature
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

# Load image
img_path = '4_32_44_1000_CLL_14.png'
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------- a. Texture Analysis ---------- #

def texture_features(img):
    # GLCM
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]


    print("GLCM Features:")
    print(f"Contrast: {contrast:.3f}, Homogeneity: {homogeneity:.3f}, Energy: {energy:.3f}, Correlation: {correlation:.3f}")

    # LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')

    # Gabor filter
    filt_real, filt_imag = gabor(img, frequency=0.6)
    props = regionprops(filt_real)
    granule_count = len(props)
    print(granule_count)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    axs[0].imshow(lbp, cmap='gray')
    axs[0].set_title('Local Binary Pattern')
    axs[1].imshow(filt_real, cmap='gray')
    axs[1].set_title('Gabor Filter (Real)')
    axs[2].imshow(filt_imag, cmap='gray')
    axs[2].set_title('Gabor Filter (Imag)')
    for ax in axs:
        ax.axis('off')
    plt.show()

# Apply to granular cell
print("\nTexture Analysis - Granular (Yes):")
texture_features(gray)




# ---------- b. Blob Detection ---------- #

def blob_detection(img, title='Image'):
    # Laplacian of Gaussian (LoG)
    log = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    log = cv2.Laplacian(log, cv2.CV_64F)

    # Difference of Gaussian (DoG)
    gaussian1 = cv2.GaussianBlur(img, (5, 5), 1)
    gaussian2 = cv2.GaussianBlur(img, (5, 5), 2)
    dog = gaussian1 - gaussian2

    # Detect blobs using DoG + threshold
    dog_blobs = dog.copy()
    dog_blobs[dog_blobs < 10] = 0  # Adjust threshold as needed

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    axs[0].imshow(log, cmap='gray')
    axs[0].set_title('Laplacian of Gaussian (LoG)')
    axs[1].imshow(dog, cmap='gray')
    axs[1].set_title('Difference of Gaussian (DoG)')
    axs[2].imshow(dog_blobs, cmap='gray')
    axs[2].set_title('Blob Regions (Thresholded DoG)')
    for ax in axs:
        ax.axis('off')
    plt.suptitle(f"Blob Detection - {title}")
    plt.show()

# Apply to both
blob_detection(gray, 'Granular (Yes)')


def morphological_features(img, title='Image'):
    # Threshold to isolate dark granules
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove large background artifacts (keep only small regions = granules)
    labeled = label(thresh)
    granules = remove_small_objects(labeled, min_size=5, connectivity=2)

    # Count granules
    props = regionprops(granules)
    granule_count = len(props)

    # Size distribution
    areas = [p.area for p in props]

    print(f"\n🔍 Morphological Features - {title}")
    print(f"Granule Count: {granule_count}")
    print(f"Granule Size (mean): {np.mean(areas):.2f}")
    print(f"Granule Size (std): {np.std(areas):.2f}")

    # Plot histogram of sizes
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title(f'{title} - Original')
    axs[0].axis('off')

    axs[1].imshow(granules, cmap='nipy_spectral')
    axs[1].set_title(f'{title} - Granules (Labeled)')
    axs[1].axis('off')

    axs[2].hist(img.ravel(), bins=32, color='purple', alpha=0.7)
    axs[2].set_title(f'{title} - Intensity Histogram')
    axs[2].set_xlabel('Pixel Intensity')
    axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Apply to both granular and non-granular cells
morphological_features(gray, 'Granular (Yes)')

