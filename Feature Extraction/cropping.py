import cv2
import os
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.morphology import remove_small_objects, label

def process_image(image_path, nucleus_folder, without_nucleus_folder):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # # Convert to LAB color space
    # image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l, A_channel, b = cv2.split(image_lab)

    # # Apply multi-Otsu thresholding
    # thresholds = threshold_multiotsu(A_channel, classes=3)
    # regions = np.digitize(A_channel, bins=thresholds)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(image_hsv)
    
    # Apply multi-Otsu threshold
    thresholds = threshold_multiotsu(saturation, classes=3)
    regions = np.digitize(saturation, bins=thresholds)

    # Create binary mask for nucleus
    binary_mask_cell = (regions == 2).astype(np.uint8)
    binary_mask_bool = binary_mask_cell.astype(bool)

    # Label connected components and remove small objects
    labeled_mask, num_features = label(binary_mask_bool, return_num=True)
    cleaned_mask = remove_small_objects(labeled_mask, min_size=100)
    cleaned_binary_mask = (cleaned_mask > 0).astype(np.uint8)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(cleaned_binary_mask * 255)

    # Remove nucleus by applying the inverted mask
    background_color = (0, 0, 0)
    masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)
    background_layer = np.full_like(image, background_color)
    result_image = cv2.addWeighted(masked_image, 1, background_layer, 1, 0)

    # Extract nucleus image
    nucleus_image = cv2.bitwise_and(image, image, mask=cleaned_binary_mask)

    # Save the images
    image_name = os.path.basename(image_path)
    nucleus_output_path = os.path.join(nucleus_folder, image_name)
    without_nucleus_output_path = os.path.join(without_nucleus_folder, image_name)

    cv2.imwrite(nucleus_output_path, nucleus_image)
    cv2.imwrite(without_nucleus_output_path, result_image)
    print(f"Processed and saved: {image_name}")

def main(dataset_folder, nucleus_folder, without_nucleus_folder):
    # Create output folders if they do not exist
    os.makedirs(nucleus_folder, exist_ok=True)
    os.makedirs(without_nucleus_folder, exist_ok=True)

    # Process all images in the dataset folder
    for image_name in os.listdir(dataset_folder):
        image_path = os.path.join(dataset_folder, image_name)
        if os.path.isfile(image_path):
            process_image(image_path, nucleus_folder, without_nucleus_folder)

if __name__ == "__main__":
    dataset_folder = "D:\\Thesis\\Dataset\\100x\\ALL_BR"  # Replace with the path to your dataset
    nucleus_folder = "D:\\Thesis\\Dataset\\ALL2\\Neucle"  # Replace with the path to save nucleus images
    without_nucleus_folder = "D:\\Thesis\\Dataset\\ALL2\\Cyto"  # Replace with the path to save without nucleus images

    main(dataset_folder, nucleus_folder, without_nucleus_folder)
