import cv2
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans

def calculate_dominant_color(image_path, n_colors=1):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Mask out black pixels (all channels are 0)
    non_black_pixels = image[(image[:, :, 0] > 0) | (image[:, :, 1] > 0) | (image[:, :, 2] > 0)]

    # Check if there are non-black pixels
    if non_black_pixels.shape[0] == 0:
        raise ValueError("Image contains no non-black pixels")

    # Apply KMeans to find the dominant color
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(non_black_pixels)

    # Get the RGB color of the cluster center (dominant color)
    dominant_color = kmeans.cluster_centers_.astype(int)

    return dominant_color[0]  # Return the dominant color (first cluster center)

def rgb_to_grayscale(r, g, b):
    # Convert RGB to grayscale using the given formula
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale

# Folder path to the images
folder_path = 'D:\\Thesis\\Dataset\\CML\\Neucle'

# List to store image names and grayscale intensities
image_data = []

# Iterate through all images in the cytoplasm folder
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    
    # Check if the file is an image
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Calculate dominant color
            dominant_rgb = calculate_dominant_color(image_path)

            # Convert the dominant RGB color to grayscale intensity
            grayscale_value = rgb_to_grayscale(*dominant_rgb)

            # Add the image name and grayscale intensity to the list
            image_data.append([filename, grayscale_value])

        except ValueError as e:
            print(f"Skipping {filename}: {e}")

# Save results to CSV
df = pd.DataFrame(image_data, columns=['Image Name', 'Grayscale Intensity'])
csv_file_path = 'CML_Neucle_Color.csv'
df.to_csv(csv_file_path, index=False)

print(f'Results saved to {csv_file_path}')
