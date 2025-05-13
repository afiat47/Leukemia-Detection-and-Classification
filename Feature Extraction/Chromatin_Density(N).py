import os
import cv2
import pandas as pd
from skimage import filters, measure

# Set your folder path
folder_path = 'D:\\Thesis\\Dataset\\ALL\\Neucle'  # <-- Change this to your folder
output_csv = "Chromatin_Density(ALL).csv"

# List to hold results
entropy_data = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        image_path = os.path.join(folder_path, filename)
        new_nucleus_image = cv2.imread(image_path)

# Convert to grayscale
        new_nucleus_gray = cv2.cvtColor(new_nucleus_image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        new_nucleus_threshold_value = filters.threshold_otsu(new_nucleus_gray)
        new_nucleus_binary = new_nucleus_gray > new_nucleus_threshold_value

        # Compute texture feature: Shannon Entropy (as a measure of information density)
        new_nucleus_entropy = measure.shannon_entropy(new_nucleus_image)
        
        # Append result
        entropy_data.append({'Image Name': filename, 'Nucleus Entropy': new_nucleus_entropy})

# Convert to DataFrame
df = pd.DataFrame(entropy_data)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Entropy data saved to {output_csv}")
