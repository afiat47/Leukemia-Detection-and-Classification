import cv2
import numpy as np
import os
import pandas as pd

# Folder path containing the nucleus images
image_folder = 'D:\\Thesis\\Dataset\\ALL\\Neucle'

# Initialize a list to store results (image name, largest area)
results = []

# Iterate over all images in the folder
for image_name in os.listdir(image_folder):
    if image_name.endswith('.png'):  # Ensure it's a PNG image file
        # Load the image
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Create a mask for non-black pixels
        mask = np.all(image_rgb != [0, 0, 0], axis=-1)

        # Create the white pixel image (non-black pixels to white)
        white_pixels_image = np.copy(image_rgb)
        white_pixels_image[mask] = [255, 255, 255]

        # Convert the mask to a binary image (0: black, 255: white)
        binary_mask = mask.astype(np.uint8) * 255

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables for the largest contour
        largest_contour = None
        largest_area = 0

        # Find the largest contour based on area
        for contour in contours:
            area = cv2.contourArea(contour)  # Area of the contour
            if area > largest_area:
                largest_area = area
                largest_contour = contour

        # If a largest contour exists, store its area and image name
        if largest_contour is not None:
            results.append([image_name, largest_area])

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['Image Name', 'Largest Area'])

# Save the DataFrame to a CSV file
csv_path = 'D:\\Thesis\\Data extract\\Nucle_Size(ALL).csv'
df.to_csv(csv_path, index=False)

print(f'Results saved to {csv_path}')
