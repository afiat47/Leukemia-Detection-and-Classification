import cv2
import numpy as np
import os
import csv

# Define the path to the 'cell' folder
cell_folder_path = 'D:\\Thesis\\Dataset\\ALL\\Neucle'

# List all files in the folder
image_files = [f for f in os.listdir(cell_folder_path) if f.endswith('.png')]  # Adjust for other image formats if necessary

# Open the CSV file for writing
csv_file_path = 'ALL_Neucle_Size.csv'  # Path where the CSV will be saved
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Area', 'Circularity'])  # Write the header

    # Loop over all the images in the folder
    for image_name in image_files:
        # Load the image
        image_path = os.path.join(cell_folder_path, image_name)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Create a mask for non-black pixels
        mask = np.all(image_rgb != [0, 0, 0], axis=-1)

        # Convert the mask to a binary image (0: black, 255: white)
        binary_mask = mask.astype(np.uint8) * 255

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there is exactly one contour, calculate area and circularity
        if contours:
            contour = contours[0]  # Only one contour
            perimeter = cv2.arcLength(contour, True)  # Perimeter of the contour
            area = cv2.contourArea(contour)  # Area of the contour

            # Calculate circularity
            if perimeter > 0:  # Avoid division by zero
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0  # No valid contour, set circularity to 0

            # Save the image name, area, and circularity in the CSV file
            writer.writerow([image_name, circularity])

print(f'CSV file saved at {csv_file_path}')
