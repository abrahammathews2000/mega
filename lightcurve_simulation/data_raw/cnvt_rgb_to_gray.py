# Convert RGB Images to Gray scale
# Code made first to convert images in folder 3
# Aug 30, 2023

import cv2
import os

# Set the path to the folder containing the images
image_folder = '/scratch/abraham/Documents/mega_git/mega/data/test/raw/shape/10_may_2024_shapes/'

# Loop through all the files in the folder
for filename in os.listdir(image_folder):
    # Check if the file is an image file
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Open the image using OpenCV
        img = cv2.imread(os.path.join(image_folder, filename))
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save the grayscale image by overwriting the original image
        cv2.imwrite(os.path.join(image_folder, filename), gray)
        print(f"Converted and saved: {filename}")
    else:
        print(f"Could not read: {filename}")
print("Conversion and saving complete.")
