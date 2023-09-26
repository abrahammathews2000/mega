# Flip horizontally
import os
import cv2

input_folder = "ms_proj_shape_lc_gen/data_raw/shape/3/"
output_folder = "ms_proj_shape_lc_gen/data_raw/shape/3/"

# Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is not None:
            flipped_image = cv2.flip(image, 1)  # Flip horizontally (1)
            new_filename = "h_" + filename
            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, flipped_image)
            print(f"Saved: {output_path}")

print("Processing complete.")