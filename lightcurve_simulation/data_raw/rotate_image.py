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
            rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
            rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            new_filename_90 = "ra_" + filename
            new_filename_180 = "rb_" + filename
            new_filename_270 = "rc_" + filename
            
            output_path_90 = os.path.join(output_folder, new_filename_90)
            output_path_180 = os.path.join(output_folder, new_filename_180)
            output_path_270 = os.path.join(output_folder, new_filename_270)
            
            cv2.imwrite(output_path_90, rotated_90)
            cv2.imwrite(output_path_180, rotated_180)
            cv2.imwrite(output_path_270, rotated_270)
            
            print(f"Saved: {output_path_90}")
            print(f"Saved: {output_path_180}")
            print(f"Saved: {output_path_270}")

print("Processing complete.")