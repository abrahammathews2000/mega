import numpy as np
import os

index_shapefolder = 5


# Load the input .npy file
input_file = f'/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled{index_shapefolder}.npy'
output_dir = f'/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled{index_shapefolder}_batches/'  # Directory to store the output .npy files
batch_size = 1000  # Number of 2D arrays per output file

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the input data
data = np.load(input_file)

# Check if the input data shape is as expected
if len(data.shape) != 3:
    raise ValueError("Input data should be a 3D array with shape (num_arrays, M, N)")

num_arrays = data.shape[0]
num_batches = num_arrays // batch_size
print('num_batches = ',num_batches)

# Split the input data into batches and save them as separate .npy files
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_data = data[start_idx:end_idx]
    output_file = os.path.join(output_dir, f'shape_filled{index_shapefolder}_b{i}.npy')
    np.save(output_file, batch_data)

# Handle any remaining arrays (if num_arrays is not a multiple of batch_size)
if num_arrays % batch_size != 0:
    start_idx = num_batches * batch_size
    end_idx = num_arrays
    remaining_data = data[start_idx:end_idx]
    output_file = os.path.join(output_dir, f'shape_filled{index_shapefolder}_b{num_batches}.npy')
    np.save(output_file, remaining_data)

print(f"Split {num_arrays} 2D arrays into {num_batches + 1} output .npy files.")