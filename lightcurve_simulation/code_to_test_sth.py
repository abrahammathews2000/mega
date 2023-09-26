import numpy as np
import os
from natsort import natsorted
# ---
# Test 1 (7 September 2023)
# Calculate the number of elements in each
# batch of shapes in  
# /data_npy/shape_npy/shape_filled5_batches/
def test_1():
    y_dir = '/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled5_batches/'  # './generatedData/shape_dict.npy'
    batch_names = natsorted(os.listdir(y_dir))
    print('batch_names = ',batch_names)
    print('number of batches = ',len(batch_names))
    count_1000_ele = 0 
    count_not1000_ele = 0

    for batch in batch_names:
        batch_link = str(y_dir) + str(batch)
        el = np.load(batch_link)
        if len(el)==1000:
            count_1000_ele = count_1000_ele + 1
        else:
            print(batch,' contains ->',len(el),' elements')
            count_not1000_ele = count_not1000_ele + 1

    print('Number of batches with 1000 elem = ',count_1000_ele)
    print('Number of batches other than 1000 elem = ',count_not1000_ele)
# Output
# number of batches =  132
# shape_filled5_b131.npy  contains -> 136  elements
# Number of batches with 1000 elem =  131
# Number of batches other than 1000 elem =  1
# ---

# ---
# Test 2
# Calculate number of shapes in
# data_npy/shape_npy/shape_filled5.npy
def test_2():
    y_dir = '/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled5.npy'  # './generatedData/shape_dict.npy'
    # y_dir = '/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled5_old1.npy'
    shapes = np.load(y_dir)
    print('number of shapes in shape_filled5.npy = ',len(shapes))
    # print('number of shapes in shape_filled5_old1.npy = ',len(shapes))
# Output:
# number of shapes in shape_filled5_old1.npy =  132136
# ---

# ---
# Test 3
# Compare first element of shape_filled5.npy
# with first element of shape_filled5_batches/..b0.npy
# Output is showing that both 2D arrays are same, 
# Now we have to look for which batch deviates from original shapes

def test_3():
    org_shapes_link = '/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled5.npy'  # './generatedData/shape_dict.npy'
    org_shapes = np.load(org_shapes_link)


    shape_batches_link = '/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled5_batches/'  # './generatedData/shape_dict.npy'
    batch_names = natsorted(os.listdir(shape_batches_link))

    batch_i_element = 0
    org_element = 0
    batch_error = 0
    batch_error_count = 0
    for batch in batch_names:
        batch_shapes = np.load(str(shape_batches_link)+str(batch))  # str(batch_names[batch_i]))
        print("Batch = ",batch)
        if np.array_equal(org_shapes[org_element], batch_shapes[batch_i_element]):
            print(f'org_shapes{org_element} == batch_shapes[{batch_i_element}]')
        else:
            print(f'org_shapes{org_element} NOT batch_shapes[{batch_i_element}]')
            batch_error = batch
            batch_error_count = batch_error_count + 1
        
        org_element = org_element + 1000

    print('No. of discrepancy in first shapes = ',batch_error_count)
    print("There is an error in batch = ",batch_error)

#--
# Calculate the number of batches
def test_4():
    index_shapefolder = 5

    # Load the input .npy file
    input_file = f'/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled{index_shapefolder}.npy'
    batch_size = 1000  # Number of 2D arrays per output file

    # Load the input data
    data = np.load(input_file)

    # Check if the input data shape is as expected
    if len(data.shape) != 3:
        raise ValueError("Input data should be a 3D array with shape (num_arrays, M, N)")

    num_arrays = data.shape[0]
    num_batches = num_arrays // batch_size
    print('num_batches = ',num_batches)


# Run
# test_4()
# test_3()
# test_2()
test_1()
