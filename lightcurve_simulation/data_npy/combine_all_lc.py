# Code to combine all light curves in 
# Test Dataset (completed) 
# Vald Dataset (completed)
# Train Dataset (completed)
# to single npy file
import numpy as np

# Train
# List of file names to be combined
folder_path = '../../data/train/npy/lc/'
file_names = [folder_path + 'lc_1_shape_5.npy',
folder_path + 'lc_2_shape_5.npy',
folder_path + 'lc_3_shape_5.npy',
folder_path + 'lc_4_shape_5.npy',
folder_path + 'lc_5_shape_5.npy',
folder_path + 'lc_6_shape_5.npy',
folder_path + 'lc_7_shape_5.npy',
folder_path + 'lc_8_shape_5.npy',
folder_path + 'lc_9_shape_5.npy']
# Initialize an empty list to store the arrays
arrays = []

print("Files to combine:")
print(file_names)
save_path = '../../data/train/npy/lc/train_lc_1_to_9.npy'
print("Save file in: "+save_path)
choice = input("Do you like to continue (y/n)")

if (choice == 'y' or choice == 'Y'):
    # Load each .npy file and append its contents to the list
    for file_name in file_names:
        array = np.load(file_name)
        arrays.append(array)

    # Combine the arrays into a single array
    combined_array = np.concatenate(arrays, axis=0)
    print("All files combined")
    # Save the combined array to a single .npy file
    np.save(save_path, combined_array)
    print("Combined files saved in "+" "+save_path)
else:
    print("Exiting the program!")


# --- #
# --- #
# --- #


# # Vald
# # List of file names to be combined
# folder_path = '../../data/vald/npy/lc/'
# file_names = [folder_path + 'lc_1_shape_1.npy',
# folder_path + 'lc_2_shape_1.npy',
# folder_path + 'lc_3_shape_1.npy',
# folder_path + 'lc_4_shape_1.npy',
# folder_path + 'lc_5_shape_1.npy',
# folder_path + 'lc_6_shape_1.npy',
# folder_path + 'lc_7_shape_1.npy',
# folder_path + 'lc_8_shape_1.npy',
# folder_path + 'lc_9_shape_1.npy']
# # Initialize an empty list to store the arrays
# arrays = []

# print("Files to combine:")
# print(file_names)
# save_path = '../../data/vald/npy/lc/vald_lc_1_to_9.npy'
# print("Save file in: "+save_path)
# choice = input("Do you like to continue (y/n)")

# if (choice == 'y' or choice == 'Y'):
#     # Load each .npy file and append its contents to the list
#     for file_name in file_names:
#         array = np.load(file_name)
#         arrays.append(array)

#     # Combine the arrays into a single array
#     combined_array = np.concatenate(arrays, axis=0)
#     print("All files combined")
#     # Save the combined array to a single .npy file
#     save_path = '../../data/vald/npy/lc/vald_lc_1_to_9.npy'
#     np.save('../../data/vald/npy/lc/vald_lc_1_to_9.npy', combined_array)
#     print("Combined files saved in "+" "+save_path)
# else:
#     print("Exiting the program!")


# --- #
# --- #
# --- #


# # Test
# # List of file names to be combined

# file_names = ['../../data/test/npy/lc/lc_1_shape_1.npy',
# '../../data/test/npy/lc/lc_2_shape_1.npy',
# '../../data/test/npy/lc/lc_3_shape_1.npy',
# '../../data/test/npy/lc/lc_4_shape_1.npy',
# '../../data/test/npy/lc/lc_5_shape_1.npy',
# '../../data/test/npy/lc/lc_6_shape_1.npy',
# '../../data/test/npy/lc/lc_7_shape_1.npy',
# '../../data/test/npy/lc/lc_8_shape_1.npy',
# '../../data/test/npy/lc/lc_9_shape_1.npy']
# # Initialize an empty list to store the arrays
# arrays = []

# # Load each .npy file and append its contents to the list
# for file_name in file_names:
#     array = np.load(file_name)
#     arrays.append(array)

# # Combine the arrays into a single array
# combined_array = np.concatenate(arrays, axis=0)

# # Save the combined array to a single .npy file
# np.save('../../data/test/npy/lc/test_lc_1_to_9.npy', combined_array)