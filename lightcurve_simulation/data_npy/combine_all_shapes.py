# Code to Concatenate all shapes in 
# Test Dataset (done) 
# Vald Dataset ()
# Train Dataset ()
# to single npy file
import numpy as np

# Train
# List of file names to be combined
folder_path = '../../data/train/npy/shape/'
file_names = [folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy',
folder_path + 'shape_5.npy']
# Initialize an empty list to store the arrays
arrays = []

print("Files to combine:")
print(file_names)
save_path = folder_path+'train_shape_5_9times.npy'
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

#--#
#--#
#--#
#--#

# # Vald
# # List of file names to be combined
# folder_path = '../../data/vald/npy/shape/'
# file_names = [folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy']
# # Initialize an empty list to store the arrays
# arrays = []

# print("Files to combine:")
# print(file_names)
# save_path = folder_path+'vald_shape_1_9times.npy'
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
#     np.save(save_path, combined_array)
#     print("Combined files saved in "+" "+save_path)
# else:
#     print("Exiting the program!")

#--#
#--#
#--#
#--#

# # Test
# # List of file names to be combined
# folder_path = '../../data/test/npy/shape/'
# file_names = [folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy',
# folder_path + 'shape_1.npy']
# # Initialize an empty list to store the arrays
# arrays = []

# print("Files to combine:")
# print(file_names)
# save_path = '../../data/test/npy/shape/test_shape_1_9times.npy'
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
#     np.save(save_path, combined_array)
#     print("Combined files saved in "+" "+save_path)
# else:
#     print("Exiting the program!")
