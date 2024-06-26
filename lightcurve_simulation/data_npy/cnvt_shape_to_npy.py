# Contains the class to convert shapes in a folder into npy
# Example with folder contaning shapes

import numpy as np
import os
from natsort import natsorted
import imageio.v3 as iio
import matplotlib.pyplot as plt
import time

# Change this everytime you run
index_shapefolder = 8
#


tic = time.perf_counter()


class SaveAsNpy:
    def __init__(self, shape_dir, name_npy_file):
        # shape_dir - folder where shapes in png are stored
        # name_npy_file - name of the npy file to be saved as. Don't include the extension
        self.name_npy_file = name_npy_file
        self.shape_dir = str(shape_dir)
        self.shape_filenames = natsorted(os.listdir(self.shape_dir))
        print('shape_filenames = ', self.shape_filenames)
        self.no_files = len(self.shape_filenames)
        print('length - shape_filenames = ', self.no_files)
        temp = np.array(iio.imread(self.shape_dir+self.shape_filenames[0]))
        temp_shape = np.array(np.shape(temp))
        print("np.shape(t) = ",temp_shape[0],temp_shape[1])
        shape_dict = np.zeros((self.no_files,temp_shape[0],temp_shape[1]))
        print('shape_dict = ',shape_dict)

        i = 0
        for shape_element in self.shape_filenames:
            shape_dict[i] = np.array(iio.imread(self.shape_dir+shape_element))
            i = i + 1
        shape_dict = np.where(shape_dict > (0.15*255.0), 255.0, 0)
        # Saved png image 255 = white = background
        # if pixel value > 0.15*255.0 --> pixel value = 255 
        # otherwise pixel value = 0

        np.save('/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/'+str(name_npy_file)+'.npy', shape_dict)
        shape_dict_read = np.load('/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/'+str(name_npy_file)+'.npy')

        print('shape_dict = ',shape_dict_read)
        plt.imshow(shape_dict_read[0],cmap='gray')
        plt.show()
        print('All the filled shapes are converted into single npy file')
    def __del__(self):
        print('Destructor called, obj deleted.')

j = SaveAsNpy('/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/shape/'+str(index_shapefolder)+'/','shape_filled'+str(index_shapefolder))
del j

toc = time.perf_counter()
print("Took (s): ",toc-tic)

# j = SaveAsNpy('./generatedData/shape_unfilled/','shape_unfilled')
# del j
# shape_filled_dir = './generatedData/shape_filled/'
# shape_filenames = natsorted(os.listdir(shape_dir))
# print('shape_filenames = ', shape_filenames)
# no_files = len(shape_filenames)
# print('length - shape_filenames = ', no_files)
#
# temp = np.array(iio.imread(shape_dir + shape_filenames[0]))
# temp_shape = np.array(np.shape(temp))
# print("np.shape(t) = ", temp_shape[0], temp_shape[1])
#
# shape_dict = np.zeros((no_files, temp_shape[0], temp_shape[1]))
# print('shape_dict = ', shape_dict)
#
# i = 0
# for shape_element in shape_filenames:
#     shape_dict[i] = np.array(iio.imread(shape_dir + shape_element))
#     i = i + 1
# shape_dict = np.where(shape_dict > (0.15 * 255.0), 255.0, 0)
#
# np.save('./generatedData/filled_shape_dict.npy', shape_dict)
# shape_dict_read = np.load('./generatedData/filled_shape_dict.npy')
#
# print('shape_dict = ', shape_dict_read)
# plt.imshow(shape_dict_read[0], cmap='gray')
# plt.show()
# print('All the filled shapes are converted into single npy file')
