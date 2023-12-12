# Contains the class to convert lcFlux in a folder into npy
# Example with folder contaning light curve

import numpy as np
import os
from natsort import natsorted
import time


# !!! Change the input and output file path in the bottom

class SaveLcAsNpy:
    def __init__(self, raw_lc_dir, output_npy_lc_path):
        self.std_len_lc = 120
        # shape_dir - folder where shapes in png are stored
        # name_npy_file - name of the npy file to be saved as. Include the extension
        self.name_npy_file = output_npy_lc_path
        self.lcFlux_dir = str(raw_lc_dir)
        self.lcFlux_filenames = natsorted(os.listdir(self.lcFlux_dir))
        print('lcFlux_filenames = ', self.lcFlux_filenames)
        self.no_files = len(self.lcFlux_filenames)
        print('No. of files = length - lcFlux_filenames = ', self.no_files)
        temp = np.loadtxt(self.lcFlux_dir+self.lcFlux_filenames[0], delimiter=',')
        # temp_shape = np.array(np.shape(temp))
        # print("np.shape(t) = ",temp_shape[0])
        lc_dict = np.zeros((self.no_files,self.std_len_lc))
        print('shape_dict = ',lc_dict.shape)
             
        i = 0
        for lc_element in self.lcFlux_filenames:
            temp_array = np.loadtxt(self.lcFlux_dir + lc_element, delimiter=',')
            lc_dict[i] = self.process_lc(temp_array)
            i = i + 1
        
        np.save(self.name_npy_file, lc_dict)
        lc_dict_read = np.load(self.name_npy_file)
        print('lc_dict = ', lc_dict_read)
        print('lc_dict = ', lc_dict_read.shape)
        print('All the LC are converted into single npy file')

    def process_lc(self,temp_array):
        self.temp_lc = temp_array
        test_lc_scaled_append = np.ones(120)
        test_lc_scaled_append[15:105] = self.temp_lc
        return test_lc_scaled_append
    print("Extended the light curves")

    def __del__(self):
        print('Destructor called, obj deleted.')


# What is the folder path for the input raw light curves?
raw_lc_dir = '/scratch/abraham/Documents/mega_git/mega/data/train/raw/lc/lc_planet_oblate/lc_1_planet_oblate_circle/'

# Where do you want to save the output npy light curves file?
# Include the full path including file name
output_npy_lc_path = '/scratch/abraham/Documents/mega_git/mega/data/train/npy/lc/lc_planet_oblate/lc_1_planet_oblate_circle.npy' 

print('raw_lc_dir = ',raw_lc_dir)
print('output_npy_lc_path = ',output_npy_lc_path)
user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    # Place the code you want to run here
    print("Running the code...")
    t = SaveLcAsNpy(raw_lc_dir=raw_lc_dir,output_npy_lc_path=output_npy_lc_path)
    del t
else :
    print("Exiting the program.")
toc = time.perf_counter()
print(toc-tic, " s")