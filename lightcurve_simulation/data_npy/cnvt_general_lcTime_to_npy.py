# Code to convert time array saved as csv to npy file
# Interpolation/Downsampling can be done


import numpy as np
import os
from natsort import natsorted
import time

# !!! Change the input and output file path in the bottom

class SaveLcAsNpy:
    def __init__(self, raw_time_lc_dir, output_npy_time_lc_path,std_len_lc=100):
        self.std_len_lc = std_len_lc
        # shape_dir - folder where shapes in png are stored
        # name_npy_file - name of the npy file to be saved as. Include the extension
        self.name_npy_file = output_npy_time_lc_path
        self.lcTime_dir = str(raw_time_lc_dir)
        self.lcTime_filenames = natsorted(os.listdir(self.lcTime_dir))
        print('lcTime_filenames = ', self.lcTime_filenames)
        self.no_files = len(self.lcTime_filenames)
        print('No. of files = length - lcTime_filenames = ', self.no_files)
        # temp = np.loadtxt(self.lcTime_dir+self.lcTime_filenames[0], delimiter=',')
        # temp_shape = np.array(np.shape(temp))
        # print("np.shape(t) = ",temp_shape[0])
        lcTime_dict = np.zeros((self.no_files,self.std_len_lc))
        print('shape_dict = ',lcTime_dict.shape)
             
        i = 0
        for lc_element in self.lcTime_filenames:
            temp_array = np.loadtxt(self.lcTime_dir + lc_element, delimiter=',')
            lcTime_dict[i] = self.process_lc(temp_array)
            i = i + 1
        
        np.save(self.name_npy_file, lcTime_dict)
        lcTime_dict_read = np.load(self.name_npy_file)
        print('lcTime_dict = ', lcTime_dict_read)
        print('lcTime_dict = ', lcTime_dict_read.shape)
        print('All the time components of LC are converted into single npy file')

    def process_lc(self,temp_array):
        self.temp_lc = temp_array
        self.length_one_lc = len(self.temp_lc)
        if self.length_one_lc >= self.std_len_lc:
            self.processed_lc = self.down_sample(self.temp_lc)
        elif self.length_one_lc < self.std_len_lc:
            self.processed_lc = self.interpolate(self.temp_lc)
        return self.processed_lc

    def down_sample(self,to_down_sample_lc):
        self.to_down_sample = to_down_sample_lc
        dwn_sample_idx = np.round(np.linspace(0, self.length_one_lc - 1, self.std_len_lc)).astype(int)
        down_sample_lc = self.to_down_sample[dwn_sample_idx]
        print("Downsampled original lightcurve")
        return down_sample_lc
    
    def interpolate(self,to_interpolate_lc):
        self.to_interpolate_lc = to_interpolate_lc
        xp = np.linspace(0, self.length_one_lc - 1,self.length_one_lc)
        x_eval = np.linspace(0, self.length_one_lc - 1, self.std_len_lc)
        interpolate_lc = np.interp(x = x_eval, xp=xp, fp = self.to_interpolate_lc)
        print("Interpolated the original lightcurve")
        return interpolate_lc


    def __del__(self):
        print('Destructor called, obj deleted.')


# What is the folder path for the input raw time axis of the light curves?
raw_time_lc_dir = '/scratch/abraham/Documents/mega_git/mega/data/random/raw/lc/14Feb2024_circle_38_38_px/time/'

# Where do you want to save the output npy light curves file?
# Include the full path including file name and extension
output_npy_time_lc_path = '/scratch/abraham/Documents/mega_git/mega/data/random/npy/lc/time/time_lc_1_14Feb2024_circle_38_38_px.npy'
std_len_lc = 100 # 100 for lc directly out of 8bit # 120 for already appended lc 

print('raw_time_lc_dir = ',raw_time_lc_dir)
print('output_npy_time_lc_path = ',output_npy_time_lc_path)
print('std_len_lc = ',std_len_lc)

user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    # Place the code you want to run here
    print("Running the code...")
    t = SaveLcAsNpy(raw_time_lc_dir=raw_time_lc_dir,output_npy_time_lc_path=output_npy_time_lc_path,std_len_lc=std_len_lc)
    del t
else :
    print("Exiting the program.")
toc = time.perf_counter()
print(toc-tic, " s")