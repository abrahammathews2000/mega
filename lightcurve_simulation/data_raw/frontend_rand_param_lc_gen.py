import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from backend_rand_param_lc_gen import LcGenerator
# from EightBitTransit import TransitingImage

import os
import time

# 1. change the path to saved LDC and size info in the backend code
# Here: /scratch/abraham/Documents/mega_git/mega/lightcurve_simulation/data_raw/backend_rand_param_lc_gen.py

# 2. Where do you want to save the lcs?
# !! Change this everytime you run !!
save_lc_in_folder = '/scratch/abraham/Documents/mega_git/mega/data/test/raw/lc/lc_shape_1_multisize_multiLDC_3/'
# save_lc_in_folder = '../../data/test/raw/lc/lc_17_shape_1/' # For Test
# save_lc_in_folder = '../../data/vald/raw/lc/lc_9_shape_1/' # For Validation
#

# 3. Which shapes do you need to simulate to get the lcs?
# Give the npy file path for the shape
# shape_dir = 
shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/test/npy/shape/shape_1.npy' # For Test
# shape_dir = '' # For Validation

# if not os.path.exists(save_lc_in_folder):
#     os.mkdir(save_lc_in_folder)

print("Which shapes do you need to simulate to get the lcs? \n ",shape_dir)
print("Where do you want to save the lcs? \n ",save_lc_in_folder)

LD_Coeff_dist_start = [0, 0]
LD_Coeff_dist_stop = [1, 0.55]
star2mega_radius_ratio_start = 2
star2mega_radius_ratio_stop = 30
print("LD_Coeff_dist_start=", LD_Coeff_dist_start)
print("LD_Coeff_dist_stop=", LD_Coeff_dist_stop)
print("star2mega_radius_ratio_start=", star2mega_radius_ratio_start)
print("star2mega_radius_ratio_stop=", star2mega_radius_ratio_stop)

star2mega_radius_ratio_info_save_path = "/scratch/abraham/Documents/mega_git/mega/data/test/info/lc_shape_1_multisize_multiLDC_3_details/star2megaRadius.csv"
LDC_info_save_path= "/scratch/abraham/Documents/mega_git/mega/data/test/info/lc_shape_1_multisize_multiLDC_3_details/LDCs.csv"

user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    os.makedirs(save_lc_in_folder, exist_ok=True)
   
    lc = LcGenerator(shape_dir=shape_dir, 
                     save_lc_folder_name=save_lc_in_folder,
                     star2mega_radius_ratio_info_save_path=star2mega_radius_ratio_info_save_path,
                     LDC_info_save_path=LDC_info_save_path,
                     LD_Coeff_dist_start=LD_Coeff_dist_start,
                     LD_Coeff_dist_stop=LD_Coeff_dist_stop,
                     star2mega_radius_ratio_start=star2mega_radius_ratio_start,
                     star2mega_radius_ratio_stop=star2mega_radius_ratio_stop
)
    del lc
else:
    print("Exiting the program.")
toc = time.perf_counter()
print(toc-tic, " s")

