import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from backend_rand_param_lc_gen import LcGenerator
# from EightBitTransit import TransitingImage

import os
import time

# Where do you want to save the lcs?
# !! Change this everytime you run !!
save_lc_in_folder = '/scratch/abraham/Documents/mega_git/mega/data/vald/raw/lc/lc_10_shape_multisize_multiLDC/'
# save_lc_in_folder = '../../data/test/raw/lc/lc_17_shape_1/' # For Test
# save_lc_in_folder = '../../data/vald/raw/lc/lc_9_shape_1/' # For Validation
#

# Which shapes do you need to simulate to get the lcs?
# Give the npy file path for the shape
# shape_dir = 
# shape_dir = '../../data/test/npy/shape/shape_1.npy' # For Test
shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/vald/npy/shape/shape_1.npy' # For Validation

# if not os.path.exists(save_lc_in_folder):
#     os.mkdir(save_lc_in_folder)






user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    os.makedirs(save_lc_in_folder, exist_ok=True)
   
    lc = LcGenerator(shape_dir = shape_dir, save_lc_folder_name = save_lc_in_folder,
                    LD_Coeff_dist_start = [0.1,0.0], LD_Coeff_dist_stop = [0.9,0.2],
                    star2mega_radius_ratio_start = 3, star2mega_radius_ratio_stop = 20)
    del lc
else:
    print("Exiting the program.")
toc = time.perf_counter()
print(toc-tic, " s")

