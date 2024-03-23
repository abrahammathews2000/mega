import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from backend_rand_param_batch_lc_gen import LcGenerator
from natsort import natsorted

import os
import time

# Which shapes do you need to simulate to get the lcs?
# Give the npy folder path where the batches are saved
shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/train/npy/shape/shape_filled5_batches/' # For Train

# Where do you want to save the lcs?
# !! Change this everytime you run !!
save_lc_in_folder = '/scratch/abraham/Documents/mega_git/mega/data/train/raw/lc/lc_10_shape_multisize_multiLDC/' # For Train

print("Which shapes do you need to simulate to get the lcs? \n ",shape_dir)
print("Where do you want to save the lcs? \n ",save_lc_in_folder)

print("LD_Coeff_dist_start = [0.1,0.0], LD_Coeff_dist_stop = [0.9,0.2],","\n",
                    "star2mega_radius_ratio_start = 3, star2mega_radius_ratio_stop = 20")
user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    print("Running the program...")
    os.makedirs(save_lc_in_folder, exist_ok=True)
    lc = LcGenerator(shape_dir = shape_dir, save_lc_folder_name = save_lc_in_folder,
                    LD_Coeff_dist_start = [0.1,0.0], LD_Coeff_dist_stop = [0.9,0.2],
                    star2mega_radius_ratio_start = 3, star2mega_radius_ratio_stop = 20)
    del lc
else:
    print("Exiting the program.")

toc = time.perf_counter()
print(toc-tic, " s")


# # Print the size of one batch for testing
# y = np.load(shape_dir+'shape_filled5_b0.npy')
# print(len(y))