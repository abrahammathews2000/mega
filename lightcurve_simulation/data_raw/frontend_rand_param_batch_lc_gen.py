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
save_lc_in_folder = '/scratch/abraham/Documents/mega_git/mega/data/train/raw/lc/lc_12_shape_5_multisize_multiLDC/' # For Train

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

user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    print("Running the program...")
    os.makedirs(save_lc_in_folder, exist_ok=True)
    lc = LcGenerator(shape_dir=shape_dir, 
                     save_lc_folder_name=save_lc_in_folder,
                     LD_Coeff_dist_start=LD_Coeff_dist_start,
                     LD_Coeff_dist_stop=LD_Coeff_dist_stop,
                     star2mega_radius_ratio_start=star2mega_radius_ratio_start,
                     star2mega_radius_ratio_stop=star2mega_radius_ratio_stop)
    del lc
else:
    print("Exiting the program.")

toc = time.perf_counter()
print(toc-tic, " s")
