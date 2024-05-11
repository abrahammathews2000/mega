import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from pll_lc_curve_generator import LcGenerator
# from EightBitTransit import TransitingImage

import os
import time

# Where do you want to save the lcs?
# !! Change this everytime you run !!
save_lc_in_folder = '/scratch/abraham/Documents/mega_git/mega/data/test/raw/lc/lightcurve_10_may_2024_shapes/'
# save_lc_in_folder = '../../data/test/raw/lc/lc_17_shape_1/' # For Test
# save_lc_in_folder = '../../data/vald/raw/lc/lc_9_shape_1/' # For Validation
#

# Which shapes do you need to simulate to get the lcs?
# Give the npy file path for the shape
shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/test/npy/shape/10_may_2024_shapes.npy'
# shape_dir = '../../data/test/npy/shape/shape_1.npy' # For Test
# shape_dir = '../../data/vald/npy/shape/shape_1.npy' # For Validation

# if not os.path.exists(save_lc_in_folder):
#     os.mkdir(save_lc_in_folder)


# Specifiy the limb darkening coefficient
LD_Coeff = [0.5,0.05]

# Specifiy the size ratio between star and megastructure
star2mega_ratio_array = np.array([4])

# Do you want to save the time array? (True/False)
save_time = False # True

print('star2mega_ratio_array =' ,star2mega_ratio_array)
print("LD_Coeff = ",LD_Coeff)
print("shape_dir = ",shape_dir)
print("save_lc_in_folder = ",save_lc_in_folder)
print("Do you want to save the time array?", save_time)

user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    os.makedirs(save_lc_in_folder, exist_ok=True)
    if save_time == True:
        time_folder = save_lc_in_folder+"time/"
        os.makedirs(time_folder, exist_ok=True)
        print("Time array folder to be saved in: ", time_folder)    
    for i in np.arange(0,len(star2mega_ratio_array),1):
        lc = LcGenerator(shape_dir=shape_dir,
                        LD_Coeff = LD_Coeff,
                        save_lc_folder_name=save_lc_in_folder,
                        star2mega_radius_ratio=star2mega_ratio_array[i],                     
                        save_time=save_time)
        del lc
else:
    print("Exiting the program.")
toc = time.perf_counter()
print(toc-tic, " s")

