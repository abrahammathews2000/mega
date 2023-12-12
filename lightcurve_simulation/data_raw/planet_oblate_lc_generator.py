import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from pll_planet_oblate_lc_gen_class import LcGenerator
# from EightBitTransit import TransitingImage

import os
import time

# Where do you want to save the lcs?
# !! Change this everytime you run !!
save_lc_in_folder = '/scratch/abraham/Documents/mega_git/mega/data/train/raw/lc/lc_planet_oblate/lc_1_planet_oblate/' # For Train
#

# Which shapes do you need to simulate to get the lcs?
# Give the npy file path for the shape
shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/train/npy/shape/shape_oblate.npy' # For Planet

# if not os.path.exists(save_lc_in_folder):
#     os.mkdir(save_lc_in_folder)


# Specifiy the limb darkening coefficient
LD_Coeff_array = np.array([[0.5,0.4],[0.5,0.5],[0.5,0.6]])

# Specifiy the size ratio between star and megastructure
star2mega_ratio = 4

print('star2mega_ratio =' ,star2mega_ratio)
print("LD_Coeff_array = ",LD_Coeff_array)
print("len of LD_Coeff_array = ",len(LD_Coeff_array))
print("shape_dir = ",shape_dir)
print("save_lc_in_folder = ",save_lc_in_folder)

user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    os.makedirs(save_lc_in_folder, exist_ok=True)
    for i in np.arange(0,len(LD_Coeff_array),1):
        lc = LcGenerator(shape_dir=shape_dir,
                        LD_Coeff = LD_Coeff_array[i],
                        save_lc_folder_name=save_lc_in_folder,
                        shape_name='circle',
                        star2mega_radius_ratio=star2mega_ratio                     
                        )
        del lc
else:
    print("Exiting the program.")
toc = time.perf_counter()
print(toc-tic, " s")

