import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from batch_pll_lc_curve_generator import LcGenerator
from natsort import natsorted
# from EightBitTransit import TransitingImage

import os
import time

# Where do you want to save the lcs?
# !! Change this everytime you run !!
save_lc_in_folder = '../../data/train/raw/lc/lc_9_shape_5/' # For Train
#

# Which shapes do you need to simulate to get the lcs?
# Give the npy folder path where the batches are saved
shape_dir = '../../data/train/npy/shape/shape_filled5_batches/' # For Train

# Specifiy the limb darkening coefficient
LD_Coeff = [0.4,0.05]

# Specifiy the size ratio between star and megastructure
star2mega_ratio_array = np.array([4])

print('star2mega_ratio_array =' ,star2mega_ratio_array)
print("LD_Coeff = ",LD_Coeff)
print("shape_dir = ",shape_dir)
print("save_lc_in_folder = ",save_lc_in_folder)

user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    print("Running the program...")
    os.makedirs(save_lc_in_folder, exist_ok=True)
    for i in np.arange(0,len(star2mega_ratio_array),1):
        lc = LcGenerator(shape_dir=shape_dir,
                        LD_Coeff= LD_Coeff,
                        save_lc_folder_name=save_lc_in_folder,
                        star2mega_radius_ratio=star2mega_ratio_array[i]
                        )
        del lc
else:
    print("Exiting the program.")

toc = time.perf_counter()
print(toc-tic, " s")

