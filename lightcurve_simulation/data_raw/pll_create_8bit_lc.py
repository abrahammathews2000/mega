import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from pll_lc_curve_generator import LcGenerator
# from EightBitTransit import TransitingImage

import os
import time

# !! Change this everytime you run !!
save_lc_in_folder = '../../data/test/raw/lc/lc_2_shape_1/' # For Test
#

if not os.path.exists(save_lc_in_folder):
    os.mkdir(save_lc_in_folder)



tic = time.perf_counter()

# Give the npy file path for the shape
shape_dir = '../../data/test/npy/shape/shape_1.npy'

# Specifiy the size ratio between star and megastructure
star2mega_ratio_array = np.array([4])
print('star2mega_ratio_array =' ,star2mega_ratio_array)

for i in np.arange(0,len(star2mega_ratio_array),1):
    lc = LcGenerator(shape_dir=shape_dir,
                     LD_Coeff = [0.1,0.05],
                     save_lc_folder_name=save_lc_in_folder,
                     star2mega_radius_ratio=star2mega_ratio_array[i]                     
                    )
    del lc


toc = time.perf_counter()
print(toc-tic, " s")

