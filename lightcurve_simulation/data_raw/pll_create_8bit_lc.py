import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from pll_lc_curve_generator import LcGenerator
# from EightBitTransit import TransitingImage

import os
import time

# Change this everytime you run
index_shapefolder = 8
#
tic = time.perf_counter()

if not os.path.exists("/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/lc/"+str(index_shapefolder) +"/"):
    os.mkdir("/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/lc/"+str(index_shapefolder) +"/")
# if not os.path.exists('./generatedData/lc/'):
#     os.mkdir('./generatedData/lc/')
# if not os.path.exists('./generatedData/lc/flux/'):
#     os.mkdir('./generatedData/lc/flux/')
# if not os.path.exists('./generatedData/lc/time/'):
#     os.mkdir('./generatedData/lc/time/')

# y_dir = './generatedData/shape_dict.npy'
y_dir = '/home/abraham/Documents/ms_proj_shape_lc_gen/data_npy/shape_npy/shape_filled' + str(index_shapefolder) + '.npy'# shape_filled.npy

# star2mega_ratio_array = np.array([2]) # np.array([1,2,3,4,5,6,7,8,9,10])
star2mega_ratio_array = np.linspace(4,5,1)
print('star2mega_ratio_array =' ,star2mega_ratio_array)

for i in np.arange(0,len(star2mega_ratio_array),1):
    lc = LcGenerator(ydir=y_dir,star2mega_radius_ratio=star2mega_ratio_array[i],folder_name=index_shapefolder)
    del lc


toc = time.perf_counter()
print(toc-tic, " s")

# y = np.load(y_dir)
# y = y/np.amax(y)
#
# y_shape = np.array(np.shape(y[0]))
# radius_mega = y_shape[0]/2
# radius_star = 4 * radius_mega
# pad_width_for_mega = int(radius_star - radius_mega)
# print('pad_width_for_mega = ',pad_width_for_mega)
#
# where_0 = np.where(y == 0)
# where_1 = np.where(y == 1)
#
# y[where_0] = 1
# y[where_1] = 0
#
# charge_index = 0
#
# # Both of these two padding will give the same result 1st one saves little more memory space
# z = np.pad(y[0], pad_width=((pad_width_for_mega, pad_width_for_mega), (0, 0))) # Padwidth = ((top,bottom),(left,right))
# # z = np.pad(y[0], pad_width=((pad_width_for_mega, pad_width_for_mega), (pad_width_for_mega, pad_width_for_mega))) # Padwidth = ((top,bottom),(left,right))
#
#
# print(z)
# # print(y[charge_index])
#
# plt.imshow(z, cmap='gray')
# plt.show()
# # plt.close()
#
# times = np.linspace(-35.,35.,500)
#
# annulus = TransitingImage(opacitymat=z,v=0.4,t_ref=0.,t_arr=times)
# annulus.plot_grid()
#
# annulus_LC, overlapTimes = annulus.gen_LC(t_arr=times)
#
# fig, ax = plt.subplots(1,1,figsize=(8,6))
# ax.plot(overlapTimes,annulus_LC,color="#1969ea",ls="-",lw=1)
# ax.set_ylim(np.amin(annulus_LC)-0.01,1.01)
# plt.xlabel("Time [days]",fontsize=14)
# plt.ylabel("Relative flux",fontsize=14)
# #plt.title(r"The annulus' light curve as it transits left-to-right across the star at $v = 0.4 d^{-1}$",fontsize=16)
# plt.show()
