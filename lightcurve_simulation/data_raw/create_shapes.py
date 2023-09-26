# Create shapes using the class ShapeGen given in create_a_shape.py
from create_a_shape import ShapeGen
import os
import numpy as np
import time

tic = time.perf_counter()

#Change everytime you run#
index_shapefolder = 8

if not os.path.exists('/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/shape/'):
    os.mkdir('/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/shape/')
# if not os.path.exists('./generatedData/shape_coord/'):
#     os.mkdir('./generatedData/shape_coord/')
if not os.path.exists('/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/shape/'+str(index_shapefolder)+'/'):
    os.mkdir('/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/shape/'+str(index_shapefolder)+'/')
# if not os.path.exists('./generatedData/shape_unfilled/'):
#     os.mkdir('./generatedData/shape_unfilled/')

# rad_array_random = np.linspace(0, 0.9, 5) #np.unique(np.random.uniform(0,1,30))
rad_array =  [0.12, 0.23, 0.45, 0.73, 0.91] # np.linspace(0.1, 0.9, 5) # np.array([1])    # rad_array =
                                # Do to limited RAM size only one element is give
                                # rad_array for every run
                                
# edgy_array_random = np.linspace(0, 9, 10) # np.unique(np.random.randint(0,20,30))   
edgy_array =  [1.2, 3.3, 5.4, 7.8, 9.2] # np.linspace(1, 9, 5)   # np.linspace(0, 9, 10)
print('rad_array =', rad_array)
print('edgy_array = ',edgy_array)
edges_array = [3,4,5,6,7,8,9,10,11,12]
variety = [0]
name = 0

for var_el in variety:
    for rad_el in rad_array:
        for edgy_el in edgy_array:
            for edges_el in edges_array:
                t = ShapeGen(rad=rad_el, edgy=edgy_el,noEdges=edges_el,name=name,Rmega_star=0.25,shape_pixel=0.50,index_shapefolder=index_shapefolder)
                # t.save_coord()
                t.save_png_filled()
                # t.save_png_unfilled()
                del t
                print("Shape no. ",name)
                name = name + 1
print("Shape generation done")

toc = time.perf_counter()
print("Took (s): ",toc-tic)

