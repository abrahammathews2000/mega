# Create shapes using the class ShapeGen given in create_a_shape.py
from create_a_shape import ShapeGen
import os
import numpy as np
import time

tic = time.perf_counter()

img_save_folder_path = '/scratch/abraham/Documents/mega_git/mega/data/train/raw/shape/20_April_2024/'

if not os.path.exists(img_save_folder_path):
    os.mkdir(img_save_folder_path)

rad_array =  [0.1, 0.2, 0.3, 0.4, 0.5]                                
edgy_array =  [1.0, 2.0, 3.0, 4.0, 5.0] 
edges_array = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

print('rad_array =', rad_array)
print('edgy_array =',edgy_array)
print('edges_array =',edges_array)
variety = [0]

name = 0
for rad_el in rad_array:
    for edgy_el in edgy_array:
        for edges_el in edges_array:
            t = ShapeGen(
                rad=rad_el,
                edgy=edgy_el,
                noEdges=edges_el,
                name=name,
                Rmega_star=0.25,
                shape_pixel=0.50,
                img_save_folder_path=img_save_folder_path
            )

            t.save_png_filled()
            del t

            print("Shape no: ",name)
            name = name + 1

print("Shape generation done")

toc = time.perf_counter()
print("Took (s): ",toc-tic)

