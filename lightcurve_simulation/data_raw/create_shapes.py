# Create shapes using the class ShapeGen given in create_a_shape.py
from create_a_shape import ShapeGen
import os
import numpy as np
import time


img_save_folder_path = '/scratch/abraham/Documents/mega_git/mega/data/train/raw/shape/20_April_2024/'

if not os.path.exists(img_save_folder_path):
    os.mkdir(img_save_folder_path)

rng = np.random.default_rng()
rad_array =  np.random.default_rng().uniform(low=0, high=1, size=15)                                
edgy_array =  np.arange(0, 15, 1)
edges_array = np.arange(3, 20, 1)

print('rad_array =', rad_array)
print('edgy_array =', edgy_array)
print('edges_array =', edges_array)
print("No of shapes to generate =",len(rad_array)*len(edgy_array)*len(edgy_array))

user_input = input("Do you want to run the code? (y/n): ")
tic = time.perf_counter()
if user_input.lower() == "y":
    name = 250
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
else:
    print("Exiting the program.")

toc = time.perf_counter()
print("Took (s): ",toc-tic)
