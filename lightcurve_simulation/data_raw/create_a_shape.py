# Aim: Contain ShapeGen class to create a bezier shape for given parameter and
# to save as png: unfilled for training ML and filled for generating lc using EightBitLC
# to save as coord for generating lc using monte carlo sampling

import numpy as np
import matplotlib.pyplot as plt
from bezier import get_random_points, get_bezier_curve
import cv2

class ShapeGen:
    def __init__(
            self, 
            rad, 
            edgy, 
            noEdges, 
            name, 
            Rmega_star, 
            shape_pixel, 
            img_save_folder_path,
        ):

        self.rad = rad
        self.edgy = edgy
        self.noEdges = noEdges
        self.name = name
        self.Rmega_star = Rmega_star
        self.shape_pixel = shape_pixel
        self.coord_bezier = self.create_bezier_shape()
        self.img_save_folder_path = img_save_folder_path

    def create_bezier_shape(self):
        # Bezier shape generation
        a = get_random_points(n=self.noEdges, scale=1)
        x, y, _ = get_bezier_curve(a, rad=self.rad, edgy=self.edgy)

        x = (x-np.mean(x)) * self.Rmega_star * 2
        y = (y-np.mean(y)) * self.Rmega_star * 2
        z = np.zeros(len(x))

        return np.stack((x, y, z), axis=1)

    def save_png_filled(self):
        plt.clf()
        plt.figure(figsize=(self.shape_pixel, self.shape_pixel))
        plt.tick_params(
            left=False, 
            right=False, 
            labelleft=False, 
            labelbottom=False, 
            bottom=False
        )

        x = self.coord_bezier[:, 0]
        y = self.coord_bezier[:, 1]

        plt.plot(x, y, "black")
        plt.fill(x, y, "black")
        plt.axis('off')  # To remove frame box
        plt.savefig(
            self.img_save_folder_path+'/shape0'+str(self.name)+'.png',
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

        # Convert RGB to grayscale
        image2cnvt = cv2.imread(self.img_save_folder_path+'/shape0'+str(self.name)+'.png')
        gray_cnvtd = cv2.cvtColor(image2cnvt, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(
            self.img_save_folder_path+'/shape0'+str(self.name)+'.png', 
            gray_cnvtd
        )

    def __del__(self):
        print('Destructor called, shape deleted.')
        