# Commented out time array saving code
# This code is test batching for input dataset
import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
from natsort import natsorted
import os


# from EightBitTransit import TransitingImage
# from EightBitTransit.cTransitingImage import TransitingImage as TIoldansit.misc import *

import concurrent.futures

class LcGenerator:
    def __init__(self, shape_dir, save_lc_folder_name, 
                 LD_Coeff_dist_start=[0, 0], LD_Coeff_dist_stop=[1, 0.55],
                 star2mega_radius_ratio_start=2, star2mega_radius_ratio_stop=30):
        rng = np.random.default_rng()
        self.save_lc_folder_name = save_lc_folder_name
        self.shape_dir = shape_dir  
        self.batch_names = natsorted(os.listdir(self.shape_dir))
        # Change the batch start(inclusive) and end (exlusive) limit in for loop
        # Selecting Files in the given folder

        # self.LD_Coeff = LD_Coeff
        start_index = int(0) # Inclusive
        stop_index = len(self.batch_names) # Exclusive

        print(self.batch_names[start_index:stop_index])

        for batch in self.batch_names[start_index:stop_index]:
            batch_link = self.shape_dir+batch
            print('batch_link =', batch_link)
            self.y = np.load(batch_link)
            self.y = self.y/np.amax(self.y) # Normalizing images pixel value = [0,1]
            print('self.y =', self.y)
            self.y_shape = np.array(np.shape(self.y[0]))
            self.radius_mega = self.y_shape[0]/2

            where_0 = np.where(self.y==0)  
            where_1 = np.where(self.y==1)
            self.y[where_0] = 1  # opacity map. 0 = no opacity (transparent); 1 = fully opaque
            self.y[where_1] = 0

            # Make the relative radius as uniform distribution
            # print(len(self.y))
            star2mega_radius_ratio = rng.integers(low = star2mega_radius_ratio_start, high=star2mega_radius_ratio_stop, size=len(self.y))
            # np.savetxt("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/star2mega_radius_ratio_details/star2megaRadius_"+str(batch)+".csv", 
                    #    star2mega_radius_ratio, delimiter=',')
            # print("star2mega_radius_ratio details save in: ")
            # print("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/star2mega_radius_ratio_details/star2megaRadius_"+str(batch)+".csv")
    
            # Make the LDCs  as uniform distribution
            a = np.random.default_rng().uniform(low=LD_Coeff_dist_start[0], 
                                                high=LD_Coeff_dist_stop[0],
                                                size=len(self.y)) 
            b = np.random.default_rng().uniform(low=LD_Coeff_dist_start[1], 
                                                high=LD_Coeff_dist_stop[1],
                                                size=len(self.y))
            LD_Coeff = np.array([a,b]).T
            # np.savetxt("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/LDC_details/LDCs_"+str(batch)+".csv", 
                    #    LD_Coeff, delimiter=',')
            # print("LDC details save in: ")
            # print("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/LDC_details/LDCs_"+str(batch)+".csv")
    
            with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                results = [executor.submit(self.gen_lc, temp_shape=self.y[i], name=str(batch)+'_'+str(i), LD_Coeff=LD_Coeff[i], star2mega_radius_ratio=star2mega_radius_ratio[i]) for i in np.arange(len(self.y))]

                num_workers = executor._max_workers
                print("Number of process pool workers:", num_workers)
                
                for f in concurrent.futures.as_completed(results):
                    print(f.result())
                
            #---
    def gen_lc(self, temp_shape, name, LD_Coeff, star2mega_radius_ratio):
        radius_star = star2mega_radius_ratio * self.radius_mega
        pad_width_for_mega = int(radius_star - self.radius_mega)
        
        z = np.pad(temp_shape,pad_width=((pad_width_for_mega, pad_width_for_mega), (6, 6)))  # Padwidth = ((top,bottom),(left,right))

        times = np.linspace(-35, 35, 1000)

        annulus = TransitingImage(opacitymat=z, 
                                  v=0.4, 
                                  t_ref=0., 
                                  t_arr=times,
                                  LDlaw='quadratic',
                                  LDCs=LD_Coeff)

        annulus_LC, overlapTimes = annulus.gen_LC(t_arr=times)

        self.save_lc(annulus_LC, name)

    def save_lc(self, lc, name):        
        np.savetxt(str(self.save_lc_folder_name)+'lc_'+str(name) +".csv", lc, delimiter=',')
        print("lc_"+str(name)+"saved")

    def __del__(self):
        print('Destructor called, lc deleted.')
