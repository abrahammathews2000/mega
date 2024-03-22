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
                 LD_Coeff_dist_start = [0.1,0.05], LD_Coeff_dist_stop = [0.9,0.2],
                 star2mega_radius_ratio_start = 3,star2mega_radius_ratio_stop = 20):
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
            batch_link = self.shape_dir + batch
            print('batch_link = ',batch_link)
            self.y = np.load(batch_link)
            self.y = self.y / np.amax(self.y) # Normalizing images pixel value = [0,1]
            print('self.y = ',self.y)
            self.y_shape = np.array(np.shape(self.y[0]))
            self.radius_mega = self.y_shape[0] / 2

            where_0 = np.where(self.y == 0)  
            where_1 = np.where(self.y == 1)
            self.y[where_0] = 1  # opacity map. 0 = no opacity (transparent); 1 = fully opaque
            self.y[where_1] = 0

            # Make the relative radius as uniform distribution
            # print(len(self.y))
            star2mega_radius_ratio = rng.integers(low = star2mega_radius_ratio_start, high=star2mega_radius_ratio_stop, size=len(self.y))
            np.savetxt("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/star2mega_radius_ratio_details/star2megaRadius_"+str(batch)+".csv", 
                       star2mega_radius_ratio, delimiter=',')
            print("star2mega_radius_ratio details save in: ")
            print("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/star2mega_radius_ratio_details/star2megaRadius_"+str(batch)+".csv")
    
            # Make the LDCs  as uniform distribution
            a = np.random.default_rng().uniform(low = LD_Coeff_dist_start[0],high = LD_Coeff_dist_stop[0],size = len(self.y)) # np.ones(5)*0.3 # 
            b = np.random.default_rng().uniform(low = LD_Coeff_dist_start[1],high = LD_Coeff_dist_stop[1],size = len(self.y)) # np.ones(5)*0.6 #
            LD_Coeff = np.array([a,b]).T
            np.savetxt("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/LDC_details/LDCs_"+str(batch)+".csv", 
                       star2mega_radius_ratio, delimiter=',')
            print("LDC details save in: ")
            print("/scratch/abraham/Documents/mega_git/mega/data/train/info/lc_10_shape_multisize_multiLDC_details/LDC_details/LDCs_"+str(batch)+".csv")
    
            with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
                results = [executor.submit(self.gen_lc,temp_shape=self.y[i],name=str(batch)+'_'+str(i),LD_Coeff=LD_Coeff[i],star2mega_radius_ratio=star2mega_radius_ratio[i]) for i in np.arange(len(self.y))]

                num_workers = executor._max_workers
                print("Number of process pool workers:", num_workers)
                
                for f in concurrent.futures.as_completed(results):
                    print(f.result())

            break  
                
            #---
    def gen_lc(self, temp_shape,name,LD_Coeff,star2mega_radius_ratio):
        radius_star = star2mega_radius_ratio * self.radius_mega
        pad_width_for_mega = int(radius_star - self.radius_mega)
        
        z = np.pad(temp_shape,pad_width=((pad_width_for_mega, pad_width_for_mega), (6, 6)))  # Padwidth = ((top,bottom),(left,right))

        times = np.linspace(-35, 35, 1000)

        annulus = TransitingImage(opacitymat=z, v=0.4, t_ref=0., t_arr=times,LDlaw='quadratic',LDCs=LD_Coeff)

        annulus_LC, overlapTimes = annulus.gen_LC(t_arr=times) #,gpu=True)

        self.save_lc(annulus_LC, overlapTimes, name)

    def save_lc(self,lc,time,name):        
        np.savetxt(str(self.save_lc_folder_name) +'lc_'+str(name) + ".csv", lc, delimiter=',')
        print("lc_"+ str(name)+"saved")

    def __del__(self):
        print('Destructor called, lc deleted.')






# ---- Old ---- #



        
        # plt.close()

        # times = np.linspace(-35., 35., 500)
        #
        # annulus = TransitingImage(opacitymat=z, v=0.4, t_ref=0., t_arr=times)
        # annulus.plot_grid()
        #
        # annulus_LC, overlapTimes = annulus.gen_LC(t_arr=times)
        #
        # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.plot(overlapTimes, annulus_LC, color="#1969ea", ls="-", lw=1)
        # ax.set_ylim(np.amin(annulus_LC) - 0.01, 1.01)
        # plt.xlabel("Time [days]", fontsize=14)
        # plt.ylabel("Relative flux", fontsize=14)
        # # plt.title(r"The annulus' light curve as it transits left-to-right across the star at $v = 0.4 d^{-1}$",fontsize=16)
        # plt.show()


# Old Codes
# #         # Testing on GPU
    # #         # for i in np.arange(len(self.y)):
    # #         #     self.gen_lc(temp_shape=self.y[i],name=i)
    # #         # 

    # #         #---
    # #         # Your large numpy array with 100,000 2D matrices
    # #         # large_array = ...  # Your data here

    # #         # Define a batch size
    # #         # self.batch_size = 10  # You can adjust this based on your system's capabilities

    # #         # Calculate the total number of batches
    # #         # self.total_batches = len(self.y) // self.batch_size + 1
            
    # #         print("len(self.y) = ",len(self.y))
    # #         print("self.batch_size = ",self.batch_size)
    # #         print("total batches = ",self.total_batches)
    # #         print('range(self.total_batches) =',range(self.total_batches))
    # #         # Process the data in batches
    #         for j in range(self.total_batches):
    #             print(" batch = ",j)
    #             start_idx = j * self.batch_size
    #             end_idx = (j + 1) * self.batch_size
    #             if j == range(self.total_batches)[-1]:
    #                 end_idx =  len(self.y) - 1
    #             print(' start_idx = ',start_idx)
    #             print(' end_idx = ',end_idx)
    #             # Extract a batch of data
    #             # self.batch = self.y[start_idx:end_idx]
    #            # Process the batch
            # For CPU only work with original 8bittransit
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     results = [executor.submit(self.gen_lc,self.batch[i],'b'+str(j)+'_'+'s'+str(i)) for i in np.arange(len(self.batch))]

            #     num_workers = executor._max_workers
            #     print("Number of process pool workers:", num_workers)
                
            #     for f in concurrent.futures.as_completed(results):
            #         print(f.result())
