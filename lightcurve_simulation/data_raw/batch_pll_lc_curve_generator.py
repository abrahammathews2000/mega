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
    def __init__(self, ydir, star2mega_radius_ratio=4,folder_name = 1):
        self.folder_name = folder_name
        self.y_dir = ydir  # './generatedData/shape_dict.npy'
        self.batch_names = natsorted(os.listdir(self.y_dir))
        # Change the batch start(inclusive) and end (exlusive) limit in for loop
        # Selecting Files in the given folder
        start_index = 131 # Inclusive
        stop_index = 133 # Exclusive
        print(self.batch_names[start_index:stop_index])
        for batch in self.batch_names[start_index:stop_index]:
            batch_link = self.y_dir + batch
            print('batch_link = ',batch_link)
            self.y = np.load(batch_link)
            self.y = self.y / np.amax(self.y)
            print('self.y = ',self.y)
            self.y_shape = np.array(np.shape(self.y[0]))
            self.radius_mega = self.y_shape[0] / 2
            self.star2mega_radius_ratio = star2mega_radius_ratio
            self.radius_star = self.star2mega_radius_ratio * self.radius_mega
            self.pad_width_for_mega = int(self.radius_star - self.radius_mega)
            # self.folder_name = folder_name
            #print('pad_width_for_mega = ', self.pad_width_for_mega)
            where_0 = np.where(self.y == 0)  
            where_1 = np.where(self.y == 1)
            self.y[where_0] = 1  # opacity map. 0 = no opacity (transparent); 1 = fully opaque
            self.y[where_1] = 0
    
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(self.gen_lc,self.y[i],str(batch)+'_'+str(i)) for i in np.arange(len(self.y))]

                num_workers = executor._max_workers
                print("Number of process pool workers:", num_workers)
                
                for f in concurrent.futures.as_completed(results):
                    print(f.result())           


                
            #---
    def gen_lc(self, temp_shape,name):
        # Both of these two padding will give the same result 1st one saves little more memory space
        z = np.pad(temp_shape,pad_width=((self.pad_width_for_mega, self.pad_width_for_mega), (6, 6)))  # Padwidth = ((top,bottom),(left,right))
        # z = np.pad(y[0], pad_width=((self.pad_width_for_mega, self.pad_width_for_mega), (pad_width_for_mega, pad_width_for_mega))) # Padwidth = ((top,bottom),(left,right))
        # print("np.shape(z) ", np.shape(z))
        # print(y[charge_index])

        # plt.imshow(z, cmap='gray')
        # plt.show()
        times = np.linspace(-35, 35, 1000)

        # annulus = TransitingImage(opacitymat=z,v=0.4,t_ref=0.,t_arr=times) # For no limb darkening
        annulus = TransitingImage(opacitymat=z, v=0.4, t_ref=0., t_arr=times,LDlaw='quadratic',LDCs=[0.5,0.05])
        # annulus.plot_grid()
        annulus_LC, overlapTimes = annulus.gen_LC(t_arr=times) #,gpu=True)

        self.save_lc(annulus_LC, overlapTimes, name)
        # return(annulus_LC,overlapTimes)

        # fig, ax = plt.subplots(1,1,figsize=(8,6))
        # # ax.plot(overlapTimes,annulus_LC,color="#1969ea",ls="-",lw=1)
        # ax.scatter(overlapTimes,annulus_LC,color="black")
        # ax.set_ylim(np.amin(annulus_LC)-0.01,1.01)
        # plt.xlabel("Time [days]",fontsize=14)
        # plt.ylabel("Relative flux",fontsize=14)
        # plt.title("Light curve",fontsize=16)
        # plt.show()
    def save_lc(self,lc,time,name):
        
        np.savetxt("/home/abraham/Documents/ms_proj_shape_lc_gen/data_raw/lc/"+str(self.folder_name) +"/lcflux0" + str(self.star2mega_radius_ratio) +'_'+str(name) + ".csv", lc, delimiter=',')
        # np.savetxt("./generatedData/lc/time/lctime0" + str(name) + ".csv", time, delimiter=',')
        print("lc"+str(self.star2mega_radius_ratio) +'_'+ str(name)+"saved")

    def __del__(self):
        print('Destructor called, lc deleted.')










        
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
