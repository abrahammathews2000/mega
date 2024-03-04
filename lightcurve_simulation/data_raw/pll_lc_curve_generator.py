# Commented out time array saving code

import numpy as np
import matplotlib.pyplot as plt
from EightBitTransit.cTransitingImage import TransitingImage
from EightBitTransit.inversion import *
from EightBitTransit.misc import *
# from EightBitTransit import TransitingImage
# from EightBitTransit.cTransitingImage import TransitingImage as TIoldansit.misc import *

import concurrent.futures

class LcGenerator:
    def __init__(self, shape_dir, LD_Coeff,save_lc_folder_name,star2mega_radius_ratio=4,save_time=False):
        self.shape_dir = shape_dir  # './generatedData/shape_dict.npy'
        self.y = np.load(self.shape_dir)
        self.y = self.y / np.amax(self.y)  # Normalizing images pixel value = [0,1]
        self.y_shape = np.array(np.shape(self.y[0]))
        self.radius_mega = self.y_shape[0] / 2
        self.star2mega_radius_ratio = star2mega_radius_ratio
        self.radius_star = self.star2mega_radius_ratio * self.radius_mega
        self.pad_width_for_mega = int(self.radius_star - self.radius_mega)
        self.folder_name = save_lc_folder_name
        self.LD_Coeff = LD_Coeff
        self.save_time = save_time
        print("LD_Coeff = ",self.LD_Coeff)
        #print('pad_width_for_mega = ', self.pad_width_for_mega)
        
        ##
        # To exchange the background and shape pixel value to opacity value
        where_0 = np.where(self.y == 0)
        where_1 = np.where(self.y == 1)
        self.y[where_0] = 1
        self.y[where_1] = 0
        ##

        # Testing on GPU
        # for i in np.arange(len(self.y)):
        #     self.gen_lc(temp_shape=self.y[i],name=i)
        # 


        # For CPU only work with original 8bittransit
        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            results = [executor.submit(self.gen_lc,self.y[i],i) for i in np.arange(len(self.y))]

            num_workers = executor._max_workers
            print("Number of process pool workers:", num_workers)
            
            for f in concurrent.futures.as_completed(results):
                print(f.result())
        # 


    def gen_lc(self, temp_shape,name):
        # Both of these two padding will give the same result 1st one saves little more memory space
        ## 1 aPixel value for padding 0, complete background =0
        z = np.pad(temp_shape,pad_width=((self.pad_width_for_mega, self.pad_width_for_mega), (6, 6)),mode = 'constant', constant_values=0.)  # Padwidth = ((top,bottom),(left,right))
        ##

        times = np.linspace(-35, 35, 1000)

        # annulus = TransitingImage(opacitymat=z,v=0.4,t_ref=0.,t_arr=times) # For no limb darkening
        annulus = TransitingImage(opacitymat=z, v=0.4, t_ref=0., t_arr=times,LDlaw='quadratic',LDCs=self.LD_Coeff)
        print('z = ',z)
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
        np.savetxt(str(self.folder_name) +'lc_'+ str(self.star2mega_radius_ratio)+'_'+str(name) + ".csv", lc, delimiter=',')
        print("lc"+str(self.star2mega_radius_ratio) +'_'+ str(name)+"saved")

        if self.save_time == True:
            np.savetxt(str(self.folder_name) +'time_lc_'+ str(self.star2mega_radius_ratio)+'_'+str(name) + ".csv", time, delimiter=',')
            print('time_lc_'+str(self.star2mega_radius_ratio) +'_'+ str(name)+"saved")

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
