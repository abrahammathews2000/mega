# This is to create ellipse shapes
import numpy as np
import matplotlib.pyplot as plt
# import cv2
# import os
from math import pi, cos, sin

u=0.       #x-position of the center
v=0.     #y-position of the center
a=2.       #radius on the x-axis
b=1.5      #radius on the y-axis


t = np.linspace(0, 2*pi, 100)
Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
     #u,v removed to keep the same center location




plt.plot( u+Ell[0,:] , v+Ell[1,:] , color ='black' )     #initial ellipse
plt.fill(u+Ell[0,:], v+Ell[1,:],"black")
plt.axis('off')  # To remove frame box
plt.show()