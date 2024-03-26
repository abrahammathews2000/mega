# Import TF and check for GPU

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)

# Import required libraries

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import save_model, load_model
import math
from sklearn.model_selection import train_test_split
import sys
from numpy import array,append,arange,zeros,exp,sin,random,std
from scipy.interpolate import interp1d

# 1. Load Dataset
## Load Train Set
train_shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/train/npy/shape/shape_5.npy'
train_lc_dir = '/scratch/abraham/Documents/mega_git/mega/data/train/npy/lc/lc_10_shape_multisize_multiLDC.npy'
train_lc = np.load(train_lc_dir)
train_shape = np.load(train_shape_dir)
# Check equality of number of dataset
print('train_lc.shape =',train_lc.shape)
print('train_shape.shape = ',train_shape.shape)
if len(train_lc)==len(train_shape):
    print("Train Set: No. of LC = No. of shapes")
else:
    sys.exit("EXIT: Train Set: No. of LC != No. of shapes")

## Load Validation Set
vald_shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/vald/npy/shape/shape_1.npy'
vald_lc_dir = '/scratch/abraham/Documents/mega_git/mega/data/vald/npy/lc/lc_10_shape_multisize_multiLDC.npy'
vald_lc = np.load(vald_lc_dir)
vald_shape = np.load(vald_shape_dir)
# Check equality of nuftmber of dataset
print('vald_lc.shape =',vald_lc.shape)
print('vald_shape.shape = ',vald_shape.shape)
if len(vald_lc)==len(vald_shape):
    print("Vald Set: No. of LC = No. of shapes")
else:
    sys.exit("Vald Set: No. of LC = No. of shapes")