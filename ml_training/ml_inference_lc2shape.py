# .py code for mega/ml_training/ml_inference_lc2shape.ipynb
# This code is used to infer transit shape from input light curve.
# It involves following steps:
# 1. Load ML model
# 2. Load input data - i.e. the light curves
# 3. Predict the shapes from the light curves

# 4. Load True Output (if any)


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
import sys
from numpy import array,append,arange,zeros,exp,sin,random,std
from scipy.interpolate import interp1d

model_load = load_model("/scratch/abraham/Documents/mega_git/mega/ml_model/nov13_model1/nov13_model1_sample_interpolate_no_noise_unf_samplept_20to60.h5")

# 1. Load Lightcurve Dataset
lc_dir = '/scratch/abraham/Documents/mega_git/mega/data/random/npy/lc/lc_1_21Feb2024_circles_38_38_px.npy'
test_lc = np.load(lc_dir)

# Normalize the lightcurves
## - Test Set
test_lc_scaled = np.zeros(test_lc.shape)
for i in np.arange(len(test_lc_scaled)):
    test_lc_scaled[i] = (test_lc[i] - np.amin(test_lc[i]))/(np.amax(test_lc[i]) - np.amin(test_lc[i]))


# Append with ones (if required only. 
# eg APPENING not required for BATMAN lightcurve since we already append BATMAN 
# while saving it as csv file and npy file)
## - Test Set
test_lc_scaled_append = np.ones((test_lc.shape[0],120))
for i in np.arange(len(test_lc_scaled)):
    test_lc_scaled_append[i,10:110] = test_lc_scaled[i]
print("Extended the light curves")   

lc_to_predict = test_lc_scaled_append
# Test on the ML model - Test Dataset - Original Simulated light curve 
# (i.e. without sampling and interpolation)
# Test dataset - Prediciton
test_predict_shape = model_load.predict(lc_to_predict)
test_predict_shape_normalized = np.zeros(test_predict_shape.shape)
for i in np.arange(len(test_predict_shape)):
    test_predict_shape_normalized[i] = (test_predict_shape[i] - np.amin(test_predict_shape[i]))/(np.amax(test_predict_shape[i]) - np.amin(test_predict_shape[i]))
print("Normalized the predicted shape")
test_predict_shape = test_predict_shape_normalized
del test_predict_shape_normalized


fig, axes = plt.subplots(1, 2, figsize=(8,2),gridspec_kw={'width_ratios': [2,1]})  # 1 row, 2 columns
ph = np.linspace(-1,1,len(lc_to_predict[0]))

# # Plot on each subplot
# k = int(5) # Index
# axes[0].set_title('Light Curve - pixelate circle')
# axes[0].plot(ph, lc_to_predict[k], color='tab:red')
# axes[0].grid('on')
# axes[0].set_ylabel('Flux')
# axes[0].set_xlabel('Phase (Arbitrary Unit)')


# axes[1].set_title('Predicted Shape')
# axes[1].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
# img = axes[1].imshow(test_predict_shape[k],cmap='inferno')
# plt.colorbar(img)

# plt.savefig("test_predicted_shape.png", dpi=150)


shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/random/npy/shape/21Feb2024_circles_38_38_px.npy'
test_shape = np.load(shape_dir)

# Normalize the image, convert to opacity map
## Test Set
test_shape = test_shape/np.amax(test_shape)
test_shape_where_0 = np.where(test_shape == 0)
test_shape_where_1 = np.where(test_shape == 1)
test_shape[test_shape_where_0] = 1  # 1 represent the shape (1 opacity)
test_shape[test_shape_where_1] = 0  # 0 represent background (0 opacity)




# Plot on each subplot
for i in np.arange(0,8,1):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2),gridspec_kw={'width_ratios': [2,1,1]})  # 1 row, 3 columns
    ph = np.linspace(-1,1,len(lc_to_predict[0]))

    k = int(i) # Index
    axes[0].set_title('Light curve of a pixelated circle')
    axes[0].plot(ph, lc_to_predict[k], color='blue')
    axes[0].grid('on')
    axes[0].set_ylabel('Flux')
    axes[0].set_xlabel('Phase (Arbitrary Unit)')

    axes[1].set_title('Predicted Shape')
    axes[1].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    img = axes[1].imshow(test_predict_shape[k],cmap='inferno')
    plt.colorbar(img)

    axes[2].set_title('Actual Shape')
    axes[2].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    img = axes[2].imshow(test_shape[k],cmap='inferno')
    plt.colorbar(img)
    plt.savefig(f"lc_predicted_shape_k{k}.png", dpi=150)
    plt.close(fig)
