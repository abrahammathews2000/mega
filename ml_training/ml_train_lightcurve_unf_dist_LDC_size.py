# .py code help to run the code in screen mode .ipynb cannot do that
# This code can be used to load light curve training dataset and do the
# preprocessing like vertical scaling and extending the light curve
# CNN Model is defined
# Training, learning rate scheduler and early stopping feature included

## -- IMP: Check whether the file name to save the model is complete ##

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

import time

## Edit the base folder to save model, weights and plots
base_model_folder = '/scratch/abraham/Documents/mega_git/mega/ml_model/april24_2024_model/'


# 1. Load Dataset
## Load Train Set
train_shape_dir = '/scratch/abraham/Documents/mega_git/mega/data/train/npy/shape/concatenated_shape/train_shape_5_2times.npy'
train_lc_dir = '/scratch/abraham/Documents/mega_git/mega/data/train/npy/lc/concatenated_lc/concat_lc_10_12_shape_5_multisize_multiLDC.npy'

print(f"train_shape_dir = {train_shape_dir}")
print(f"train_lc_dir = {train_lc_dir}")

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

print(f"vald_shape_dir = {vald_shape_dir}")
print(f"vald_lc_dir = {vald_lc_dir}")

vald_lc = np.load(vald_lc_dir)
vald_shape = np.load(vald_shape_dir)
# Check equality of nuftmber of dataset
print('vald_lc.shape =',vald_lc.shape)
print('vald_shape.shape = ',vald_shape.shape)
if len(vald_lc)==len(vald_shape):
    print("Vald Set: No. of LC = No. of shapes")
else:
    sys.exit("Vald Set: No. of LC = No. of shapes")

# 2. Normalize the image, convert to opacity map
## Train Set
train_shape = train_shape/np.amax(train_shape)
train_shape_where_0 = np.where(train_shape == 0)
train_shape_where_1 = np.where(train_shape == 1)
train_shape[train_shape_where_0] = 1  # 1 represent the shape (1 opacity)
train_shape[train_shape_where_1] = 0  # 0 represent background (0 opacity)

## Valdn Set
vald_shape = vald_shape/np.amax(vald_shape)
vald_shape_where_0 = np.where(vald_shape == 0)
vald_shape_where_1 = np.where(vald_shape == 1)
vald_shape[vald_shape_where_0] = 1  # 1 represent the shape (1 opacity)
vald_shape[vald_shape_where_1] = 0  # 0 represent background (0 opacity)
print("Normalized the shape")

# 3. Vertically Scaling
## - Train Set
train_lc_scaled = np.zeros(train_lc.shape)
for i in np.arange(len(train_lc_scaled)):
    train_lc_scaled[i] = (train_lc[i] - np.amin(train_lc[i]))/(np.amax(train_lc[i]) - np.amin(train_lc[i]))

## - Vald Set
vald_lc_scaled = np.zeros(vald_lc.shape)
for i in np.arange(len(vald_lc_scaled)):
    vald_lc_scaled[i] = (vald_lc[i] - np.amin(vald_lc[i]))/(np.amax(vald_lc[i]) - np.amin(vald_lc[i]))
print("Vertically Scaled the light curves")


# Add flat line towards left and right of dip
# 10 data points on each side
# 4. Extend the lightcurves
## - Train Set
train_lc_scaled_append = np.ones((train_lc.shape[0],150))
print('train_lc_scaled_append.shape = ',train_lc_scaled_append.shape)
print("len(train_lc_scaled_append[0,25:125]) = ",len(train_lc_scaled_append[0,25:125]))

for i in np.arange(len(train_lc_scaled)):
    train_lc_scaled_append[i,25:125] = train_lc_scaled[i]

## - Vald Set
vald_lc_scaled_append = np.ones((vald_lc.shape[0],150))
for i in np.arange(len(vald_lc_scaled)):
    vald_lc_scaled_append[i,25:125] = vald_lc_scaled[i]
print("Extended the light curves")

del vald_lc
del train_lc
del vald_lc_scaled
del train_lc_scaled

# 5. Horizontal scaling
def scale_horizontally(input_lc_dataset):
    # lc_np_array_offset_mask used to select the flat part by certain percentage
    input_lc_dataset_mask = np.copy(input_lc_dataset)

    for iteration in np.arange(len(input_lc_dataset)):
        # 0.988 is working good | lower it and see changes # 0.96 - 0.97 -better # 0.95 -worse
        input_lc_dataset_mask[iteration][(input_lc_dataset[iteration]>=0.98)] = 1.0
        input_lc_dataset_mask[iteration][(input_lc_dataset[iteration]<0.98)] = 0.0

    print("Length of one LC = ", len(input_lc_dataset_mask[0]))

    count_zeros_array = np.zeros((len(input_lc_dataset_mask),))
    for iteration in np.arange(len(input_lc_dataset_mask)):
        # Calculate the number of occurrences of '0'
        count_zeros = np.count_nonzero(input_lc_dataset_mask[iteration] == 0)
        count_zeros_array[iteration] = count_zeros
    
    # Interpolate the light curve
    input_lc_dataset_interpol = np.zeros((len(input_lc_dataset), 120))
    len_selected_portion = np.zeros(len(input_lc_dataset))
    print("input_lc_dataset_interpol.shape =", input_lc_dataset_interpol.shape)

    center_index = int(len(input_lc_dataset[0])/2)
    print("center_index =", center_index)

    for iteration in np.arange(len(input_lc_dataset_interpol)):

        left_index = int(center_index - int(count_zeros_array[iteration]/2) - int(count_zeros_array[iteration]/6))
        right_index = int(center_index + int(count_zeros_array[iteration]/2) + int(count_zeros_array[iteration]/6))
        selected_portion = input_lc_dataset[iteration][left_index:right_index]
        # print("left_index =", left_index)
        # print("right_index =", right_index)

        # Calculate the length of the selected region
        len_selected_portion[iteration] = len(selected_portion)

        # Interpolate the selected portion
        # Original data
        original_x = np.linspace(-1, 1, num=len(selected_portion))
        original_y = selected_portion

        # Create a quadratic interpolation function
        f = interp1d(original_x, original_y, kind='quadratic')

        # Define the range of x-values for the interpolation with 120 elements
        x_interpolation = np.linspace(-1, 1, num=120)

        # Perform the interpolation
        y_interpolated = f(x_interpolation)
        input_lc_dataset_interpol[iteration] = y_interpolated

    return input_lc_dataset_interpol

train_lc_horiz_scaled = scale_horizontally(train_lc_scaled_append)
vald_lc_horiz_scaled = scale_horizontally(vald_lc_scaled_append)

del train_lc_scaled_append
del vald_lc_scaled_append



## START - Add noise
random_generator = np.random.default_rng()

SNR_array_train = random_generator.uniform(50, 500, len(train_lc_horiz_scaled))
std_dev_train = 1/SNR_array_train
del SNR_array_train

SNR_array_vald = random_generator.uniform(50, 500, len(vald_lc_horiz_scaled))
std_dev_vald = 1/SNR_array_vald
del SNR_array_vald

train_lc_horiz_scaled_noise = np.zeros(train_lc_horiz_scaled.shape)
vald_lc_horiz_scaled_noise = np.zeros(vald_lc_horiz_scaled.shape)

for i in np.arange(len(train_lc_horiz_scaled)):
    train_lc_horiz_scaled_noise[i] = train_lc_horiz_scaled[i] + np.random.normal(loc=0.0, scale=std_dev_train[i], size=len(train_lc_horiz_scaled[i]))

for i in np.arange(len(vald_lc_horiz_scaled)):
    vald_lc_horiz_scaled_noise[i] = vald_lc_horiz_scaled[i] + np.random.normal(loc=0.0, scale=std_dev_vald[i], size=len(vald_lc_horiz_scaled[i]))
## END - Add noise

processed_train_lc = train_lc_horiz_scaled_noise
processed_vald_lc = vald_lc_horiz_scaled_noise

print(f"processed_train_lc = {processed_train_lc}")
print(f"processed_vald_lc = {processed_vald_lc}")

# Verification
# Plot - Train LCs
plt.clf()
num = 3
fig,ax=plt.subplots(num,
                    2, 
                    figsize=(6,5), 
                    gridspec_kw={'width_ratios': [2,1], 'wspace': 0.2, 'hspace': 0.4}
)

ax[0][1].set_title('Shape',size=15)
ax[0][0].set_title('Light Curve (Train Dataset)',size=15)
ax[num-1][0].set_xlabel('Phase',size=13)
ph = np.linspace(-1,1,len(processed_train_lc[0]))

i = 0
for i in np.arange(0, num):
    k = np.random.randint(0, len(processed_train_lc)-1)
    ax[i][1].tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    if(i<num-1): ax[i][0].tick_params(labelbottom = False, bottom = False)
    img = ax[i][1].imshow(train_shape[k], cmap='inferno')
    plt.colorbar(img)
    ax[i][0].set_ylabel('Flux', size=13)
    # ax[i][0].set_ylim(-0.5, 1.5)
    ax[i][0].plot(ph, processed_train_lc[k]-train_lc_horiz_scaled[k], color = 'tab:red', linewidth='2')
    ax[i][0].grid('on')
    i = i + 1
plt.savefig(base_model_folder+'plot_train_lc.png')
plt.close()

# Plot - Vald LCs
plt.clf()
num = 3
fig,ax=plt.subplots(num,2, figsize=(4,3), gridspec_kw={ 'width_ratios': [2,1],
        'wspace': 0.2,'hspace': 0.4})

ax[0][1].set_title('Shape',size=15)
ax[0][0].set_title('Light Curve (Vald Dataset)',size=15)
ax[num-1][0].set_xlabel('Phase',size=13)
ph = np.linspace(-1,1,len(processed_vald_lc[0]))

i = 0
for i in np.arange(0,num):
    k = np.random.randint(0, len(processed_vald_lc)-1)
    ax[i][1].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    if(i<num-1): ax[i][0].tick_params(labelbottom = False, bottom = False)
    img = ax[i][1].imshow(vald_shape[k],cmap='inferno')
    plt.colorbar(img)
    ax[i][0].set_ylabel('Flux',size=13)
    # ax[i][0].set_ylim(-0.5,1.5)
    # ax[i][0].scatter(ph, vald_lc_scaled_append[k],color = 'black',marker='.')
    ax[i][0].plot(ph, processed_vald_lc[k]-vald_lc_horiz_scaled[k], color = 'tab:red', linewidth='2')
    ax[i][0].grid('on')
    i = i + 1
plt.savefig(base_model_folder+'plot_vald_lc.png')
plt.close()

# Delete lightcurves no longer required
del train_lc_horiz_scaled
del vald_lc_horiz_scaled

del train_lc_horiz_scaled_noise
del vald_lc_horiz_scaled_noise

# Model path to be saved to
model_save_path = base_model_folder+"april24_2024_model_unfDist_LDC_size_horz_scale.h5"
print(f"model_save_path = {model_save_path}")

no_epochs = int(3) # For testing start with small value of 3 or 5
print("no_epochs =",no_epochs)

user_input = input("Do you want to run the code? (y/n): ")
if user_input.lower() != "y":
    sys.exit("EXIT: User declined to continue the program")

tic = time.perf_counter()
print("Creating ML Pipeline")
# 6. ML Pipeline
## Train Set
train_dataset = tf.data.Dataset.from_tensor_slices((processed_train_lc, train_shape))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.batch(100)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
print(train_dataset)

## Vald Set
vald_dataset = tf.data.Dataset.from_tensor_slices((processed_vald_lc, vald_shape))
vald_dataset = vald_dataset.batch(100)
vald_dataset = vald_dataset.cache()
vald_dataset = vald_dataset.prefetch(tf.data.AUTOTUNE)
print(vald_dataset)

# CNN Model
input_shape = np.array(np.shape(processed_train_lc[0]))
print("np.shape(input_shape) =", input_shape[0])

output_shape = np.array(np.shape(train_shape[0]))
print("np.shape(input_shape) =", output_shape[0], output_shape[1])

START = input_shape[0]
END = output_shape[0]
print("Start =", START)
print("End =", END)

conv_ip = keras.layers.Input(shape=(START,), name='Input')
x= keras.layers.Reshape((START, 1), input_shape=(START,), name='reshape_1')(conv_ip)
x= keras.layers.BatchNormalization()(x)

x=keras.layers.Conv1D(16,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='conv16_5', 
                      padding='same')(x)

x=keras.layers.Conv1D(16,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='second_conv16_5', 
                      padding='same')(x)

x=keras.layers.MaxPool1D(5,
                         strides=2,
                         data_format='channels_last',
                         name='maxpool_1', 
                         padding='same')(x)

x=keras.layers.Conv1D(32,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='first_conv32_5', 
                      padding='same')(x)

x=keras.layers.Conv1D(32,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='second_conv32_5', 
                      padding='same')(x)

x=keras.layers.MaxPool1D(5,
                         strides=2,
                         data_format='channels_last',
                         name='maxpool_2', 
                         padding='same')(x) #200

x=keras.layers.Conv1D(64,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='first_conv64_5', 
                      padding='same')(x)

x=keras.layers.Conv1D(64,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='second_conv64_5', 
                      padding='same')(x)

x=keras.layers.MaxPool1D(5,
                         strides=2,
                         data_format='channels_last',
                         name='maxpool_3', 
                         padding='same')(x) #100

x=keras.layers.Flatten(name='flat_1')(x)

x2=keras.layers.Dense(256, name='dense_layer_5', activation='relu')(x)
x2=keras.layers.Dense(256, name='dense_layer_6', activation='relu')(x2)

x2= keras.layers.Dense(END**2, name='dense_layer_u', activation='relu')(x2)
x2 = keras.layers.Reshape(target_shape=(END, END, 1), name='reshape_2')(x2)

x2=keras.layers.Conv2D(32, 
                       kernel_size=(3,3), 
                       strides=1,
                       activation='relu',
                       name='second_conv64_52', 
                       padding='same')(x2)

x2=keras.layers.Conv2D(32, 
                       kernel_size=(3,3), 
                       strides=1,
                       activation='relu',
                       name='second_conv64_522', 
                       padding='same')(x2)

x2=keras.layers.Conv2D(16,
                       kernel_size=(3,3),
                       strides=1,
                       activation='relu',
                       name='second_conv64_524', 
                       padding='same')(x2)

x2=keras.layers.Conv2D(1,
                       kernel_size=3,
                       strides=1,
                       activation='relu',
                       name='second_conv64_53', 
                       padding='same')(x2)

conv_op = keras.layers.Reshape(target_shape=(END, END), name='reshape_3')(x2)

model = keras.Model(inputs=conv_ip, outputs=conv_op, name="predict_shape_from_LC")
model.summary()

print("Model is defined")

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
print("Model is compiled")

# Patience early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   mode='min', 
                                   verbose=1, 
                                   patience=30
)
print("Early stopping defined")

# Learning rate scheduler
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 20
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
lr_sched = keras.callbacks.LearningRateScheduler(step_decay)
print("Learning rate scheduler defined")

# Model checkpoint
checkpoint_path = base_model_folder+"april24_2024_model/ckpt/checkpoint.weights.h5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='min',
    save_freq='epoch',
    initial_value_threshold=None
)
print("Model checkpoint defined")

# Training 
print("Training will start now")

history = model.fit(train_dataset, 
                    epochs=no_epochs,
                    verbose=2, 
                    validation_data=vald_dataset,
                    callbacks=[es, lr_sched, model_checkpoint_callback]
)

# Save Model
save_model(model, str(model_save_path))
print("Model saved")

plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.savefig(base_model_folder+'history_loss_graph.png')
plt.close()

toc = time.perf_counter()
print(toc-tic, " s")

tf.keras.backend.clear_session()
print("End of code")