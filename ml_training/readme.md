This folder contains contains code for ml training and testing

1. ml_training/calc_MSE_predict_shapes.ipynb
- This code is used to calculate Mean Square Error (MSE) of the predicted image by folding in the middle horizontally

2. ml_training/compare_ml_model_snr_v1.ipynb
- This code is used to compare the performance of ML model predictions for lightcurves of different SNR.

3. ml_training/ML_model_cyberplanet_v4_noise.ipynb
- This code is used to add noise such that the distribution is uniform distribution of SNR from 50 to 500. Then train an ML model on this.

4. ml_training/ML_model_cyberplanet_v3_noise.ipynb
- This code was used to give Gaussian noise for single SNR

5. ml_training/ML_model_v5_noise_nine_LDCs-training_only.ipynb
- This code is used to train an ML model on light curves for 9 limb darkening cases simultaneously [a,b] =[0.1,0.5] to [0.9,0.5]

6. ml_training/ML_model_v5_noise_nine_LDCs-testing_only.ipynb
- This code is used to test the ML model (trained using ml_training/ML_model_v5_noise_nine_LDCs-training_only.ipynb) on simulated test light curves for 9 limb darkening cases simultaneously [a,b] =[0.1,0.5] to [0.9,0.5]