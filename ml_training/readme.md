1. /scratch/abraham/Documents/mega_git/mega/ml_training/ml_train_lightcurve_unf_dist_LDC_size.py

- .py code help to run the code in screen mode .ipynb cannot do that
- This code can be used to load light curve training dataset and do the
- preprocessing like vertical scaling and extending the light curve
- CNN Model is defined
- Training, learning rate scheduler and early stopping feature included

2. /scratch/abraham/Documents/mega_git/mega/ml_training/ml_inference_lc2shape.ipynb

This code is used to infer transit shape from input light curve.
It involves following steps:
- Load ML model
- Load input data - i.e. the light curves
- Predict the shapes from the light curves
- Load True Output (if any)

Miscellance Operations
- Plot input lightcurve
- Show predicted output
- Show true output

3. /scratch/abraham/Documents/mega_git/mega/ml_training/Google_Colab_Codes

This folder contains the final codes used to train our final model. Also inference on 4 difference cases are also done:
1. Test Lightcurves from EightBitTransit
2. Ideal Exoplanet light curves from BATMAN package
3. Kepler lightcurves downloaded using Lightkurve package
4. Exocomet light curve fit