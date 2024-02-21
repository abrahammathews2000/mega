This folder contains codes to create raw datas
Total number of programs in this folder: 16
No. of codes documented in readme: 5
1. To generate light curves of a set of shapes saved in single .npy file
Use code: lightcurve_simulation/data_raw/pll_create_8bit_lc.py
- You can change limb darkening coefficients, star2mega_ratio
- You have to give the file path of npy file where shapes are stored
- You have to also give the output path to save the light curves
The above code is dependent on: lightcurve_simulation/data_raw/pll_lc_curve_generator.py
- In this code you can work with EightBitTransit Parameters like, limb darkening law, velocity of objects,
observation time, ...

2. To generate light curves of a set of shapes saved in multiple .npy file
Use code: lightcurve_simulation/data_raw/batch_pll_create_8bit_lc.pys
- You can change limb darkening coefficients, star2mega_ratio
- You have to give the file path of npy file where shapes are stored
- You have to also give the output path to save the light curves
The above code is dependent on: lightcurve_simulation/data_raw/batch_pll_lc_curve_generator.py
- In this code you can work with EightBitTransit Parameters like, limb darkening law, velocity of objects,
observation time, ...

3. To first convert RGB images to single channel 
Use the code: lightcurve_simulation/data_raw/cnvt_rgb_to_gray.py

4. lightcurve_simulation/data_raw/plot_circle.ipynb
- to plot circles with different size within 38 x 38 pixel