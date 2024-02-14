This folder contains code to convert raw data to npy files.
Total number of programs in this folder: 9
No. of codes documented in readme: 4

1. To convert multiple light curves (for Bezier shapes) saved as csv into a single npy file
Use the code: lightcurve_simulation/data_npy/cnvt_lcFlux_to_npy.py
- To change to the size of the final light curve, you can adjust the parameter std_len_lc.
- Code you sample/interpolate the input light curve to match the std_len_lc parameter

2. To convert shapes with folder named with number (for Bezier shapes) in a folder into npy
Use the code: lightcurve_simulation/data_npy/cnvt_shape_to_npy.py

3. To convert multiple light curves (for general shapes) saved as csv into a single npy file
Use the code: lightcurve_simulation/data_npy/cnvt_general_lcFlux_to_npy.py

4. To convert shapes (for general shapes) in a folder into npy
Use the code: lightcurve_simulation/data_npy/cnvt_general_shape_to_npy.py
