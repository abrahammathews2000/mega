# Alien Megastructure in Transit

This repository contains the codes I am using for my Master's Thesis.

All the datasets and final trained model are uploaded in this Google Drive Link: https://drive.google.com/drive/folders/15Cz5QkND9pC6HosmHSYo_b3DK3SVAEVp?usp=sharing

The main steps involved in this projects is as follows:

1. Generated arbitrary shapes for the transiting object using Bezier curves and save them as image files. I have used one publicly available code to generate the random shapes.

2. Generated transit light curves for the each shape, we used the package EightBitTransit for this simulations.

3. Saved both images and transit light curves as .npy file.

4. We used a 1D CNN machine learning model for mapping transit light curve flux value to the 2D shadow image.

5. After training the model, we  tested it on analytical planet light curves from BATMAN package.