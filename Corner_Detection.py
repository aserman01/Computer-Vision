#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:16:38 2022

@author: aniltanaktan
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read in the image
imageL = cv2.imread('Images/imageLeft.jpg')
imageR = cv2.imread('Images/imageRight.jpg')

def corner_detect(image):
    # Make a copy of the image
    image_copy = np.copy(image)
    
    # Change color to RGB (from BGR)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    
    ''' uncomment to see your image in grayscale
    plt.imshow(gray, cmap="gray") # cmap='gray' let us view the gray image as it is
    '''
    
    # Detect corners using Harris Corner Detection (image, neighborhood size, Sobel derivative, k = 0.04)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Dilate image to enhance corner points
    dst = cv2.dilate(dst,None)
    
    #########
    
    # Threshold value determines how many corners you desire to accept as valid and results 
    # may vary with each image. Try changing 0.08 and find best value for yourself.
    thresh = 0.08*dst.max()
    
    # Create an image copy to draw corners on
    corner_image = np.copy(image)
    
    # Iterate through all the corners and draw them on the image (if they pass the threshold)
    for j in range(0, dst.shape[0]): # Select each row
        for i in range(0, dst.shape[1]): # check every collumn at that row
            if(dst[j,i] > thresh): # Check if pixel is bigger than threshold
            
                # image, center pt, radius, color, thickness
                cv2.circle( corner_image, (i, j), 1, (0,255,0), 1) # Put a green dot on that pixel
                
    return corner_image
            
cv2.imshow("Corners of ImgL", corner_detect(imageL))
cv2.imshow("Corners of ImgR", corner_detect(imageR))
cv2.waitKey(0)

