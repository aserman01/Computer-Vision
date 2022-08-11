#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:59:55 2022

@author: aniltanaktan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def SIFT(imageL,imageR):
    
    # Convert images into gray scale
    grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
    
    # Convert original images into RGB for display
    #imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
    #imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)
    
    # SIFT Keypoint Detection
    sift = cv2.xfeatures2d.SIFT_create()

    left_keypoints, left_descriptor = sift.detectAndCompute(grayL, None)
    right_keypoints, right_descriptor = sift.detectAndCompute(grayR, None)
    

    print("Number of Keypoints Detected In The Left Image: ", len(left_keypoints))
    
    print("Number of Keypoints Detected In The Right Image: ", len(right_keypoints))
    
    
    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False) # crossCheck is enabled to get better matching points
                                                        # crossCheck checks both matching points' distances    
                                                        # to use KNN Method, crosscheck needs to be False
                                                        
    matches = bf.match(left_descriptor, right_descriptor)

    # Get only matches with only a short distance (eliminate false matches)
    matches = sorted(matches, key = lambda x : x.distance)

    # We will only display first 100 matches for simplicity
    result = cv2.drawMatches(imageL, left_keypoints, imageR, right_keypoints, matches[:100], grayR, flags = 2)

    # Display the matches
    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Matches')
    plt.imshow(result)
    plt.show()
    
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imshow('SIFT Matches', result)
    
    # Print total number of matching points between the training and query images
    print("\nSIFT Matches are ready. \nNumber of Matching Keypoints: ", len(matches))
    cv2.waitKey(0)
    
    # KNN Matching
    
    ratio = 0.85 # This ratio will be a threshold value to check matches found by KNN
    raw_matches = bf.knnMatch(left_descriptor, right_descriptor, k=2)   # Using KNN we can find the best two matches
    good_points = []                                                    # for a point in first image.
    good_matches=[] 
    
    for match1, match2 in raw_matches: # We check every two matches for each point
        if match1.distance < match2.distance * ratio:               # If points inlies in our desired treshold 
            good_points.append((match1.trainIdx, match1.queryIdx))  # we declare them as good points.
            good_matches.append([match1])
    
    knnResult = cv2.drawMatchesKnn(imageL, left_keypoints, imageR, right_keypoints, good_matches, None, flags=2)
    cv2.imshow('KNN Matches', knnResult)
    
    print("\nKNN Matches are ready. \nNumber of Matching Keypoints: ", len(good_matches))
    cv2.waitKey(0)
    
    # Calculating Homography using good matches and RANSAC
    # I have selected ratio, min_match, RANSAC values according to a study by Caparas, Fajardo and Medina
    # said paper: https://www.warse.org/IJATCSE/static/pdf/file/ijatcse18911sl2020.pdf
    
    min_match = 10
    
    if len(good_points) > min_match: # Check if we have enough good points (minimum of 4 needed to calculate H)
        imageL_kp = np.float32(
            [left_keypoints[i].pt for (_, i) in good_points])
        imageR_kp = np.float32(
            [right_keypoints[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(imageR_kp, imageL_kp, cv2.RANSAC,5.0)    # H gives us a 3x3 Matrix for our
                                                                                # desired transformation.
    
    # Assigning Panaroma Height and Width
    
    height_imgL = imageL.shape[0] # Shape command gives us height and width of an image in a list 
    width_imgL = imageL.shape[1]  # 0 -> height, 1 -> width
    width_imgR = imageR.shape[1]
    height_panorama = height_imgL
    width_panorama = width_imgL + width_imgR
    
    # Creating a mask for better blending
    # Mask will be a weighted filter to make transition between images more seamlessly
    
    def create_mask(img1,img2,version):
        smoothing_window_size=800
        height_img1 = img1.shape[0] # Shape command gives us height and width of an image in a list 
        width_img1 = img1.shape[1]  # 0 -> height, 1 -> width
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(smoothing_window_size / 2)
        barrier = img1.shape[1] - int(smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])
    
    # Creating the panaroma
    
    height_img1 = imageL.shape[0]
    width_img1 = imageL.shape[1]
    width_img2 = imageR.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2
    
    panorama1 = np.zeros((height_panorama, width_panorama, 3))  # We create the shape of our panorama
    mask1 = create_mask(imageL,imageR,version='left_image')      
    panorama1[0:imageL.shape[0], 0:imageL.shape[1], :] = imageL
    panorama1 *= mask1
    mask2 = create_mask(imageL,imageR,version='right_image')
    panorama2 = cv2.warpPerspective(imageR, H, (width_panorama, height_panorama))*mask2
    result=panorama1+panorama2

    # Displaying all results
    
    cv2.imshow('Panorama_1', panorama1)

    print("\nPanorama 1 is ready")
    cv2.waitKey(0)
    
    cv2.imshow('Mask_1', mask1)

    print("\nMask_1 is ready")
    cv2.waitKey(0)
    
    cv2.imshow('Panorama_2', panorama2)

    print("\nPanorama 2 is ready")
    cv2.waitKey(0)
    
    cv2.imshow('Mask_2', mask2)

    print("\nMask_2 is ready")
    cv2.waitKey(0)
    
    # Get rid of black borders created by perspective differences
    
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]

   
    cv2.imwrite('Panorama_Final.png', final_result)

    print("\nPanorama Final is created with the name Panorama_Final.png")

    
                                                          
                                                                                
    
    
    
    

# Load images
image1 = cv2.imread('Problem/imageLeft.jpg')
image2 = cv2.imread('Problem/imageRight.jpg')

SIFT(image1,image2)
