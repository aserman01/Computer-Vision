#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:59:55 2022

@author: aniltanaktan
"""

import cv2
import numpy as np


def ImageStitching(imageL,imageR, outname):
    
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
    
    cv2.imshow('SIFT Matches', result)
    
    # Print total number of matching points between the training and query images
    print("\nSIFT Matches are ready. \nNumber of Matching Keypoints: ", len(matches))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # KNN Matching
    
    ratio = 0.85 # This ratio will be a threshold value to check matches found by KNN
    raw_matches = bf.knnMatch(left_descriptor, right_descriptor, k=2)   # Using KNN we can find the best two matches
    good_points = []                                                    # for a point in first image.
    good_matches=[] 
    
    for match1, match2 in raw_matches: # We check every two matches for each point
        if match1.distance < match2.distance * ratio:               # If points inlies in our desired treshold 
            good_points.append((match1.trainIdx, match1.queryIdx))  # we declare them as good points.
            good_matches.append([match1])
    
    
    # We will only display first 100 matches for simplicity
    knnResult = cv2.drawMatchesKnn(imageL, left_keypoints, imageR, right_keypoints, good_matches[:100], None, flags=2)
    cv2.imshow('KNN Matches', knnResult)
    
    print("\nKNN Matches are ready. \nNumber of Matching Keypoints: ", len(good_matches))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Calculating Homography using good matches and RANSAC
    # I have selected ratio, min_match and RANSAC values according to a study by Caparas, Fajardo and Medina
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
    # For creating this mask function I used the code of linrl3 (https://github.com/linrl3)
    # His work also give me many inspirations for this project
    
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
        if version== 'left_image':  # Used for creating mask1
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:                       # Used for creating mask2 
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])
    
    # Creating the panorama
    
    height_img1 = imageL.shape[0]
    width_img1 = imageL.shape[1]
    width_img2 = imageR.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2
    
    panorama1 = np.zeros((height_panorama, width_panorama, 3))  # 1. create the shape of our panorama
    mask1 = create_mask(imageL,imageR,version='left_image')     # 2. create our mask with this shape
    panorama1[0:imageL.shape[0], 0:imageL.shape[1], :] = imageL # 3. include color of each pixel to the shape
    panorama1 *= mask1                                          # 4. apply our mask to panorama
    mask2 = create_mask(imageL,imageR,version='right_image')
    
    #For right half of the panorama, we warp it with H we found and apply the mask
    panorama2 = cv2.warpPerspective(imageR, H, (width_panorama, height_panorama))*mask2
    result=panorama1+panorama2 #We combine both of them to have our result


    #Normalize panoramas for display
    norm_p1 = cv2.normalize(panorama1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_p2 = cv2.normalize(panorama2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Displaying all results
    
    cv2.imshow('Panorama_1', norm_p1)
    
    print("\nPanorama 1 is ready")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Mask_1', mask1)

    print("\nMask_1 is ready")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Panorama_2', norm_p2)

    print("\nPanorama 2 is ready")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('Mask_2', mask2)

    print("\nMask_2 is ready")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Get rid of black borders created by perspective differences
    
    rows, cols = np.where(result[:, :, 0] != 0) # Check if a pixel is pure black or not (0-255)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :] # Resize image without black borders

    norm_pf = cv2.normalize(final_result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite(outname+'.png', final_result)
    
    
    
    cv2.imshow(outname, norm_pf)
    
    print("\nFinal Panorama is created with the name "+outname+".png")
    cv2.waitKey(0)
    
    # A simple code to fix a bug preventing last image window to close
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)
        return


    
                                                          
                                                                                
    
    
    
    
'''
# Load images
image1 = cv2.imread('Problem/test1.jpg')
image2 = cv2.imread('Problem/test2.jpg')
'''

output_name = "Panorama_Final32"
image1 = cv2.imread('Problem/imageLeft.jpg')
image2 = cv2.imread('Problem/imageRight.jpg')


ImageStitching(image1,image2, output_name) #(image1, image2, name of the output file)
