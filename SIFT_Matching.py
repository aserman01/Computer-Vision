#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:46:39 2022

@author: aniltanaktan
"""
import cv2
import matplotlib.pyplot as plt


def SIFT(imageL,imageR):
    
    # Convert images into gray scale
    grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
    
    # Convert original images into RGB for display
    imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
    imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)
    
    # SIFT Keypoint Detection
    sift = cv2.xfeatures2d.SIFT_create()

    left_keypoints, left_descriptor = sift.detectAndCompute(grayL, None)
    right_keypoints, right_descriptor = sift.detectAndCompute(grayR, None)
    

    print("Number of Keypoints Detected In The Left Image: ", len(left_keypoints))
    
    print("Number of Keypoints Detected In The Right Image: ", len(right_keypoints))
    
    
    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)  #crossCheck is enabled to get better matching points
                                                        #crossCheck checks both matching points' distances    
                                                        
    matches = bf.match(left_descriptor, right_descriptor)

    # Get only matches with only a short distance (eliminate false matches)
    matches = sorted(matches, key = lambda x : x.distance)

    # We will only display first 1000 matches for simplicity
    result = cv2.drawMatches(imageL, left_keypoints, imageR, right_keypoints, matches[:1000], grayR, flags = 2)

    # Display the matches
    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Matches')
    plt.imshow(result)
    plt.show()

    # Print total number of matching points between the training and query images
    print("\nImage is ready. \nNumber of Matching Keypoints: ", len(matches))
    

# Load images
image1 = cv2.imread('imageLeft.jpg')
image2 = cv2.imread('imageRight.jpg')

SIFT(image1,image2)
