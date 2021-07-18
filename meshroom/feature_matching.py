import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img1 = cv.imread('build_3d_object/image/image_video/frame11.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('build_3d_object/image/image_video/frame22.jpg',cv.IMREAD_GRAYSCALE)        # trainImage

# Initiate ORB (oriented BRIEF) detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1, des2, k=2)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()