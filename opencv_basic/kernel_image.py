"""
@file filter2D.py
@brief Sample code that shows how to implement your own linear filters by using filter2D function
"""
import cv2 as cv
import numpy as np

img = cv.imread('image/cube_dark.jpg')
img = cv.resize(img, (800, 800))
kernel_size = 5 
# kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)/(kernel_size*kernel_size)
kernel = np.array([[1,1,1,1,1],
                  [1,2,2,2,1],
                  [1,2,3,2,1],
                  [1,2,2,2,1],
                  [1,1,1,1,1]])/35

dst = cv.filter2D(img, -1, kernel)
# laplacian = np.array([[0,1,0],
#                    [1,-4,1],
#                    [0,1,0]])
# laplacian_img = cv.filter2D(img, -1, laplacian)
# laplacian_img = cv.addWeighted(laplacian_img,6,laplacian_img,0,0)
sharp  =np.array([[0,-1,0],
                  [-1,5,-1],
                  [0,-1,0]])
sharper = cv.filter2D(img, -1, kernel)
cv.imshow('img', img)
cv.imshow('dst', dst)
# cv.imshow('laplace_img', laplacian_img)
cv.imshow('sharper', sharper)
cv.waitKey(0)
cv.destroyAllWindows()
    