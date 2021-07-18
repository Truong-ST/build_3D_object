import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# cat theo 2 anh 1 eg. 12-14  18-20
imgL = cv.imread('build_3d_object/image/house.jpg',0)
imgR = cv.imread('build_3d_object/image/house1.jpg',0)
stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()