import numpy as np
import cv2 as cv


img = cv.imread('build_3d_object/image/image_video/frame11.jpg')
img = cv.resize(img, (800,800))
# img = np.zeros((800,800,3), dtype=np.uint8)
# img = cv.bilateralFilter(img, 7, 25,25)
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# _,gray = cv.threshold(img, 125,255,cv.THRESH_BINARY) # >T =1 and

sift = cv.SIFT_create()
# kp, des = sift.detectAndCompute(gray,None)
kp = sift.detect(gray,None)
# pts = cv.KeyPoint_convert(kp)
# print(pts)
# print(kp[0].pt)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('sift_keypoints.jpg',img)
cv.imshow('img', img)
# cv.imshow('gray', gray)


cv.waitKey(0)
cv.destroyAllWindows()