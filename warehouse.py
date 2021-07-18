import cv2 as cv
import numpy as np




# --- use for gamma factor image ---
# lookUpTable = np.empty((1,256), np.uint8)
# gamma = 0.8
# for i in range(256):
#     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
# res = cv.LUT(image, lookUpTable)
# cv.imshow('res', res)


# --- use laplacian for detect edge ---
# laplacian = cv.Laplacian(blur,-1)
# laplacian = cv.addWeighted(laplacian, 8, laplacian, 0,0)
# cv.imshow('laplacian', laplacian)


# --- mouse event ---
# def draw_circle(event,x,y,flags,param):
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(img,(x,y),100,(255,0,0),-1)
# # Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image',draw_circle)