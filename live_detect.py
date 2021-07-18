import cv2 as cv
import numpy as np
from detect.detect_canny_edge import canny_edge_detection, dilate_erode
# from function_main import run

# cap = cv.VideoCapture('image/cube_around.mov')
cap = cv.VideoCapture(0)
# font = cv.FONT_HERSHEY_COMPLEX

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv.resize(frame, (800,800))
        # blur = cv.bilateralFilter(frame, 7, 75,75) # blur image
        gray= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv.drawKeypoints(gray,kp,frame)
        cv.imshow('img', img)
    
        # hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # # cv.imshow('frame', frame)

        # img_canny = canny_edge_detection(blur, 5, 120, 50)
        # image = cv.addWeighted(frame, 1.2, frame,0, gamma=-2)
        # image_canny = canny_edge_detection(image, 5, 120, 50)

        # final_edge = cv.addWeighted(image_canny,0.5,img_canny, 0.5,0)
        # final_edge = dilate_erode(final_edge, 1, 1, 1, 1)
        # imag = run(frame, final_edge)
        # cv.imshow('cont', imag)

        # # cv.imshow('final_edge', final_edge)
        # cv.imshow('diler', final_edge)

        if cv.waitKey(1) == 27:
            break
    else: 
        break

cap.release()
cv.destroyAllWindows()