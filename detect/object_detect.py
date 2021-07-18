from os import lseek
import cv2 as cv
import numpy as np


def nothing():
    pass


cv.namedWindow('tracking')
cv.createTrackbar('Low_Hue', 'tracking', 2, 255, nothing)
cv.createTrackbar('Low_Saturation', 'tracking', 2, 255, nothing)
cv.createTrackbar('Low_Value', 'tracking', 2, 255, nothing)
cv.createTrackbar('High_Hue', 'tracking', 200, 255, nothing)
cv.createTrackbar('High_Saturation', 'tracking', 200, 255, nothing)
cv.createTrackbar('High_Value', 'tracking', 200, 255, nothing)

cap = cv.VideoCapture('image/cube.mov')
# font = cv.FONT_HERSHEY_COMPLEX
while True:
    _, frame = cap.read()
    # frame = cv.imread('image/cube_light.jpg')
    frame = cv.resize(frame, (500, 500))
    frame = cv.bilateralFilter(frame, 9, 75,75) # blur image
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    l_h = cv.getTrackbarPos('Low_Hue', 'tracking')
    l_s = cv.getTrackbarPos('Low_Saturation', 'tracking')
    l_v = cv.getTrackbarPos('Low_Value', 'tracking')
    h_h = cv.getTrackbarPos('High_Hue', 'tracking')
    h_s = cv.getTrackbarPos('High_Saturation', 'tracking')
    h_v = cv.getTrackbarPos('High_Value', 'tracking')
    
    low_b = np.array([l_h, l_s, l_v])
    high_b = np.array([h_h, h_s, h_v])
    mask = cv.inRange(hsv_frame, low_b, high_b)
    # kernel = np.ones((5,5), np.uint8)
    # mask = cv.erode(mask, kernel)
    result = cv.bitwise_and(frame, frame, mask=mask)
    
    # find countors
    # contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     area = cv.contourArea(cnt)
    #     approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
    #     x, y, *_ = approx.ravel()
    #     if area > 100:
    #         cv.drawContours(frame, [approx], 0, (0, 0, 0), 5) 

    #         if len(approx == 4):
    #             cv.putText(frame, 'rectangle', (x, y), font, 1, (0,0,0))
            
    
    # cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('result', result)
    
    key = cv.waitKey(1)
    if key == 27:
        break
    # track with 3 color 
    # # Red color
    # low_red = np.array([161, 155, 84])
    # high_red = np.array([179, 255, 255])
    # red_mask = cv.inRange(hsv_frame, low_red, high_red)
    # red = cv.bitwise_and(frame, frame, mask=red_mask)
    # # Blue color
    # low_blue = np.array([94, 80, 2])
    # high_blue = np.array([126, 255, 255])
    # blue_mask = cv.inRange(hsv_frame, low_blue, high_blue)
    # blue = cv.bitwise_and(frame, frame, mask=blue_mask)
    # # Green color
    # low_green = np.array([25, 52, 72])
    # high_green = np.array([102, 255, 255])
    # green_mask = cv.inRange(hsv_frame, low_green, high_green)
    # green = cv.bitwise_and(frame, frame, mask=green_mask)
    # # Every color except white
    # low = np.array([0, 42, 0])
    # high = np.array([179, 255, 255])
    # mask = cv.inRange(hsv_frame, low, high)
    # result = cv.bitwise_and(frame, frame, mask=mask)
    
    # cv.imshow("Frame", frame)
    # cv.imshow("Red", red)
    # cv.imshow("Blue", blue)
    # cv.imshow("Green", green)
    # cv.imshow("Result", result)
    # if (cv.waitKey(1) & 0xFF == ord('q')):
    #     break
    

# cap.release()
cv.destroyAllWindows()