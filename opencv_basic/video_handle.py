import numpy as np
import cv2 as cv


cap = cv.VideoCapture('image/cube.mov')
while cap.isOpened():
    ret, frame = cap.read()
    width, height = frame.shape[:2]
    frame = cv.resize(frame, (int(width / 2), int(height / 2)))
    width, height = frame.shape[:2]
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', frame)
    # cv.imshow('gray', gray)
    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()