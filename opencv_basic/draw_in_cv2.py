import cv2 
import numpy as np


# inputImage = cv2.imread("CMB.jpg", 1)
# img = cv2.resize(inputImage,(1200,800)) 
img = np.zeros([600,800,3], np.uint8)

cv2.line(img, (0,0), (200,200),(255,0,0),10)
cv2.arrowedLine(img, (0,200), (200,200), (0,255,0),5)

cv2.rectangle(img,(200,0), (300,100), (0,0,255), -1 )

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'truong', (0,300), font, 5, (255,255,255), 5, cv2.LINE_AA)

cv2.circle(img, (200,200), radius=2, color=(0, 0, 255), thickness=-1)
cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

