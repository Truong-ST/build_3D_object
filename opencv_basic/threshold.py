import cv2
import matplotlib.pyplot as plt


inImage = cv2.imread('image/cube_dark.jpg', 0)
img = cv2.resize(inImage, (800,800))

# cv2.imshow('image', img)

# plt.imshow(img)
# plt.show()
threshold = 70
maxval = 120
_,th1 = cv2.threshold(img, threshold,maxval,cv2.THRESH_BINARY) # >T =1 and
_,th2 = cv2.threshold(img, threshold,maxval,cv2.THRESH_BINARY_INV) 
_,th3 = cv2.threshold(img, threshold,maxval,cv2.THRESH_MASK)
_,th4 = cv2.threshold(img, threshold,maxval,cv2.THRESH_TOZERO) # <T = 0
_,th5 = cv2.threshold(img, threshold,maxval,cv2.THRESH_TOZERO_INV)
_,th6 = cv2.threshold(img, threshold,maxval,cv2.THRESH_TRUNC) # >T = T
th7 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th8 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) # better
_,th9 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # can use gaussian blur to increase e

titles = ['original','THRESH_BINARY','THRESH_MASK','THRESH_TOZERO','THRESH_TOZERO_INV','THRESH_TRUNC', 'ADAPTIVE MEAN', 'ADAPTIVE GAUSSIAN', 'otsu']
images = [img, th1, th3, th4, th5, th6, th7, th8, th9]

for i in range (9):
    plt.subplot(3,3 , i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows