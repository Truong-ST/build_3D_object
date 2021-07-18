import cv2
import numpy as np


def canny_edge_detection(img, blur_ksize=5, threshold1=100, threshold2=200):
    """
    image_path: link to image
    blur_ksize: Gaussian kernel size
    threshold1: min threshold 
    threshold2: max threshold
    """
    
    # img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)
    # img_canny = cv2.Canny(img_gaussian,threshold1,threshold2)
    img_bila = cv2.bilateralFilter(img,9,75,75)
    img_canny = cv2.Canny(img_bila,threshold1,threshold2)

    return img_canny

def dilate_erode(image, kernel_dilate_size, iter_dilate, kernel_erode_size,  iter_erode):
    kernel_dilate = np.array((kernel_dilate_size, kernel_dilate_size))
    kernel_erode = np.array((kernel_erode_size, kernel_erode_size))
    # dilate = cv2.dilate(image, kernel=kernel_dilate, iterations=iter_dilate)
    # dilate = cv2.filter2D(image, 2, kernel_dilate)
    dilatation_size = kernel_dilate_size
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilate = cv2.dilate(image, element, iter_dilate)
    erosion_size = kernel_erode_size
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    erosion_dst = cv2.erode(dilate, element, iterations=iter_erode)
    # erode = cv2.erode(dilate, kernel=kernel_erode, iterations=iter_erode)
    return erosion_dst


    
if __name__ == '__main__':
    # image_path = 'cube_light.jpg'
    img = cv2.imread('image/cube_light.jpg')
    # gray = cv2.imread(image_path, 0)
    img = cv2.resize(img, (800,800))
    img_canny = canny_edge_detection(img, 5, 120, 80)
    cv2.imshow('image', img_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()