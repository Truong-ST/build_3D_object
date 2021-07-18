import numpy as np
from detect.detect_canny_edge import *
import cv2 as cv
import math
import random
import matplotlib.pyplot as plt
from utils import *
from structure.node import Node
from structure.graph import CharacteristicGraph


img = cv.imread('build_3d_object/image/cube_dark.jpg')
width, height = img.shape[:2]
if width > 1000:
    img = cv.resize(img, (int(width/4), int(height/4)))
    width, height = img.shape[:2]
# cv.imshow('origin', img)

# preprocess   *****************************************************************
blur = cv.bilateralFilter(img,9,75,75)
blur5 = cv.filter2D(img, -1, np.ones((5,5), dtype=int)/25)
strong_blur = cv.bilateralFilter(blur5,9,75,75)
hsv_blur = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect edge ****************************************************************************
img_canny = canny_edge_detection(strong_blur, 5, 120, 50)
image = cv.addWeighted(img, 1.15, img,0, gamma=-5)
image_canny = canny_edge_detection(image, 5, 120, 50)
final_edge = cv.addWeighted(image_canny,0.5, img_canny,0.5, 0)
cv.imshow('final_edge',final_edge)
final_edge = dilate_erode(final_edge, 2, 1, 2, 1)
cv.imshow('dier',final_edge)

# cv.imshow('image_canny',image_canny)
# cv.imshow('img_canny',img_canny)

# contour ****************************************************************************
_, thresh = cv.threshold(final_edge, 100, 255,cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
print(len(contours))
# print(hierarchy)

final_contours = []
for cont in contours:
    approx = cv.approxPolyDP(cont,0.01*cv.arcLength(cont,True), True)
    area = cv.contourArea(cont)
    if area / len(approx) > 150:
        if area > 1600:
            final_contours.append(cont)


print(len(final_contours))
cv.drawContours(img, final_contours, -1, (0,255,0), 2)
# cv.imshow('cont', img)

normal_final = []
for cont in final_contours:
    con = []
    for c in cont:
        con.append(c[0])
    normal_final.append(con)


# detect color *********************************************************************
# get list of sample point
sample_list = []
n_per_contour = 5
for i in range(1,len(normal_final)):
    l = []
    for j in range(n_per_contour):
        sample = random.sample(list(normal_final[i]),16)
        l.append(np.mean(sample, axis=0).astype(int))
    sample_list.append(l)

# find color from sample
hsv_list = []
for conto in sample_list:
    color = []
    for point in conto:
        # print(point)
        color.append(hsv_blur[point[1]][point[0]])
    hsv_list.append(np.mean(color,axis=0).astype(int))


# mask
# low_hsv = hsv_list[0]-50
# high_hsv = hsv_list[0]+50
# mask = cv.inRange(hsv_blur,low_hsv,high_hsv)
# result = cv.bitwise_and(blur,blur,mask=mask)
# cv.imshow('result', result)
            

# feature node ************************************************************************
# node : number of vertices, number of edge, color, para, shape*0.5, Iv, Ie
nodes = [[] for i in range(len(final_contours)-1)]
color = ['green', 'red', 'blue', 'orange', 'yellow','pink']
approxes = []
for i, contour in enumerate(final_contours[1:]):
    approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True), True)
    approxes.append(approx)
    # cv.drawContours(img, [approx], 0,(0,0,0),3)
    # x,y,*_ = approx.ravel()
    # cv.putText(img, str(i),(x,y),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0))
    nodes[i].append(len(approx))
    nodes[i].append(len(approx)) # if closed contour then vertices = edges
    nodes[i].append(hsv_list[i])
    para = parallel(approx, width)
    nodes[i].append(len(para))
    # print(para)
    if i ==0:
        # print(approx)
        for j in range(len(approx)):
            cv.circle(img, tuple(approx[j][0]), radius=4, color=(0,0,50*j), thickness=-1)
        # print(para[0][0])
        # cv.line(img, tuple(para[0][0][0]),tuple(para[0][0][1]), (0,0,255), 3)
        # cv.line(img, tuple(para[0][1][0]),tuple(para[0][1][1]), (0,0,255), 3)


# interact
cv.imshow('image',img)
# cv.imshow('final', final_contours)
print('nodes:',nodes)


# buid graph *************************************************************************
cgraph = CharacteristicGraph()
cgraph.nodes = [Node(*nodes[i]) for i in range(len(final_contours)-1)]
# print(cgraph.nodes)

for i in range(len(final_contours)-2):
    for j in range(i+1,len(final_contours)-1):
        common = find_common_edges(approxes[i], approxes[j])
        if common > 0:
            cgraph.connect_node(i, j, common)
print('edges: ',cgraph.edges)
print()


# compare graph ****************************************************************************************

# build object ****************************************************************************************



cv.waitKey(0)
cv.destroyAllWindows()