import numpy as np
import csv
import cv2 as cv
import random


def clean_edge(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 127, 255, 0)
    con = image
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(con, contours, -1, (0,255,0), 3)
    return con


def parallel(list_point, width):
    rs = []
    l = len(list_point)
    maxl = l-1
    for i in range(maxl):
        n1 = list_point[-1][0] - list_point[i][0]
        mid1 = (list_point[maxl][0] + list_point[i][0])/2
        n2 = list_point[(i+1)%maxl][0] - list_point[(i+2)%maxl][0]
        mid2 = (list_point[(i+1)%maxl][0] + list_point[(i+2)%maxl][0])/2
        d = np.linalg.norm(mid1-mid2)
        # print(math.acos(abs(n1.dot(n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)))))
        
        if abs(n1.dot(n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))) > 0.95-0.1*d/width: # add d
            # print('parallel', n1, n2)
            rs.append([(list_point[-1][0],list_point[i][0]), (list_point[(i+1)%maxl][0], list_point[(i+2)%maxl][0])])
            
    return rs


def smooth_matrix(matix, rangeR):
    pass


def find_common_edges(vertices1, vertices2):
    common = 0
    for i in range(len(vertices1)):
        for j in range(len(vertices2)):
            if np.linalg.norm(vertices1[i][0]-vertices2[j][0]) < 12:
                common += 1
    return common


def run(img, final_edge):
    width = 800
    _, thresh = cv.threshold(final_edge, 100, 255, cv.THRESH_BINARY)
    blur = cv.bilateralFilter(img, 9, 75, 75)
    hsv_blur = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    final_contours = []
    for cont in contours:
        area = cv.contourArea(cont)
        # approx = cv.approxPolyDP(cont, 0.01 * cv.arcLength(cont, True), True)
        if area > 1000:
            final_contours.append(cont)
    print(len(final_contours))
    cv.drawContours(img, final_contours, -1, (0, 255, 0), 2)

    normal_final = []
    for cont in final_contours:
        con = []
        for c in cont:
            con.append(c[0])
        normal_final.append(con)

    # get list of sample point
    sample_list = []
    n_per_contour = 5
    for i in range(1, len(normal_final)):
        l = []
        for j in range(n_per_contour):
            sample = random.sample(list(normal_final[i]), 16)
            l.append(np.mean(sample, axis=0).astype(int))
        sample_list.append(l)

    # find color from sample
    hsv_list = []
    for conto in sample_list:
        color = []
        for point in conto:
            color.append(hsv_blur[point[1]][point[0]])
        hsv_list.append(np.mean(color, axis=0).astype(int))


    # feature node ************************************************************************
    # node : number of vertices, number of edge, color, para, shape*0.5, Iv, Ie
    nodes = [[] for i in range(len(final_contours) - 1)]
    color = ['green', 'red', 'blue', 'orange', 'yellow', 'pink']
    approxes = []
    for i, contour in enumerate(final_contours[1:]):
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        approxes.append(approx)
        # cv.drawContours(img, [approx], 0,(0,0,0),3)
        # x,y,*_ = approx.ravel()
        # cv.putText(img, str(i),(x,y),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0))
        nodes[i].append(len(approx))
        nodes[i].append(len(approx))  # if closed contour then vertices = edges
        nodes[i].append(hsv_list[i])
        para = parallel(approx, width)
        nodes[i].append(len(para))
        # print(para)

    print(nodes)
    return img