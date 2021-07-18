import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pyvista as pv


'''Get point cloud 3D
'''

img1 = cv.imread('build_3d_object/image/image_video/frame22.jpg',0)  #queryimage # left image
img2 = cv.imread('build_3d_object/image/image_video/frame33.jpg',0) #trainimage # right image
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

E, mask = cv.findEssentialMat(pts1,pts2)
# print(E)

# fileter inlier point
pts1 = [pts1[mask.ravel()==1]]
pts2 = [pts2[mask.ravel()==1]]


# ---------------------- find 3D position --------------------------
# E,R,T are known
fix_T = np.array([[0,1,0],
               [1,0,0],
               [0,0,0]])
fix_R = fix_T
fix_R[2][2] = 1 


# U, s, V = np.linalg.svd(E, full_matrices=True)
# S = np.zeros((3,3))
# for i in range(len(s)):
#     S[i][i] = s[i]
# T = np.linalg.multi_dot([U, fix_T, S, V.transpose()])
# T = np.around(T, decimals=2)
# R = np.linalg.multi_dot([U, fix_T, V.transpose()])
# R = np.around(R, decimals=2)

# R1, R2, Tr = list(map(lambda x: np.around(x, decimals=2), cv.decomposeEssentialMat(E)))
R1, R2, Translate = [np.around(i, decimals=2) for i in cv.decomposeEssentialMat(E)]
# print(R2)
# print(Translate)


def trigulate_3D(K, R1, T1, R2, T2, imagePoints1, imagePoints2):
    P1 = np.hstack([R1.T, -R1.T.dot(T1)])
    P2 = np.hstack([R2.T, -R2.T.dot(T2)])
    P1 = K.dot(P1)
    P2 = K.dot(P2)

    # Triangulate
    list_point3D = [] 
    for i in range(len(imagePoints1)):
        point = cv.triangulatePoints(P1, P2, imagePoints1[i], imagePoints2[i]).T
        point = point[:, :3] / point[:, 3:4]
        list_point3D.append(point)
        
    return list_point3D


def error_project(K, R1, T1, R2, T2, points3D, imagePoints1, imagePoints2):
    # Reproject back into the two cameras
    rvec1, _ = cv.Rodrigues(R1.T)
    rvec2, _ = cv.Rodrigues(R2.T)

    list_error = []
    # measure difference between original image point and reporjected image point 
    for i in range(len(points3D)):
        p1, _ = cv.projectPoints(points3D[i], rvec1, -T1, K, distCoeffs=None)
        p2, _ = cv.projectPoints(points3D[i], rvec2, -T2, K, distCoeffs=None)

        reprojection_error1 = np.linalg.norm(imagePoints1[i] - p1[0, :])
        reprojection_error2 = np.linalg.norm(imagePoints2[i] - p2[0, :])
        list_error.append([reprojection_error1, reprojection_error2])
    return list_error


K = np.array([
    [718.856 ,   0.    ,   426.],
    [  0.    , 718.856 ,   240.],
    [  0.    ,     0.  ,     1.],
])
R1 = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])
R2 = np.array([
    [ 0.99999183 ,-0.00280829 ,-0.00290702],
    [ 0.0028008  , 0.99999276, -0.00257697],
    [ 0.00291424 , 0.00256881 , 0.99999245]
])
T1 = np.array([[0.], [0.], [0.]])
T2 = np.array([[-0.02182627], [ 0.00733316], [ 0.99973488]])

# point cloud
imagePoint1 = np.array([371.91915894, 221.53485107])
imagePoint2 = np.array([368.26071167, 224.86262512])
points1 = list(map(lambda x: np.array(x).astype(float), pts1[0]))
points2 = list(map(lambda x: np.array(x).astype(float), pts2[0]))
X = trigulate_3D(K, R1, T1, R2, T2, points1, points2)
print(error_project(K, R1,T1,R2,T2,X,points1, points2))
X = list(map(lambda x: x[0], X))
print(X)

cloud = pv.PolyData(X)
cloud.plot()




