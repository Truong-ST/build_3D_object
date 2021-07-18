import numpy as np
import cv2 as cv
import yaml
import matplotlib.pyplot as plt
import pyvista as pv
from image_matching import list_image_matching


file = open("build_3d_object/meshroom/config.yaml", 'r')
parsed_yaml_file = yaml.safe_load(file)
file.close()
number_image = parsed_yaml_file['number_image']
frame_per_image = parsed_yaml_file['frame_per_image']

pairs_matching = list_image_matching(number_image, frame_per_image)

# find the keypoints and descriptors with SIFT
sift = cv.SIFT_create()
kp = [i for i in range(number_image)]
des = [i for i in range(number_image)]
for i in range(number_image):
    img = cv.imread('build_3d_object/image/image_video/frame{}.jpg'.format(i*frame_per_image),0)
    kp[i], des[i] = sift.detectAndCompute(img,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)

def get_pts(kp1, des1, kp2, des2):
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    try:
        E, mask = cv.findEssentialMat(pts1,pts2)

        # fileter inlier point
        pts1 = [pts1[mask.ravel()==1]]
        pts2 = [pts2[mask.ravel()==1]]

        return pts1, pts2, E
    except:
        return None, None, None

# ---------------------- find 3D position --------------------------
K = np.array([  [2.64079833e+03, 0.00000000e+00, 1.26071294e+03],
                [0.00000000e+00, 2.64002704e+03, 1.76064955e+03], 
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])



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


def get_pair_point_cloud(R1, T1, R2, T2, pts1, pts2):
    global K
    points1 = list(map(lambda x: np.array(x).astype(float), pts1[0]))
    points2 = list(map(lambda x: np.array(x).astype(float), pts2[0]))
    cloud = trigulate_3D(K, R1, T1, R2, T2, points1, points2)
    # cloud = list(map(lambda x: x[0], cloud))
    return cloud

pose_cam = []

# point cloud
point_cloud = []
accumulateR = np.identity(3, dtype=float)
accumulateT = [[0.], [0.], [0.]]
nextR = accumulateR
nextT = accumulateT
for i in range(len(pairs_matching)-1):
    pts1, pts2, E = get_pts(kp[i], des[i], kp[i+1], des[i+1])
    if pts1 == None:
        continue
    accumulateR = nextR
    accumulateT = nextT
    # print(np.ravel(accumulateT))
    # pose_cam.append(accumulateR.copy().dot(np.ravel(accumulateT).copy()))
    # print(accumulateR)
    try:
        R1, R2, T = [np.around(i, decimals=2) for i in cv.decomposeEssentialMat(E)]
    except:
        continue
    nextR = nextR.dot(R1)
    nextT += T

    cloud = get_pair_point_cloud(accumulateR, accumulateT, nextR, nextT, pts1, pts2)
    points1 = list(map(lambda x: np.array(x).astype(float), pts1[0]))
    points2 = list(map(lambda x: np.array(x).astype(float), pts2[0]))
    # print(error_project(K, accumulateR, accumulateT, nextR, nextT,cloud,points1,points2))
    cloud = list(map(lambda x: x[0], cloud))
    for i in range(len(cloud)):
        point_cloud.append(cloud[i])

print(len(point_cloud))
print(np.round(point_cloud, 1))

cloud = pv.PolyData(point_cloud)
# cloud = pv.PolyData(pose_cam)
cloud.plot()

# volume = cloud.delaunay_3d(alpha=3.)
# shell = volume.extract_geometry()
# shell.plot()





