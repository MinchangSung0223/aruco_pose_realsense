# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
import pyrealsense2 as rs
import numpy as np
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import cv2.aruco as aruco
import sys
import os
import pickle
import math
# Check for camera calibration data

cameraMatrix = np.load('mtx.npy')
distCoeffs = np.load('dist.npy')
if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration/CameraCalibration.pckl and recalibrate your camera with calibration_ChAruco.py.")
        exit()


def drawCube(img, corners, imgpts,ids):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color

    for i,j in zip(range(4),range(4,8)):
        #
        if(ids == 5):
           img = cv2.line(img, tuple(imgpts[i+8]), tuple(imgpts[j+8]),(0,255,0),2) 
        elif(ids == 9):
           img = cv2.line(img, tuple(imgpts[i+36]), tuple(imgpts[j+36]),(255),3)
        else:
           img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    #img = cv2.drawContours(img, [imgpts[4:8]],-1,(0,0,255),3)
    print(imgpts.shape)
    if(ids == 5):
       img = cv2.drawContours(img, [imgpts[8:12]],-1,(0,255,0),2)
       img = cv2.drawContours(img, [imgpts[12:16]],-1,(0,255,0),2)
       img = cv2.drawContours(img, [imgpts[16:36]],-1,(0,0,255),5)
    elif(ids == 9):
       img = cv2.drawContours(img, [imgpts[40:44]],-1,(255,0,0),3)
       img = cv2.drawContours(img, [imgpts[36:40]],-1,(255,0,0),3)
       img = cv2.drawContours(img, [imgpts[44:]],-1,(0,0,255),5)

    else:
       img = cv2.drawContours(img, [imgpts[4:8]],-1,(0,0,255),3)
    return img

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Create grid board object we're using in our stream
board = aruco.GridBoard_create(
        markersX=1,
        markersY=1,
        markerLength=0.09,
        markerSeparation=0.01,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rotation_vectors, translation_vectors = None, None
axis = np.float32([[-.5,-.5,0], [-.5,.5,0], [.5,.5,0], [.5,-.5,0],
                   [-.5,-.5,1],[-.5,.5,1],[.5,.5,1], [.5,-.5,1],
#[-0.5,-2,-2.5],[-0.5,2,-2.5], [3.5,2,-2.5], [3.5,-2,-2.5], [-0.5,-1.5,0.5],[-0.5,1.5,0.5],[3.5,1.5,0.5],[3.5,-1.5,0.5]
 
   [-2.4151  , -2.0000  , -0.8170],
   [-2.4151  ,  2.0000  , -0.8170],
   [-0.4151  ,  2.0000  , -4.2811],
   [-0.4151  , -2.0000  , -4.2811],
   [ 0.1830  , -1.5000  ,  0.6830],
   [ 0.1830  ,  1.5000  ,  0.6830],
   [ 2.1830  ,  1.5000  , -2.7811],
   [ 2.1830 ,  -1.5000  , -2.7811],


   [ 2.1405  ,  1.2500   ,-0.2075],
   [ 2.3187  ,  1.2330   ,-0.1047],
   [ 2.4920  ,  1.1823   ,-0.0046],
   [ 2.6558  ,  1.0993   , 0.0899],
   [ 2.8054  ,  0.9864   , 0.1764],
   [ 2.9370  ,  0.8466   , 0.2523],
   [ 3.0468  ,  0.6837   , 0.3157],
   [ 3.1319  ,  0.5021   , 0.3648],
   [ 3.1900  ,  0.3069   , 0.3983],
   [ 3.2194  ,  0.1032   , 0.4153],
   [ 3.2194  , -0.1032   , 0.4153],
   [ 3.1900  , -0.3069   , 0.3983],
   [ 3.1319  , -0.5021   , 0.3648],
   [ 3.0468  , -0.6837   , 0.3157],
   [ 2.9370  , -0.8466   , 0.2523],
   [ 2.8054  , -0.9864   , 0.1764],
   [ 2.6558  , -1.0993   , 0.0899],
   [ 2.4920  , -1.1823   ,-0.0046],
   [ 2.3187  , -1.2330   ,-0.1047],
   [ 2.1405  , -1.2500   ,-0.2075],
  
   [-1.5,-1.5,-2.5], [-1.5,1.5,-2.5], [5,1.5,-2.5], [5,-1.5,-2.5],
    [-1.5,-1.5,0],[-1.5,1.5,0],[5,1.5,0], [5,-1.5,0],[0.1,2.5,-1.25], [3,2.5,-1.25],

])

# Make output image fullscreen
cv2.namedWindow('ProjectImage',cv2.WINDOW_NORMAL)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        orig_color_image = color_image.copy()
        gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)


        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        print(ids)
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = gray,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)
        ProjectImage = color_image.copy()
        ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, borderColor=(0, 0, 255))
        if ids is not None and len(ids) > 0:
            # Estimate the posture per each Aruco marker
            rotation_vectors, translation_vectors, _objPoints = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
            print(rotation_vectors.shape)
            for rvec in rotation_vectors:
              rotation_mat = cv2.Rodrigues(rvec[0])[0]
              print(rotation_mat)

            for rvec, tvec,idss in zip(rotation_vectors, translation_vectors,ids):
                
                if len(sys.argv) == 2 and sys.argv[1] == 'cube':
                    try:

                        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)
                        ProjectImage = drawCube(ProjectImage, corners, imgpts,idss)
                    except:
                        continue
                else:    
                    ProjectImage = aruco.drawAxis(ProjectImage, cameraMatrix, distCoeffs, rvec, tvec, 1)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('ProjectImage', ProjectImage)
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', images)
        #cv2.imshow('Found markers2', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

cv2.destroyAllWindows()
