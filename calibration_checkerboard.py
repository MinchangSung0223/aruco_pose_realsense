"""
Framework   : OpenCV Aruco
Description : Calibration using checkerboard 
Status      : Running
References  :
    1) https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
    2) https://calib.io/pages/camera-calibration-pattern-generator
"""

import pyrealsense2 as rs
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import glob

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)


# Wait time to show calibration in 'ms'
WAIT_TIME = 100

# termination criteria for iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# generalizable checkerboard dimensions
# https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
cbrow = 6
cbcol = 9


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# IMPORTANT : Object points must be changed to get real physical distance.
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calib_images/*.jpg')
count=0
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


        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret == True:
           objpoints.append(objp)
           corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
           imgpoints.append(corners2)
           color_image = cv2.drawChessboardCorners(color_image, (cbcol, cbrow), corners2,ret)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.imshow('Found markers2', color_image)
        key = cv2.waitKey(1) & 0xFF
        if  key == ord('s'):
           cv2.imwrite("./images/"+str(count)+".jpg",orig_color_image)
           print("Save image "+str(count)+".jpg")
           count = count+1

finally:

    # Stop streaming
    pipeline.stop()

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# ---------- Saving the calibration -----------------
cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)

# note you *release* you don't close() a FileStorage object
cv_file.release()


