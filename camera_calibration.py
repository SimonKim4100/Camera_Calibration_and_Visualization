import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
import calibration_functions as fun
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

import feature

size = (7, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((np.prod(size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

objpoints = [] # real world space
imgpoints = [] # image plane

images = glob.glob('data/*.jpg')
images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, size, None)
    
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (23,23),(-1,-1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, size, corners, ret)

    scale_factor = 0.5
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR) 
    # cv2.imshow(f'img: {idx+1}', resized_img)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# ret, mtx, dist, rvecs, tvecs = fun.custom_calibrate_camera(objpoints, gray.shape[::-1], img_size)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# Camera matrix (intrinsic parameters)
camera_matrix = mtx

# Distortion coefficients
distortion_coefficients = dist

# Extrinsics matrices (rotation + translation for each pose)
extrinsics = []
for rvec, tvec in zip(rvecs, tvecs):
    R, _ = cv2.Rodrigues(rvec)
    extrinsic_matrix = np.hstack((R, tvec))
    extrinsics.append(extrinsic_matrix)

# Convert to NumPy array
extrinsics = np.array(extrinsics)

# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("auto")

cam_width = 0.64/1
cam_height = 0.32/1
scale_focal = 40
# chess board setting
board_width = 7
board_height = 11
square_size = 1
# # display
# # True -> fix board, moving cameras
# # False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()