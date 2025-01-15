# Camera Calibration and 3D Visualization
Referenced from OpenCV: https://github.com/opencv/opencv

## Files
1. camera_calibration: Main file
2. calibration_functions: Functions for calibration from scratch, without using cv2.calibrateCamera()
3. camera_calibration_show_extrinsics: Modified file of original(from OpenCV repo), to fit our case
4. feature: Calculation functions, supplementary, not necessary

## Features
1. Manual calibration: You may check this by uncommenting:
```python
ret, mtx, dist, rvecs, tvecs = fun.custom_calibrate_camera(objpoints, gray.shape[::-1], img_size)
```
2. Visualize moving cameras: From the following line:
```python
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)
```
   Keep the final parameter(patternCentric) as True. <br>
  3. Visualize moving boards: Change `patternCentric` to False

## Disclaimers
1. Code comes from CV2 and was modified a bit, original code does not come from me
2. There may be some experimental codes here and there, please change them to fit your purposes
