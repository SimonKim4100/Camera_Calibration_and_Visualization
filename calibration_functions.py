import numpy as np
import cv2
from scipy.optimize import minimize

def custom_calibrate_camera(objpoints, imgpoints, image_size):

    assert len(objpoints) == len(imgpoints), "Number of object points and image points must match."
    
    # Number of images
    num_images = len(objpoints)
    
    # Homographies
    homographies = []
    for objp, imgp in zip(objpoints, imgpoints):
        H, _ = cv2.findHomography(objp[:, :2], imgp)
        homographies.append(H)
    
    # Intrinsic Parameters
    def compute_v(H, i, j):
        return np.array([
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j]
        ])
    
    V = []
    for H in homographies:
        V.append(compute_v(H, 0, 1))  # v_01
        V.append(compute_v(H, 0, 0) - compute_v(H, 1, 1))  # v_00 - v_11
    V = np.array(V)
    
    # Solve V * b = 0 using SVD
    _, _, VT = np.linalg.svd(V)
    b = VT[-1, :]
    
    # Extract intrinsic parameters from b
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_
    
    camera_matrix = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
    
    # Extrinsic Parameters
    rotation_vectors = []
    translation_vectors = []
    
    for H in homographies:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        lambda_ = 1 / np.linalg.norm(np.dot(np.linalg.inv(camera_matrix), h1))
        r1 = lambda_ * np.dot(np.linalg.inv(camera_matrix), h1)
        r2 = lambda_ * np.dot(np.linalg.inv(camera_matrix), h2)
        t = lambda_ * np.dot(np.linalg.inv(camera_matrix), h3)
        r3 = np.cross(r1, r2)
        R = np.column_stack((r1, r2, r3))
        rotation_vectors.append(cv2.Rodrigues(R)[0])
        translation_vectors.append(t)
    
    # Distortion Coefficients
    distortion_coefficients = np.zeros((5,), dtype=np.float64)

    # Nonlinear Refinement
    def refine_parameters(params):
        fx, fy, cx, cy = params[:4]
        k1, k2, p1, p2, k3 = params[4:9]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        D = np.array([k1, k2, p1, p2, k3])
        
        total_error = 0
        for i in range(num_images):
            objp = objpoints[i]
            imgp = imgpoints[i]
            
            # Project points with updated parameters
            projected_points, _ = cv2.projectPoints(objp, rotation_vectors[i], translation_vectors[i], K, D)
            
            # Calculate reprojection error
            total_error += np.linalg.norm(imgp - projected_points.squeeze(), axis=1).mean()
        
        return total_error / num_images
    
    # Initial guess for optimization
    initial_guess = np.hstack([
        camera_matrix[0, 0],  # fx
        camera_matrix[1, 1],  # fy
        camera_matrix[0, 2],  # cx
        camera_matrix[1, 2],  # cy
        distortion_coefficients  # k1, k2, p1, p2, k3
    ])
    
    # Optimize
    result = minimize(refine_parameters, initial_guess, method='L-BFGS-B')
    optimized_params = result.x
    
    # Extract refined parameters
    fx, fy, cx, cy = optimized_params[:4]
    k1, k2, p1, p2, k3 = optimized_params[4:9]
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    distortion_coefficients = np.array([k1, k2, p1, p2, k3])
    
    # Reprojection Error
    final_error = refine_parameters(optimized_params)
    
    return final_error, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors