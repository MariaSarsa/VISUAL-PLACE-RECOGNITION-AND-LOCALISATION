import os
import csv
import cv2
import time
import random
import numpy as np
import scipy as sc
import scipy.io as sio
import scipy.linalg as scAlg
import scipy.optimize as scOptim
import matplotlib.pyplot as plt

from PIL import Image
from scipy.linalg import logm
from scipy.linalg import expm
from tabulate import tabulate

# & -------------------- BASIC PLOTTING -------------------- 
#region BASIC PLOTTING
def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)    
    
def drawRefSystem_norm(ax, T_w_c, strStyle, nameStr):
    """
    Draw a normalized reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
        -input:
            ax: axis handle
            T_w_c: (4x4 matrix) Reference system C seen from W.
            strStyle: lines style.
            nameStr: Name of the reference system.
    """
    # Get the origin and the axis vectors
    origin = T_w_c[0:3, 3]  # The origin (translation)
    x_axis = T_w_c[0:3, 0]  # X axis
    y_axis = T_w_c[0:3, 1]  # Y axis
    z_axis = T_w_c[0:3, 2]  # Z axis

    # Normalize the axis vectors so they are unit vectors
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Draw the axes with length 1 (unit length) from the origin
    draw3DLine(ax, origin, origin + x_axis, strStyle, 'r', 1)  # X axis in red
    draw3DLine(ax, origin, origin + y_axis, strStyle, 'g', 1)  # Y axis in green
    draw3DLine(ax, origin, origin + z_axis, strStyle, 'b', 1)  # Z axis in blue

    # Add a label to the origin
    ax.text(np.squeeze(origin[0] + 0.1), np.squeeze(origin[1] + 0.1), np.squeeze(origin[2] + 0.1), nameStr, color='k', fontsize=12)

#endregion

# & -------------------- PARAMETRIZATION -------------------- 
#region PARAMETRIZATION
def crossMatrixInv(M): 
    x = [M[2, 1].item(), M[0, 2].item(), M[1, 0].item()] 
    return x

def crossMatrix(x): 
    x = np.asarray(x).flatten()
    M = np.array([[0.0, -x[2].item(), x[1].item()], 
                  [x[2].item(), 0.0, -x[0].item()], 
                  [-x[1].item(), x[0].item(), 0.0]], dtype=np.float64)
    return M 

def Obtain_t_polar(t):
    
    t = t / np.linalg.norm(t)
    
    theta_tras = np.arccos(t[2])
    #theta_tras = np.arccos( np.clip(t[2], -1, 1))
    
    phi_tras = np.arctan2(t[1], t[0])
    #phi_tras = np.arcsin(np.clip(t[1], -1, 1)/np.sin(theta_tras))
    
    tras = [theta_tras, phi_tras]
    
    return tras

def parametrization_T(T, polar=False):
    '''
    input
        T: transformation matrix [3, 3]
        
    output
        theta: 3 angles vector [3, 1]
        t: translational [3, 1]
    
    '''
    R = np.array(T[:3, :3])
    t = np.array(T[:3, 3])
    
    if (polar):
        t = Obtain_t_polar(t)

    theta =  np.array(crossMatrixInv(logm(R))) 
    
    return t, theta

def ComputeT(theta, t):
    R = expm(crossMatrix(theta))  
    
    if len(t) == 2:
        theta_tras = t[0]
        phi_tras = t[1]
    
        t = np.array([np.sin(theta_tras)*np.cos(phi_tras),
                      np.sin(theta_tras)*np.sin(phi_tras),
                      np.cos(theta_tras)])
        
    T = np.hstack((R, t.reshape(3, 1))) 
    T = np.vstack((T, np.array([0, 0, 0, 1])))  
    
    return T

def decomposePmatrix(P):
    """
    Decompose the projection matrix
    From slides 3 - Structure from Motion (pg 18)
        input:
            :P: Projection matrix
        output:   
            :T: Transformation matrix
            :K_c: Intrinsic calibration matrix
    """
    
    K_c_, R_cw, t_wc_, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    K_c = K_c_/K_c_[-1, -1]
    R_wc = R_cw.T
    t_wc = (t_wc_[:3] / t_wc_[3]).reshape((3,))
    
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc     
    T_wc[:3, 3] = t_wc

    #T = np.linalg.inv(T_wc)

    return T_wc, K_c
#endregion

# & -------------------- BUNDLE ADJUSTMENT -------------------- 
#region BUNDLE ADJUSTMENT
def resBundleProjection(Op, xData, K_c, nPoints): 
    '''
    -input: 
    Op: Optimization parameters: this must include a 
    parametrization for T_12 (reference 2 seen from reference 1) 
    in a proper way and for X1 (3D points in ref 1) 
    
    xData: containing 2D points of both cameras
        x1Data: (3xnPoints) 2D points on image 1 (homogeneous 
        coordinates) 
        
        x2Data: (3xnPoints) 2D points on image 2 (homogeneous 
        coordinates)
    
    K_c: (3x3) Intrinsic calibration matrix 
    
    nPoints: Number of points -output: 
    
    -output:
    res: residuals from the error between the 2D matched points 
    and the projected points from the 3D points  
    (2 equations/residuals per 2D point)         

    '''
    theta = Op[:3]
    t_12 = Op[3:6]
    
    res = []

    X_3D = np.array(Op[6:]).reshape((3, nPoints))
    X_3D = np.vstack([X_3D, np.ones((1, nPoints))])
    
    # Residual reference camera
    P1 = K_c @ np.eye(3, 4)
    
    X1_p = P1 @ X_3D
    
    X1_p = X1_p / X1_p[2, :]
    
    res1 = (xData[0] - X1_p[:2, :])
    res1 = res1.flatten()
    
    res.append(res1)
    
    # Residual camera 1
    T_12 = ComputeT(theta, t_12)
    
    X2 = K_c @ np.eye(3, 4) @ np.linalg.inv(T_12) @ X_3D

    X2 = X2 / X2[2, :]

    x2Data = xData[1]

    res2 = x2Data[:2, :] - X2[:2, :]
    
    res2 = res2.flatten()  
    
    res.append(res2)
    
    res = np.array(res)

    return res.flatten()

def resBundleProjectionCameras(Op, xData, K_c, nPoints, nCameras): 
    '''
    -input: 
    Op: Optimization parameters: this must include a 
    parametrization for T_21 (reference 1 seen from reference 2) 
    in a proper way and for X1 (3D points in ref 1) 
    
    xData: (3xnPoints) 2D points of cameras
    
    K_c: (3x3) Intrinsic calibration matrix 
    
    nPoints: Number of points -output: 
    
    -output:
    res: residuals from the error between the 2D matched points 
    and the projected points from the 3D points  
    (2 equations/residuals per 2D point)         

    '''
    # Get the 3D points
    X_3D = np.array(Op[0:3*nPoints]).reshape((3, nPoints))
    X_3D = np.vstack([X_3D, np.ones((1, nPoints))])
    
    # Get the projection matrix for camera 1    
    P1 = K_c @ np.eye(3, 4)
    
    X1_p = P1 @ X_3D
    
    X1_p = X1_p / X1_p[2, :]
    
    res = []
    
    res1 = (xData[0] - X1_p[:2, :])    
    res1 = res1.flatten()
    
    res.append(res1)
    
    # Loop for the remaining cameras
    for i in range (nCameras - 1):
        
        indx_theta = 3*nPoints + 3*i
        indx_tras = 3*nPoints + 3*(nCameras - 1) + 3*i
        
        # Get transformation parameters
        theta = (Op[indx_theta:indx_theta + 3]).reshape(3, 1)
        #tras = (Op[indx_tras:indx_tras + 3]).reshape(3, 1)
        
        
        if i == 0: # First camera in polar coordinates
            tras = (Op[indx_tras:indx_tras + 2]).reshape(2, 1)
            
            T_ref_other = ComputeT(np.array(theta), np.array(tras)) 
            
        else: # Remaining cameras in Cartesian coordinates
            indx_tras = 3*nPoints + 3*(nCameras - 1) + 2 + 3*(i - 1)
                
            tras = (Op[indx_tras:indx_tras + 3]).reshape(3, 1)
                
            T_ref_other = ComputeT(np.array(theta), np.array(tras)) 
        
        
        # Calculate projection matrix
        P_other_ref = K_c @ np.eye(3, 4) @ np.linalg.inv(T_ref_other)
                        
        # Project 3D points onto images
        X_other = P_other_ref @ X_3D
        
        # Scale with the homogeneous component
        X_other = X_other / X_other[2, :]
        
        # Calculate residuals
        resC = (xData[i+1] - X_other[:2, :])
        resC = resC.flatten()
        
        res.append(resC)
        

    res = np.array(res)

    return res.flatten()

def resBundleProjectionOldCamera(Op, x_2D, x_3D, nPoints): 
    '''
    -input: 
    Op: Optimization parameters: this is the projection matrix of the old camera
    
    x_2D: 2D points on old camera
    
    x_3D: 3D points matching the 2D points with respect to reference system

    
    nPoints: Number of points -output: 
    
    -output:
    res: residuals from the error between the 2D matched points 
    and the projected points from the 3D points  
    (2 equations/residuals per 2D point)         

    '''
    # Pass the entire P matrix
    P = Op.reshape(3, 4)  # First 9 parameters are K_c
     
    res = []

    if x_3D.shape[0] == 3:
        x_3D = np.vstack([x_3D, np.ones((1, nPoints))])
    
    # Residual old camera
    
    Xp = P @ x_3D 
    
    Xp = Xp / Xp[2, :]
    
    res = (x_2D - Xp[:2, :])
    
    res = res.flatten()
    res = np.array(res)

    return res.flatten()

def resBundleProjectionOldCameraCalibration(Op, x_2D, x_3D, T_old_ref, nPoints): 
    '''
    -input: 
    Op: Optimization parameters: this is the projection matrix of the old camera
    
    x_2D: 2D points on old camera
    
    x_3D: 3D points matching the 2D points with respect to reference system

    
    nPoints: Number of points -output: 
    
    -output:
    res: residuals from the error between the 2D matched points 
    and the projected points from the 3D points  
    (2 equations/residuals per 2D point)         

    '''
    K_c = Op.reshape((3, 3))
    
    P_old_ref = K_c @ np.eye(3,4) @ T_old_ref
    
    res = []

    if x_3D.shape[0] == 3:
        x_3D = np.vstack([x_3D, np.ones((1, nPoints))])
    
    # Residual old camera
    
    Xp = P_old_ref @ x_3D 
    
    Xp = Xp / Xp[2, :]
    
    res = (x_2D - Xp[:2, :])
    
    res = res.flatten()
    res = np.array(res)

    return res.flatten()

def resBundleProjectionOldCameraTransformation(Op, x_2D, x_3D, K_c, nPoints): 
    '''
    -input: 
    Op: Optimization parameters: this is the projection matrix of the old camera
    
    x_2D: 2D points on old camera
    
    x_3D: 3D points matching the 2D points with respect to reference system

    
    nPoints: Number of points -output: 
    
    -output:
    res: residuals from the error between the 2D matched points 
    and the projected points from the 3D points  
    (2 equations/residuals per 2D point)         

    '''
    T_old_ref = Op.reshape((4, 4))
    
    P_old_ref = K_c @ np.eye(3,4) @ T_old_ref
    
    res = []

    if x_3D.shape[0] == 3:
        x_3D = np.vstack([x_3D, np.ones((1, nPoints))])
    
    # Residual old camera
    
    Xp = P_old_ref @ x_3D 
    
    Xp = Xp / Xp[2, :]
    
    res = (x_2D - Xp[:2, :])
    
    res = res.flatten()
    res = np.array(res)

    return res.flatten()
#endregion

# & -------------------- TRIANGULATION & SfM & DLT & F -------------------- 
#region TRIANGULATION & SfM & DLT
def triangulacion(X1, X2, P1, P2):
    """
    2D point projection from two images to triangulate their 3D coordinates.

    Parameters:
    X1 : 2D points (homogeneous) from image 1 ([3, N]).
    X2 : 2D points (homogeneous) from image 2 ([3, N]).
    P1 : Projection matrix of image 1 ([3, 4]).
    P2 : Projection matrix of image 2 ([3, 4]).

    Returns: 3D coordinates obtained through triangulation ([3, N]).
    """
    '''
    V4 = []  # Auxiliary variable to save the last row of V from SVD

    # We go through all the points
    for i in range(X1.shape[1]):
        A = np.eye(4) # Assemble a matrix with 2m equations from m observations
        for j in range(P1.shape[1]):  # Go through the columns of the projection matrices
            vector = np.array([
                P1[2, j] * X1[0, i] - P1[0, j] * X1[2, i],
                P1[2, j] * X1[1, i] - P1[1, j] * X1[2, i],
                P2[2, j] * X2[0, i] - P2[0, j] * X2[2, i],
                P2[2, j] * X2[1, i] - P2[1, j] * X2[2, i]
            ])
            A[:, j] = vector.reshape(4, )

        U, S, V = np.linalg.svd(np.array(A)) 
        v4 = V[-1]  
        V4.append(v4) # 3D coordinates triangulated

    V4 = np.array(V4).T # Change to type array and transpose because append function adds row
    comp_hom = V4[3, :]  # Save homogeneous component to scale
    V4_scaled = V4 / comp_hom[np.newaxis, :] # Scaled 3D points
    '''

    #return V4_scaled
    
    # Number of points
    n = X1.shape[1] 

    Xs = []

    for i in range(0, n, 1):

        # Create matrix A for the system of equations x = P * X
        A = np.array([
            (X1[0][i] * P1[2] - P1[0]),  # First equation of camera 1
            (X1[1][i] * P1[2] - P1[1]),  # Second equation of camera 1
            (X2[0][i] * P2[2] - P2[0]),  # First equation of camera 2
            (X2[1][i] * P2[2] - P2[1])   # Second equation of camera 2
        ])

        # Solve using SVD
        u, s, vh = np.linalg.svd(A)

        # The solution is the last element of Vh
        Xsn = vh[-1, :]
        # Normalize
        Xsn = Xsn / Xsn[3]

        Xs.append(Xsn) 
    
    Xs = np.array(Xs)

    return Xs.T

def SfM (Kc, F, x1, x2):
    '''
    Structure from motion with 8 points
    
    1. Computing 4 possible solutions
    2. Finding best solution (most points in front of the 2 cameras)
    3. Point triangulation
    
    output:
        camera pose 2 wrt camera 1
        3D points
    '''
    
    E = Kc.T @ F @ Kc

    U, S, Vt = np.linalg.svd(E)

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])


    R_mas_90 = U @ W @ Vt
    R_menos_90 = U @ W.T @ Vt

    if np.linalg.det(R_mas_90) < 0:
        R_mas_90 *= -1

    if np.linalg.det(R_menos_90) < 0:
        R_menos_90 *= -1

    t = U[:, 2]
    t = t.reshape(3, 1)

    P1 = Kc @ np.eye(3, 4)

    sol1_T2 = np.hstack((R_mas_90, t))
    sol2_T2 = np.hstack((R_mas_90, -t))
    sol3_T2 = np.hstack((R_menos_90, t))
    sol4_T2 = np.hstack((R_menos_90, -t))
    
    solutions = []

    solutions.append(sol1_T2)
    solutions.append(sol2_T2)
    solutions.append(sol3_T2)
    solutions.append(sol4_T2)
    
    X1 = np.hstack([np.asarray(x1), np.ones((x1.shape[0], 1))])
    X2 = np.hstack([np.asarray(x2), np.ones((x2.shape[0], 1))])
    
    max_points = 0
    best_sol = 0
    for k in range(4):
        
        T2 = np.eye(4)  
        T2[:3, :4] = solutions[k]  
        
        P2 = Kc @ T2[:3, :]
        
        X_3D = triangulacion(X1.T, X2.T, P1,  P2)

        condition1 = (X_3D[2, :] > 0)
        X_cam2 = (np.array(solutions[k])[:3, :3] @ X_3D[:3, :]) + np.array(solutions[k])[:3, 3].reshape(-1, 1)

        # Condition 2: The transformed points are in front of camera 2
        condition2 = (X_cam2[2, :] > 0)
        # Compare the four solutions
        count = np.sum(condition1 & condition2)

        if count > max_points:
            max_points = count
            best_sol = k
            X_3D_solution = X_3D
        #print("Number of points in front of camera 1: ", count_1)
        #print("Number of points in front of camera 2: ", count_2)
        
    #print(best_sol)
    #print("3d points size", X_3D.shape)
    #print("points in front", max_points)
    #print("array size", X1.shape)

    T_solution = np.vstack((np.array(solutions[best_sol]), np.array([0, 0, 0, 1])))
    
    return T_solution, X_3D_solution

def DLT(X_3D, X_2D):
    '''
    Input:
        X_3D [4, nPoints]
        X_2D [2, nPoints] 
        
    Output:
        P [3, 4]
    '''
    
    nPoints = X_3D.shape[1]
    if X_3D.shape [0] == 3:
        X_3D = np.vstack((X_3D, np.ones((1, X_3D.shape[1]))))
    
    A = []
    
    for i in range(nPoints):
        
        X = X_3D[0, i]
        Y = X_3D[1, i]
        Z = X_3D[2, i]
        W = X_3D[3, i]
    
        x = X_2D[0, i]
        y = X_2D[1, i]

        A.append([-X, -Y, -Z, -W, 0, 0, 0, 0,  x*X, x*Y, x*Z, x*W])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -W, y*X, y*Y, y*Z, y*W])


    U, S, Vt = np.linalg.svd(np.array(A))
    
    P = Vt[-1].reshape(3, 4)

    return P

def Ransac_DLT(X_3D, X_2D, threshold = 5):

    nAttemps = 100000
    
    best_P = None
    best_votes = -1
    
    nPoints = X_3D.shape[1]
    if X_3D.shape [0] == 3:
        X_3D = np.vstack((X_3D, np.ones((1, X_3D.shape[1]))))
    
    for i in range(nAttemps):
        if i % 10000 == 0:
            print("Iteration ", i)
        
        random_6 = np.random.randint(0, nPoints, 6)
        
        A = []
        
        inliers_number = 0
        
        for k in random_6:

            X = X_3D[0, k]
            Y = X_3D[1, k]
            Z = X_3D[2, k]
            W = X_3D[3, k]
        
            x = X_2D[0, k]
            y = X_2D[1, k]

            A.append([-X, -Y, -Z, -W, 0, 0, 0, 0,  x*X, x*Y, x*Z, x*W])
            A.append([0, 0, 0, 0, -X, -Y, -Z, -W, y*X, y*Y, y*Z, y*W])
         
        U, S, Vt = np.linalg.svd(np.array(A))
    
        P = Vt[-1].reshape(3, 4) 
        
       
        
        puntos2D_proyectados = P @ X_3D
        puntos2D_proyectados = puntos2D_proyectados[:2, :] / puntos2D_proyectados[2, :]
        
        error_reprojection = X_2D - puntos2D_proyectados
        error_reprojection_norm = np.linalg.norm(error_reprojection, axis=0)
        
        inliers_indices = np.where(error_reprojection_norm < threshold)[0]
        inliers_number = len(inliers_indices)

        if (inliers_number > best_votes):
                
            best_votes = inliers_number
            best_indices = inliers_indices
            
            print("Most votes achieved: ", best_votes)
            best_P = P


    print("Most votes achieved: ", best_votes)
    #print("Inliers indices", matches_indices)
    #- save the inliers
    
        
    return best_P, best_indices

def fundamental_matches(X1, X2):
    '''
    Computes the fundamental matrix F that relates two sets of corresponding points in images 1 and 2.
    
    X1 : 2D array containing the homogeneous coordinates of points in the first image [3, N].
    X2 : 2D array containing the homogeneous coordinates of corresponding points in the second image [3, N].
    '''
    Correspondences = []
    for i in range (X1.shape[1]): 
        x1 = X1[:, i].reshape(3, 1) # Vector [3,1]
        x2 = X2[:, i].reshape(1, 3) # Vector [1,3]

        Aux = x1 @ x2 # Auxiliary matrix to obtain the matrix multiplication between a match

        vector = Aux.flatten(order='F')  # Flatten matrix to vector using Fortran order (column by column)
        
        Correspondences.append(vector)
    

    Correspondences= np.array(Correspondences) 
    Correspondences = np.array(Correspondences, dtype=np.float64)


    U, S, V = np.linalg.svd(Correspondences) # Decomposition SVD to obtain solution from last row of matrix V
    v4 = V[-1] 

    F = v4.reshape(3,3)
    
    return F

def norm_F_pix (image1, image2, x1, x2):
    width_1, height_1 = image1.shape[:2]
    width_2, height_2 = image2.shape[:2]

    T_1 = np.array([
        [1/width_1, 0, -1/2],
        [0, 1/height_1, -1/2],
        [0, 0, 1]
        ])
    T_2 = np.array([
        [1/width_2, 0, -1/2],
        [0, 1/height_2, -1/2],
        [0, 0, 1]
        ])
    
    x1_norm = T_1 @ x1   
    x2_norm = T_2 @ x2
    
    F_norm_1 = fundamental_matches(x1_norm, x2_norm)
    
    U, S, V = np.linalg.svd(F_norm_1)
            
    S[2]=0
            
    F_norm = U @ np.diag(S) @ V

    F = T_2.T @ F_norm @ T_1
    
    
    return F

def ransac_F (matches1, matches2, image1, image2,threshold = 1):
    
    nAttemps = 100000
    
    best_F = None
    best_votes = -1
    
    if (matches1.shape)[0] == 2:
         np.vstack((matches1, np.ones(matches1.shape[1])))
    if (matches2.shape)[0] == 2:
         np.vstack((matches2, np.ones(matches2.shape[1])))
    
    for i in range(nAttemps):
        inliers = 0
        random_8 = np.random.randint(0, (matches1.shape)[1], 8)
        
        matches1_sample = matches1[:, random_8]  # Selected points from matches1
        matches2_sample = matches2[:, random_8]
  
        F = norm_F_pix(image1, image2, matches1_sample, matches2_sample)
        
        epipolar_lines_x2_est = F @ matches1
        epipolar_lines_x2_est = np.array(epipolar_lines_x2_est, dtype=np.float64)

              
        denominator = np.sqrt(epipolar_lines_x2_est[0]**2 + epipolar_lines_x2_est[1]**2)

        # Calculate the epipolar error
        error = np.abs(epipolar_lines_x2_est[2] + epipolar_lines_x2_est[0] * matches2[0, :] + epipolar_lines_x2_est[1] * matches2[1, :]) / denominator
                  
        inliers_indices = np.where(error < threshold)[0]
        inliers = len(inliers_indices)

        if (inliers > best_votes):
                
            best_votes = inliers
            best_F = F
            
    print("Most votes achieved: ", best_votes)

    return best_F

#endregion

# & ----------- PLOTTING SCENE FOR DIFFERENT nº CAMERAS ----------- 
#region PLOTTING SCENE FOR DIFFERENT nº CAMERAS
def plot_3D_points(X, title):
    """
    Function to plot points in 3D.

    x: list or array of x-coordinates of the points.
    y: list or array of y-coordinates of the points.
    z: list or array of z-coordinates of the points.
    """
    x = X[0, :]
    y = X[1, :]
    z = X[2, :]
    
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points in 3D
    ax.scatter(x, y, z, c='r', marker='o')  # You can change 'r' to another color
    
    # Axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title(title)
    
    # Show the plot
    plt.show()

def plot_3D_points_sets(X1, X2):
    """
    Function to plot two sets of points in 3D.
    
    - X1: First set of points, 3xN array (N 3D points).
    - X2: Second set of points, 3xN array (N 3D points).
    """
    # Get the coordinates of the points for both sets
    x1, y1, z1 = X1[0, :], X1[1, :], X1[2, :]
    x2, y2, z2 = X2[0, :], X2[1, :], X2[2, :]
    
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the first set of points in 3D (red)
    ax.scatter(x1, y1, z1, c='r', marker='o', label='Changed', s=20)
    
    # Plot the second set of points in 3D (green)
    ax.scatter(x2, y2, z2, c='g', marker='o', label='Unchanged', s=20)
    
    # Axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

def plot_3D_points_cambios(X, x_same):
    """
    Function to plot points in 3D.

    x: list or array of x-coordinates of the points.
    y: list or array of y-coordinates of the points.
    z: list or array of z-coordinates of the points.
    """
    x = X[0, :]
    y = X[1, :]
    z = X[2, :]
    
    x_s = x_same[0, :]
    y_s = x_same[1, :]
    z_s = x_same[2, :]
    
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points in 3D
    ax.scatter(x, y, z, c='r', marker='o')  # You can change 'r' to another color
    ax.scatter(x_s, y_s, z_s, c='g', marker='o')
    
    # Axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Show the plot
    plt.show()

def plot_3D_points_color(X, highlight_indices=None):
    """
    Function to plot points in 3D and highlight specific points with a different color.

    - X: np.array of 3D coordinates with shape (3, N).
    - highlight_indices: List of indices for the points that should be highlighted (optional).
    """
    x = X[0, :]
    y = X[1, :]
    z = X[2, :]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initial colors: red for all
    colors = ['r'] * X.shape[1]

    # Change color to blue for the highlighted indices
    if highlight_indices is not None:
        for idx in highlight_indices:
            if 0 <= idx < len(colors):  # Ensure the index is valid
                colors[idx] = 'b'

    # Plot the 3D points with assigned colors
    for i in range(X.shape[1]):
        ax.scatter(x[i], y[i], z[i], c=colors[i], marker='o')

    # Axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Show the plot
    plt.show()

def plot_scene_2Cam(points, points_gt, T_ref_cam1, T_ref_cam1_gt):
    '''
     Plots 3D triangulated points along with world and two camera reference systems.
     Parameters:
        points : 3D points [3, N].
        points_gt : 3D points ground truth [3, N].
        T_ref_cam1 : Transformation matrix of camera 1 wrt ref [3, 3].
        T_ref_cam1_gt : Transformation ground truth matrix of camera 1 wrt ref [3, 3].

    Returns: 3D plot.
    '''
    
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4), '-', 'REF')
    
    drawRefSystem(ax, T_ref_cam1, '-', f'CAM 1')
    drawRefSystem(ax, T_ref_cam1_gt, '-', f'GT 1')

    ax.scatter(points[0, :], points[1, :], points[2, :], c='r', marker='.', label='3D Points')
    ax.scatter(points_gt[0, :], points_gt[1, :], points_gt[2, :], c='b', marker='.', label='Points GT')

    xFakeBoundingBox = np.linspace(-10, 10, 2)
    yFakeBoundingBox = np.linspace(-10, 10, 2)
    zFakeBoundingBox = np.linspace(-6, 22, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    # Set the limits of the plot based on the fake bounding box
    ax.set_xlim([min(xFakeBoundingBox), max(xFakeBoundingBox)])
    ax.set_ylim([min(yFakeBoundingBox), max(yFakeBoundingBox)])
    ax.set_zlim([min(zFakeBoundingBox), max(zFakeBoundingBox)])

    # Set the ticks for each axis based on the bounding box range
    ax.set_xticks(np.arange(min(xFakeBoundingBox), max(xFakeBoundingBox) + 1, 2))
    ax.set_yticks(np.arange(min(yFakeBoundingBox), max(yFakeBoundingBox) + 1, 2))
    ax.set_zticks(np.arange(min(zFakeBoundingBox), max(zFakeBoundingBox) + 1, 2))
    
    
    ax.legend()
    plt.title('SfM')
    plt.show()
    
def plot_scene_3Cam(points, points_gt, T_ref_cam1, T_ref_cam1_gt, T_ref_cam2, T_ref_cam2_gt):
    '''
     Plots 3D triangulated points along with world and two camera reference systems.
     Parameters:
        points : 3D points [3, N].
        points_gt : 3D points ground truth [3, N].
        T_ref_cam1 : Transformation matrix of camera 1 wrt ref [3, 3].
        T_ref_cam1_gt : Transformation ground truth matrix of camera 1 wrt ref [3, 3].

    Returns: 3D plot.
    '''
    
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4), '-', 'REF')
    
    drawRefSystem(ax, T_ref_cam1, '-', f'CAM 1')
    drawRefSystem(ax, T_ref_cam1_gt, '-', f'GT 1')
    
    drawRefSystem(ax, T_ref_cam2, '-', f'CAM 2')
    drawRefSystem(ax, T_ref_cam2_gt, '-', f'GT 2')

    ax.scatter(points[0, :], points[1, :], points[2, :], c='r', marker='.', label='3D Points')
    ax.scatter(points_gt[0, :], points_gt[1, :], points_gt[2, :], c='b', marker='.', label='Points GT')


    xFakeBoundingBox = np.linspace(-10, 10, 2)
    yFakeBoundingBox = np.linspace(-10, 10, 2)
    zFakeBoundingBox = np.linspace(-6, 22, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    # Set the limits of the plot based on the fake bounding box
    ax.set_xlim([min(xFakeBoundingBox), max(xFakeBoundingBox)])
    ax.set_ylim([min(yFakeBoundingBox), max(yFakeBoundingBox)])
    ax.set_zlim([min(zFakeBoundingBox), max(zFakeBoundingBox)])

    # Set the ticks for each axis based on the bounding box range
    ax.set_xticks(np.arange(min(xFakeBoundingBox), max(xFakeBoundingBox) + 1, 2))
    ax.set_yticks(np.arange(min(yFakeBoundingBox), max(yFakeBoundingBox) + 1, 2))
    ax.set_zticks(np.arange(min(zFakeBoundingBox), max(zFakeBoundingBox) + 1, 2))
    
    ax.legend()
    plt.title('PnP')
    plt.show()
    
def plot_scene_4Cam(points, points_gt, T_ref_cam1, T_ref_cam1_gt, T_ref_cam2, T_ref_cam2_gt, T_ref_old, T_ref_old_gt):
    '''
     Plots 3D triangulated points along with world and two camera reference systems.
     Parameters:
        points : 3D points [3, N].
        points_gt : 3D points ground truth [3, N].
        T_ref_cam1 : Transformation matrix of camera 1 wrt ref [3, 3].
        T_ref_cam1_gt : Transformation ground truth matrix of camera 1 wrt ref [3, 3].

    Returns: 3D plot.
    '''
    
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4), '-', 'REF')
    
    drawRefSystem(ax, T_ref_cam1, '-', f'CAM 1')
    drawRefSystem(ax, T_ref_cam1_gt, '-', f'GT 1')
    
    drawRefSystem(ax, T_ref_cam2, '-', f'CAM 2')
    drawRefSystem(ax, T_ref_cam2_gt, '-', f'GT 2')
    
    drawRefSystem(ax, T_ref_old, '-', f'OLD')
    drawRefSystem(ax, T_ref_old_gt, '-', f'GT old')

    ax.scatter(points[0, :], points[1, :], points[2, :], c='r', marker='.', label='3D Points')
    ax.scatter(points_gt[0, :], points_gt[1, :], points_gt[2, :], c='b', marker='.', label='Points GT')

    xFakeBoundingBox = np.linspace(-15, 25, 2)
    yFakeBoundingBox = np.linspace(-5, 25, 2)
    zFakeBoundingBox = np.linspace(-5, 25, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    # Set the limits of the plot based on the fake bounding box
    ax.set_xlim([min(xFakeBoundingBox), max(xFakeBoundingBox)])
    ax.set_ylim([min(yFakeBoundingBox), max(yFakeBoundingBox)])
    ax.set_zlim([min(zFakeBoundingBox), max(zFakeBoundingBox)])

    # Set the ticks for each axis based on the bounding box range
    ax.set_xticks(np.arange(min(xFakeBoundingBox), max(xFakeBoundingBox) + 1, 25))
    ax.set_yticks(np.arange(min(yFakeBoundingBox), max(yFakeBoundingBox) + 1, 25))
    ax.set_zticks(np.arange(min(zFakeBoundingBox), max(zFakeBoundingBox) + 1, 25))
    
    ax.legend()
    plt.title('DLT')
    plt.show()
#endregion
    
# & -------------------- CHECKING CORRECTNESS -------------------- 
#region CHECKING CORRECTNESS
def epipole_epipolarLines_click_plot (image1, F, image2):
    """
    Plots epipolar lines on two images based on user-clicked points.
        User clicks 5 points on 'image1'.
        Corresponding epipolar lines are calculated using the fundamental matrix 'F'.
        The epipole is determined via SVD of F.
        Epipolar lines and the epipole are plotted on 'image2'.
    """
    
    figure_1_id = 1
    plt.figure(figure_1_id)
    plt.imshow(image1)
    plt.title('Image 1 - Click a point (5 in total)')

    coord_clicked_points = []
    epipole_lines = []

    # Loop for all points
    for i in range(5):

        coord_clicked_point = plt.ginput(1, show_clicks=False)
        
        # Extract coordinates from clicked points and add homogeneous coordinate
        p_clicked = np.append(coord_clicked_point[0], 1)
        coord_clicked_points.append(p_clicked)
        
        ''' Epipolar line formula --> l = F @ x '''
        ep_line = F @ p_clicked 
        epipole_lines.append(ep_line)
        
        # Mark clicked point
        plt.plot(coord_clicked_point[0][0], coord_clicked_point[0][1], marker='x', color='red')
        plt.draw()


    epipole_lines = np.array(epipole_lines)
    
    ''' Epipole determination --> F @ e = 0 --> SVD '''
    #U, S, V_true = np.linalg.svd(F_test.T)
    U, S, V = np.linalg.svd(F.T)
    #epipole_true = V_true[-1] # Solution is last row of V from decomposition
    epipole = V[-1] 
    
    #comp_hom_true = epipole_true[-1]
    #epipole_true = epipole_true/ comp_hom_true

    comp_hom = epipole[-1]
    epipole = epipole / comp_hom # Scaled epipole


    # Plot epipolar lines and epipole
    figure_2_id = 2
    plt.figure(figure_2_id)

    plt.imshow(image2, zorder=0) 

    #plt.plot(epipole_true[0], epipole_true[1], marker='+', color='green')
    plt.plot(epipole[0], epipole[1], marker='+', color='blue')

    for i in range (5):
        
        drawLine(epipole_lines[i], 'r-', 2)  # 'r-' is the red line
        
    plt.show()

def visualizar_error_reproyeccion(puntos_2D_observados, puntos_3D, P, imagen, bool):
    """
    Visualizes the reprojection error of 3D points to 2D onto an image.
    
    Parameters:
        puntos_2D_observados: numpy array of size (N, 2), observed 2D points on the image.
        puntos_3D: numpy array of size (N, 3) or (N, 4), corresponding 3D points in space.
        P: numpy array of size (3, 4), camera projection matrix.
        imagen: Image where the points will be projected.
    """
    puntos_3D_homogeneos = puntos_3D
    
    # Convert 3D points to homogeneous coordinates
    if puntos_3D.shape[1] == 3:
        puntos_3D_homogeneos = np.hstack((puntos_3D, np.ones((puntos_3D.shape[0], 1))))
    
    # Project 3D points to 2D using the projection matrix
    puntos_2D_proyectados_homogeneos = (P @ puntos_3D_homogeneos.T).T
    puntos_2D_proyectados = puntos_2D_proyectados_homogeneos[:, :2] / puntos_2D_proyectados_homogeneos[:, 2, np.newaxis]
    
    # Calculate the reprojection error
    errores = np.linalg.norm(puntos_2D_observados - puntos_2D_proyectados, axis=1)
    
    # Visualize points on the image
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.scatter(puntos_2D_observados[:, 0], puntos_2D_observados[:, 1], color='green', label='Observed points')
    plt.scatter(puntos_2D_proyectados[:, 0], puntos_2D_proyectados[:, 1], color='red', label='Reprojected points')
    
    # Draw lines between observed and reprojected points
    for i in range(len(puntos_2D_observados)):
        plt.plot(
            [puntos_2D_observados[i, 0], puntos_2D_proyectados[i, 0]],
            [puntos_2D_observados[i, 1], puntos_2D_proyectados[i, 1]],
            color='blue', linewidth=1, alpha=0.7, label='Error' if i == 0 else ""
        )
    
    # Plot configuration
    plt.title('Reprojection error')
    plt.legend()
    if bool:
        plt.show()
    
    # Display error statistics
    #print("Mean reprojection error:", np.mean(errores))
    #print("Maximum reprojection error:", np.max(errores))
    #print("Minimum reprojection error:", np.min(errores))
    
    RMSE = np.sqrt(np.mean(errores**2))
    print("Reprojection RMSE:", RMSE)  
    
    return RMSE

def compute_error(vec_gt, vec_est):
    """
    Compute the angular error between two 3D rotation vectors.
    
    Args:
        vec_gt: Ground truth  (3,).
        vec_est: Estimated (3,).
        
    Returns:
        rotation_error: Angular error in radians.
    """
    # Compute the difference between the rotation vectors
    diff_vector = vec_gt - vec_est
    
    # Compute the magnitude of the difference vector (angular error in radians)
    rotation_error = np.linalg.norm(diff_vector)
    
    return rotation_error

def visualizar_matches(image_ref, image_2, matches_ref_final, matches_2_final):
    """
    Visualizes the matching points between two images combined into a single image.
    
    - input:
        image_ref: Reference image (Image 1).
        image_2: Image to compare (Image 2).
        matches_ref_final: List of corresponding points in the reference image.
        matches_2_final: List of corresponding points in image 2.
    
    - output:
        Displays a window with the combined images and matching points connected by lines.
    """
    # Combine images
    height = max(image_ref.shape[0], image_2.shape[0])
    width = image_ref.shape[1] + image_2.shape[1]
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    combined_image[:image_ref.shape[0], :image_ref.shape[1]] = image_ref
    combined_image[:image_2.shape[0], image_ref.shape[1]:] = image_2

    offset_x = image_ref.shape[1]

    # Draw match lines
    for point1, point2 in zip(matches_ref_final, matches_2_final):
        x1, y1 = map(int, point1)
        x2, y2 = map(int, point2)
        x2 += offset_x
        # Draw green line between matching points
        cv2.line(combined_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        # Draw red and blue circles on the points
        cv2.circle(combined_image, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(combined_image, (x2, y2), radius=5, color=(255, 0, 0), thickness=-1)
        
    scale = 0.8
    combined_image = cv2.resize(combined_image, (0, 0), fx=scale, fy=scale)

    # Display the combined image with the matching points
    cv2.imshow('Matches Ref & Cam', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualizar_puntos(image, points_set):
    """
    Visualizes a set of points on an image.
    
    - input:
        image: Image on which the points will be drawn.
        points_set: List of points to draw on the image.
    
    - output:
        Displays a window with the image and the drawn points.
    """
    # Create a copy of the image to avoid modifying it directly
    image_with_points = image.copy()

    # Draw the points on the image
    for point in points_set:
        x, y = map(int, point)
        # Draw a circle on each point
        cv2.circle(image_with_points, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red
    
    # Scale the image for better visualization (optional)
    scale = 0.8
    image_with_points = cv2.resize(image_with_points, (0, 0), fx=scale, fy=scale)

    # Display the image with the points
    cv2.imshow('Points', image_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualizar_matches_numbered(image_ref, image_2, matches_ref_final, matches_2_final):
    """
    Visualizes the matching points between two images combined into a single image with numbered indices.
    
    - input:
        image_ref: Reference image (Image 1).
        image_2: Image to compare (Image 2).
        matches_ref_final: List of corresponding points in the reference image.
        matches_2_final: List of corresponding points in image 2.
    
    - output:
        Displays a window with the combined images and matching points connected by lines and numbered indices.
    """
    # Combine images
    height = max(image_ref.shape[0], image_2.shape[0])
    width = image_ref.shape[1] + image_2.shape[1]
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    combined_image[:image_ref.shape[0], :image_ref.shape[1]] = image_ref
    combined_image[:image_2.shape[0], image_ref.shape[1]:] = image_2

    offset_x = image_ref.shape[1]

    # Draw match lines with numbered indices
    for idx, (point1, point2) in enumerate(zip(matches_ref_final, matches_2_final)):
        x1, y1 = map(int, point1)
        x2, y2 = map(int, point2)
        x2 += offset_x

        # Draw green line between matching points
        cv2.line(combined_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        
        # Draw red and blue circles on the points
        cv2.circle(combined_image, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(combined_image, (x2, y2), radius=5, color=(255, 0, 0), thickness=-1)
        
        # Add numerical indices to the points
        cv2.putText(combined_image, str(idx), (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combined_image, str(idx), (x2 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Scale the combined image
    scale = 0.8
    combined_image = cv2.resize(combined_image, (0, 0), fx=scale, fy=scale)

    # Display the combined image with the matching points
    cv2.imshow('Matches Ref & Cam', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#endregion