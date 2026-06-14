from utils import *


print("-----------------------------------------------------------------")
print("----------------3D RECONSTRUCTION & POSE LOCATION----------------")
print("-----------------------------------------------------------------")
print() 
print() 

# --------------- Loading files & images & matches --------------- 
#region 1: Loading files & images & matches
# =================================================================
# STEP 1: LOADING FILES, IMAGES, & MATCHING KEYPOINTS
# =================================================================
# In this step, we read all the foundation data: actual images, ground-truth 
# camera positions from COLMAP, internal calibration parameters, and matching 
# pixel coordinates across images computed by SuperGlue
# =================================================================
print("-----------------Step 1:  Loading files & images-----------------")
print()
 
#region 1.1: Load ground truth (COLMAP)
# We load the exact real-world 3D coordinates and camera matrices so we can 
# compare our computer vision estimations against the absolute true values later

#-Colmap system with respect to world
colmap_3D = np.loadtxt('Data\\Colmap\\points_3D.txt')

T_ref_gt = np.linalg.inv(np.loadtxt(f'Data\\Colmap\\T_Test_pilar8.jpg.txt'))

#- Inverse of reference transformation matrix
T_ref_gt_inv = np.linalg.inv(T_ref_gt)

cam_indices = [1, 2, 3, 4, 5, 6, 7, 9, 10]
T_cam_gt = {
    f'cam{i}': np.linalg.inv(np.loadtxt(f'Data\\Colmap\\T_Test_pilar{i}.jpg.txt'))
    for i in cam_indices
}

T_cam_gt['old'] = np.linalg.inv(np.loadtxt('Data\\Colmap\\T_Test_pilar0.jpg.txt'))

# Transform the true 3D point cloud so everything is relative to our reference camera
colmap_3D = np.vstack((colmap_3D.T, np.ones((1, colmap_3D.shape[0]))))
colmap_3D = T_ref_gt_inv @ colmap_3D

T_ref_cam_gt = {
    f'T_ref_{key}_gt': T_ref_gt_inv @ mat 
    for key, mat in T_cam_gt.items()
}



#region 1.2: Load images & matches
# We read the digital images and load the .npz packages containing matching features

#-Images
image_ref = cv2.imread('Data\\Images\\cam_ref.jpg')
image_1 = cv2.imread('Data\\Images\\cam_1.jpg')
image_2 = cv2.imread('Data\\Images\\cam_2.jpg')
image_old = cv2.imread('Data\\Images\\cam_old.jpg')

# We resize the old historical photo to match the rest of the pictures (the new oness)
image_old_good = cv2.resize(image_old, (1008, 756)) 

#-Matches
match_ref_1_path = 'Data\\Superglue\\matches_superglue_ref_1.npz'
match_ref_2_path = 'Data\\Superglue\\matches_superglue_ref_2.npz'
match_ref_old_path = 'Data\\Superglue\\matches_superglue_ref_old.npz'

npz_ref_1 = np.load(match_ref_1_path)
npz_ref_2 = np.load(match_ref_2_path)
npz_ref_old = np.load(match_ref_old_path)

npz_ref_1.files
npz_ref_2.files
npz_ref_old.files
#endregion


#region 1.3: Obtain matches 
# SuperGlue tells us which pixels match across images. Here we isolate those matching 
# coordinate pairs into clean arrays so our algorithms can use them directly

#-Matches of REF - CAM 1
SuperGlue_ref_1 = npz_ref_1['matches']
mask_ref_1 = SuperGlue_ref_1 > -1

matches_ref_SfM = npz_ref_1['keypoints0'] [mask_ref_1]
matches_cam1_SfM = npz_ref_1['keypoints1']  [SuperGlue_ref_1[mask_ref_1]]

#-Matches of REF - CAM 2
SuperGlue_ref_2 = npz_ref_2['matches']
mask_ref_2 = SuperGlue_ref_2 > -1

matches_ref_2 = npz_ref_1['keypoints0']  [mask_ref_2]
matches_cam2 = npz_ref_2['keypoints1'][SuperGlue_ref_2[mask_ref_2]]

#-Matches of REF - OLD
SuperGlue_ref_old = npz_ref_old['matches']
mask_ref_old = SuperGlue_ref_old > -1

matches_ref_sg_old = npz_ref_old['keypoints0'][mask_ref_old]
matches_old = npz_ref_old['keypoints1'][SuperGlue_ref_old[mask_ref_old]]

#-Matches of REF - CAM 1 - CAM 2 (2D) / Track matching pixels that appear 
# simultaneously across the Reference, Cam 1, and Cam 2 views
mask_ref_1_2 = (SuperGlue_ref_1 > -1) & (SuperGlue_ref_2 > -1)

idx_2 = SuperGlue_ref_2[mask_ref_1_2]
idx_2_1 = SuperGlue_ref_1[mask_ref_1_2]

matches_ref_PnP = npz_ref_2['keypoints0'][mask_ref_1_2]    
matches_cam1_PnP = npz_ref_1['keypoints1'][idx_2_1]  
matches_cam2_PnP = npz_ref_2['keypoints1'][idx_2]

#endregion


#Load internal calibration of new camera (in this case my phone camera)
K_c = np.loadtxt('Calibration\\K_new_cam.txt')
#endregion
#endregion


# ------------------- SfM to obtain T_ref_cam1 ------------------- 
#region 2: Structure from Motion to obtain T_ref_cam1
# =================================================================
# STEP 2: STRUCTURE FROM MOTION (SfM) FOR CAMERA 1 LOCATION
# =================================================================
# We use geometry to find out where Camera 1 was positioned relative to the 
# Reference Camera, and build our very first rough 3D point cloud from their shared matches
# =================================================================
print("-------------------Step 2: SfM for T_ref_cam1--------------------")
print()

#region 2.1: Fundamental matrix F_ref_cam1

#-Obtain fundamental matrix REF - CAM 1
F_ref_cam1, _ = cv2.findFundamentalMat(matches_ref_SfM, matches_cam1_SfM, method=cv2.FM_RANSAC, ransacReprojThreshold=1,confidence=0.90)
# Find the geometric relationship blueprint between the Ref camera and Camera 1
F_own = np.array([
    [2.0048476917576653e-09, -4.037290620172873e-07, 0.00019301602930457828],
    [5.95285670148014e-07, -1.3268097417828665e-07, 0.0003343069885774621],
    [-0.0003301022062881087, -0.0004330205870516242, 0.055666526044085674]
])

#-Check epipolar lines
#epipole_epipolarLines_click_plot (image_ref, F_ref_cam1, image_1)
#endregion


#region 2.2: SfM
# By combining pixel match angles and our camera matrix, we triangulate and calculate 3D coordinates

#-Obtain pose of CAM 1 and 3D points
t_start = time.time()
T_cam1_ref, points_3D_SfM = SfM(K_c, F_own, matches_ref_SfM, matches_cam1_SfM)
#T_cam1_ref, points_3D_SfM = SfM(K_c, F_ref_cam1, matches_ref_SfM, matches_cam1_SfM)
t_end = time.time()

T_ref_cam1 = np.linalg.inv(T_cam1_ref)

t_SfM = t_end - t_start
#endregion


# Scale our computed model to match COLMAP's physical tracking system metric scale
escala_colmap_ref_cam1 = np.linalg.norm(T_ref_cam_gt['T_ref_cam5_gt'][:3, 3])
escala_pilar_ref_cam1 = np.linalg.norm(T_ref_cam1[:3, 3])
scale_factor = escala_colmap_ref_cam1 / escala_pilar_ref_cam1

#-Visualization
plot_scene_2Cam(points_3D_SfM * scale_factor, colmap_3D, T_ref_cam1 * scale_factor, T_ref_cam_gt['T_ref_cam5_gt'])

#region 2.3: Reprojection errors
# We project our new 3D points back onto the 2D pictures to see how many pixels off they are

#- Reference camera
print("\u25CF Reprojection error reference camera (SfM)")
P_ref = K_c @ np.eye(3,4)
rmse_ref_SfM = visualizar_error_reproyeccion(matches_ref_SfM, points_3D_SfM.T, P_ref, image_ref)

#- Camera 1
print() 
print("\u25CF Reprojection error camera 1 (SfM)")
P_cam1_ref= K_c @ T_cam1_ref[:3, :]
rmse_cam1_SfM = visualizar_error_reproyeccion(matches_cam1_SfM, points_3D_SfM.T, P_cam1_ref, image_1)
print() 
#endregion
#endregion


# -------------------- BA to refine T_ref_cam1 ------------------- 
#region 3: BA to refine T_ref_cam1
# =================================================================
# STEP 3: BUNDLE ADJUSTMENT (BA) TO REFINE CAMERA 1 & 3D POINTS
# =================================================================
# Since the initial math has errors, Bundle Adjustment minimizes errors
# =================================================================
print("----------------Step 3: BA to refine T_ref_cam1------------------")
print() 

#region 3.1: Optimization parameters
# Flatten our parameters into a single long list so the mathematical optimizer can read them

#-Parametrization of T_ref_cam1
t_ref_cam1, theta_ref_cam1 = parametrization_T(T_ref_cam1, polar=False)

#-Optimization list
Op = []
Op = np.concatenate([theta_ref_cam1.flatten(), t_ref_cam1.flatten(), points_3D_SfM[:3, :].flatten()])
#endregion


#region 3.2: BA

#-Preparation
nPoints = points_3D_SfM.shape[1]
 
xData = [matches_ref_SfM.T, matches_cam1_SfM.T]
 
args = (xData, K_c, nPoints)

#-Optimization
t_start = time.time()
OpOptim = scOptim.least_squares(resBundleProjection, Op, args=args, method="trf") 
t_end = time.time()

t_Bundle_SfM = t_end - t_start

#-Optimized parameters
OpOptim = OpOptim.x 

theta_op = OpOptim[:3]  
t_op = OpOptim[3:6]     
points_3D_SfM_op = (OpOptim[6:]).reshape(3, nPoints)

T_ref_cam1_op = ComputeT(theta_op, t_op)
#endregion


#-Change scale to colmap
escala_pilar_ref_cam1 = np.linalg.norm(T_ref_cam1_op[:3, 3])

scale_factor = escala_colmap_ref_cam1 / escala_pilar_ref_cam1

#-Visualization
plot_scene_2Cam(points_3D_SfM_op * scale_factor, colmap_3D, T_ref_cam1_op * scale_factor, T_ref_cam_gt['T_ref_cam5_gt'])


#region 3.3: Reprojection errors post Bundle Adjustment

#- Reference camera
print("\u25CF Reprojection error reference camera (SfM + BA)")
P_ref = K_c @ np.eye(3,4)
rmse_ref_SfM = visualizar_error_reproyeccion(matches_ref_SfM, points_3D_SfM_op.T, P_ref, image_ref,)

#- Camera 1
print() 
print("\u25CF Reprojection error camera 1 (SfM + BA)")
T_cam1_ref_op = np.linalg.inv(T_ref_cam1_op)
P_cam1_ref_op= K_c @ T_cam1_ref_op[:3, :]
rmse_cam1_SfM = visualizar_error_reproyeccion(matches_cam1_SfM, points_3D_SfM_op.T, P_cam1_ref_op, image_1)
print() 
#endregion
#endregion


# ------------------- PnP to obtain T_ref_cam2 ------------------- 
#region 4: PnP to obtain T_ref_cam2
# =================================================================
# STEP 4: PERSPECTIVE-N-POINT (PnP) TO LOCATE CAMERA 2
# =================================================================
# Now that we have a solid 3D point cloud map, we can figure out exactly where 
# Camera 2 is by matching its 2D pixels directly against known 3D points
# =================================================================
print("----------------Step 4: PnP for T_ref_cam2------------------")
print() 

#region 4.1: Obtain common 3D matches
idx_3d = SuperGlue_ref_1[mask_ref_1_2]
mask_3d = np.isin(SuperGlue_ref_1[mask_ref_1], idx_3d)
points_3D_PnP  = points_3D_SfM_op[:, mask_3d]

#-Visualization
plot_3D_points(points_3D_PnP, "Common Triangulated Points for cameras 1 & 2")
#endregion


#region 4.2: Solving PnP
objectPoints =  np.ascontiguousarray(points_3D_PnP.T).reshape((points_3D_PnP.shape[1], 1, 3))

imagePoints = matches_cam2_PnP
imagePoints = np.ascontiguousarray(imagePoints).reshape((imagePoints.shape[0], 1, 2)) 
 
t_start = time.time()
retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, K_c, distCoeffs=None,flags=cv2.SOLVEPNP_EPNP)
t_end = time.time()

t_PnP = t_end - t_start

rvec_flat = np.array(rvec, dtype=float).flatten().reshape(3, 1)
tvec_flat = np.array(tvec, dtype=float).flatten().reshape(3, 1)

T_cam2_ref = ComputeT(rvec_flat, tvec_flat)
T_ref_cam2 = np.linalg.inv(T_cam2_ref)
#endregion


#-Visualization, of the now 3 cameras
plot_scene_3Cam(points_3D_PnP * scale_factor, colmap_3D, T_ref_cam1_op * scale_factor, T_ref_cam_gt['T_ref_cam5_gt'], T_ref_cam2 * scale_factor, T_ref_cam_gt['T_ref_cam9_gt'])

#region 4.3: Reprojection error of camera 2

#- Camera 2
print("\u25CF Reprojection error camera 2 (PnP)")
P_cam2_ref= K_c @ T_cam2_ref[:3, :]
rmse_cam1_SfM = visualizar_error_reproyeccion(matches_cam2_PnP, points_3D_PnP.T, P_cam2_ref, image_2)
print() 
#endregion
#endregion


# ------------------- BA to refine T_ref_cam2 -------------------- 
#region 5: BA to refine T_ref_cam2
print("----------------Step 5: BA to refine T_ref_cam2------------------")
print() 

#region 5.1: Optimization parameters
# =================================================================
# STEP 5: MULTI-CAMERA BUNDLE ADJUSTMENT FOR REF, CAM 1, & CAM 2
# =================================================================
# We perform another adjustment, but this time optimizing all 3 cameras 
# simultaneously 
# =================================================================

#-Parametrization of T_ref_cam1
t_ref_cam1, theta_ref_cam1 = parametrization_T(T_ref_cam1_op, polar=True)

#-Parametrization of T_ref_cam2
t_ref_cam2, theta_ref_cam2 = parametrization_T(T_ref_cam2, polar=False)

#-Optimization list
Op = []

theta = np.array([theta_ref_cam1, theta_ref_cam2])
t = np.concatenate([t_ref_cam1, t_ref_cam2]) 

Op = np.concatenate([points_3D_PnP.flatten(), theta.flatten(), t.flatten()])
#endregion


#region 5.2: BA

#-Preparation
nPoints = points_3D_PnP.shape[1]
 
xData = [matches_ref_PnP.T, matches_cam1_PnP.T, matches_cam2_PnP.T]

nCameras = 3
 
args = (xData, K_c, nPoints, nCameras)

#-Optimization
t_start = time.time()
OpOptim = scOptim.least_squares(resBundleProjectionCameras, Op, args=args, method="trf")   
t_end = time.time()

t_Bundle_PnP = t_end - t_start

#-Optimized parameters
OpOptim = OpOptim.x 

points_3D_PnP_op = OpOptim[0:3*nPoints].reshape((3, nPoints))

theta_ref_cam1_op2 =OpOptim[3*nPoints:3*nPoints+3]
theta_ref_cam2_op2 =OpOptim[3*nPoints+3:3*nPoints+6]
    
tras_ref_cam1_op2 = OpOptim[3*nPoints+6:3*nPoints+8]
tras_ref_cam2_op2 = OpOptim[3*nPoints+8:3*nPoints+11]
    
T_ref_cam1_op2 = ComputeT(theta_ref_cam1_op2, tras_ref_cam1_op2)
T_ref_cam2_op2  = ComputeT(theta_ref_cam2_op2, tras_ref_cam2_op2)
#endregion


#-Change scale to colmap
escala_pilar_ref_cam1 = np.linalg.norm(T_ref_cam1_op2[:3, 3])

scale_factor = escala_colmap_ref_cam1 / escala_pilar_ref_cam1

#-Visualization of the 3 cameras
plot_scene_3Cam(points_3D_PnP_op * scale_factor, colmap_3D, T_ref_cam1_op2 * scale_factor, T_ref_cam_gt['T_ref_cam5_gt'], T_ref_cam2_op2 * scale_factor, T_ref_cam_gt['T_ref_cam9_gt'])


#region 5.3: Reprojection errors

#- Reference camera
print("\u25CF Reprojection error reference camera (PnP + BA)")
P_ref = K_c @ np.eye(3,4)
rmse_ref_SfM = visualizar_error_reproyeccion(matches_ref_PnP, points_3D_PnP_op.T, P_ref, image_ref)

#- Camera 1
print() 
print("\u25CF Reprojection error camera 1 (PnP + BA)")
T_cam1_ref_op2 = np.linalg.inv(T_ref_cam1_op2)
P_cam1_ref_op2= K_c @ T_cam1_ref_op2[:3, :]
rmse_cam1_SfM = visualizar_error_reproyeccion(matches_cam1_PnP, points_3D_PnP_op.T, P_cam1_ref_op2, image_1)

#- Camera 2
print() 
print("\u25CF Reprojection error camera 2 (PnP + BA)")
T_cam2_ref_op2 = np.linalg.inv(T_ref_cam2_op2)
P_cam2_ref_op2= K_c @ T_cam2_ref_op2[:3, :]
rmse_cam1_SfM = visualizar_error_reproyeccion(matches_cam2_PnP, points_3D_PnP_op.T, P_cam2_ref_op2, image_2)
print() 
#endregion
#endregion


# -------------------- DLT to obtain T_ref_old ------------------- 
#region 6: DLT to obtain T_ref_old
print("----------------Step 6: DLR to obtain T_ref_old------------------")
print() 

#region 6.1: Obtain matches 3D
# =================================================================
# STEP 6: DIRECT LINEAR TRANSFORM (DLT) TO LOCATE THE HISTORICAL CAMERA
# =================================================================
# Because the old photo was taken from an old camera we do not know 
# calibration parameters, we use DLT geometry matrices to tie its 
# pixel positions to our current modern 3D map 
# =================================================================

#-Matches of REF - CAM 1 - OLD (2D)
mask_ref_1_old = (SuperGlue_ref_1 > -1) & (SuperGlue_ref_old > -1)

idx_2 = SuperGlue_ref_old[mask_ref_1_old]
idx_2_1 = SuperGlue_ref_1[mask_ref_1_old]

matches_ref_1_DLT = npz_ref_old['keypoints0'][mask_ref_1_old]    
matches_cam1_DLT = npz_ref_1['keypoints1'][idx_2_1]  
matches_old_1_DLT = npz_ref_old['keypoints1'][idx_2]

#-Matches of REF - CAM 2 - OLD (2D)
mask_ref_2_old = (SuperGlue_ref_2 > -1) & (SuperGlue_ref_old > -1)

idx_2 = SuperGlue_ref_old[mask_ref_2_old]
idx_2_1 = SuperGlue_ref_2[mask_ref_2_old]

matches_ref_2_DLT = npz_ref_old['keypoints0'][mask_ref_2_old]    
matches_cam2_DLT = npz_ref_2['keypoints1'][idx_2_1]  
matches_old_2_DLT = npz_ref_old['keypoints1'][idx_2]

# Filter noise using RANSAC so tracking outliers don't warp our matrices
F_ref_cam1, inliers_1 = cv2.findFundamentalMat(matches_ref_1_DLT, matches_cam1_DLT, method=cv2.FM_RANSAC, ransacReprojThreshold=1,confidence=0.90)
F_ref_cam2, inliers_2 = cv2.findFundamentalMat(matches_ref_2_DLT, matches_cam2_DLT, method=cv2.FM_RANSAC, ransacReprojThreshold=1,confidence=0.90)

inliers_mask_1 = inliers_1.flatten()
inliers_mask_2 = inliers_2.flatten()

matches_ref_1_DLT_inliers = matches_ref_1_DLT[inliers_mask_1 == 1]
matches_ref_2_DLT_inliers = matches_ref_2_DLT[inliers_mask_2 == 1]

matches_cam1_DLT_inliers = matches_cam1_DLT[inliers_mask_1 == 1]
matches_cam2_DLT_inliers = matches_cam2_DLT[inliers_mask_2 == 1]

matches_old_1_DLT_inliers = matches_old_1_DLT[inliers_mask_1 == 1]
matches_old_2_DLT_inliers = matches_old_2_DLT[inliers_mask_2 == 1]

#-Keep unique matches with inliers
mask_ref_1_not_in_ref_2_inliers = ~np.isin(matches_ref_1_DLT_inliers, matches_ref_2_DLT_inliers).all(axis=1)

matches_ref_1_DLT_unique = matches_ref_1_DLT_inliers[mask_ref_1_not_in_ref_2_inliers]
matches_cam1_DLT_unique = matches_cam1_DLT_inliers[mask_ref_1_not_in_ref_2_inliers]
matches_old_1_DLT_unique = matches_old_1_DLT_inliers[mask_ref_1_not_in_ref_2_inliers]

#-Keep unique matches not considering inliers
mask_ref_1_not_in_ref_2 = ~np.isin(matches_ref_1_DLT, matches_ref_2_DLT).all(axis=1)

matches_ref_1_DLT = matches_ref_1_DLT[mask_ref_1_not_in_ref_2]
matches_cam1_DLT = matches_cam1_DLT[mask_ref_1_not_in_ref_2]
matches_old_1_DLT = matches_old_1_DLT[mask_ref_1_not_in_ref_2]

#-Triangulation of 3D points with inliers
X_3D_1_DLT_inliers = triangulacion(matches_ref_1_DLT_unique.T, matches_cam1_DLT_unique.T, P_ref, P_cam1_ref_op2)
X_3D_2_DLT_inliers = triangulacion(matches_ref_2_DLT_inliers.T, matches_cam2_DLT_inliers.T, P_ref, P_cam2_ref_op2)

#-Triangulation of 3D points no inliers
X_3D_1_DLT = triangulacion(matches_ref_1_DLT.T, matches_cam1_DLT.T, P_ref, P_cam1_ref_op2)
X_3D_2_DLT = triangulacion(matches_ref_2_DLT.T, matches_cam2_DLT.T, P_ref, P_cam2_ref_op2)

#-Combine unique 2D & 3D points
#combined_2D_points_inlirs = np.concatenate([matches_old_1_DLT_unique, matches_old_2_DLT_inliers], axis=0) #Inliers
combined_2D_points = np.concatenate([matches_old_1_DLT, matches_old_2_DLT], axis=0)
combined_3D_points = np.concatenate([X_3D_1_DLT, X_3D_2_DLT], axis=1)
#endregion


#region 6.2: DLT 

#-DLT

t_start = time.time()
P_old_ref = np.loadtxt('Data\\P_old_ref\\P_old_ref.txt')
best_indices = np.loadtxt('Data\\P_old_ref\\best_indices.txt').astype(int)
t_end = time.time()

t_DLT = t_end - t_start

#-Decompose the raw projection matrix P into isolated rotation and translation views
T_ref_old, K_old = decomposePmatrix(P_old_ref)
T_old_ref = np.linalg.inv(T_ref_old)
#endregion


#-Visualization
plot_scene_4Cam(combined_3D_points * scale_factor, colmap_3D, T_ref_cam1_op2 * scale_factor, T_ref_cam_gt['T_ref_cam5_gt'], T_ref_cam2_op2 * scale_factor, T_ref_cam_gt['T_ref_cam9_gt'], T_ref_old * scale_factor, T_ref_cam_gt['T_ref_old_gt'])

#region 6.3: Reprojection error

#- Camera OLD reprojection
print("\u25CF Reprojection error old camera (DLT)")
rmse_old_DLT = visualizar_error_reproyeccion(combined_2D_points[best_indices, :], (combined_3D_points[:, best_indices]).T, P_old_ref, image_old_good)
print() 
#endregion
#endregion


# -------------------- BA to refine P_ref_old -------------------- 
#region 7: BA to refine P_ref_old
# =================================================================
# STEP 7: BUNDLE ADJUSTMENT FOR THE HISTORICAL CAMERA SPACE
# =================================================================
# We refine the old camera matrix positions specifically so its baseline 
# aligns perfectly with the modern coordinate frame vectors
# =================================================================
print("----------------Step 7: BA to refine P_ref_old------------------")
print() 

#region 7.1: Optimization parameters

#-Optimization list
Op = np.hstack((P_old_ref.flatten()))
#endregion


#region 7.2: BA

#-Preparation
nPoints = combined_2D_points[best_indices, :].shape[1]
 
args = (combined_2D_points[best_indices, :].T, combined_3D_points[:, best_indices], nPoints)

#-Optimization
t_start = time.time()
OpOptim = scOptim.least_squares(resBundleProjectionOldCamera, Op.flatten(), args=args, method="trf") 
t_end = time.time()

t_Bundle_DLT = t_end - t_start

#-Optimized parameters
OpOptim = OpOptim.x 

#K_old_op = OpOptim[:9].reshape(3, 3)
#T_old_ref_op = OpOptim[9:].reshape(4, 4)
#T_ref_old_op = np.linalg.inv(T_old_ref_op)
P_op = OpOptim.reshape(3, 4)
T_ref_old_op, K_old_op = decomposePmatrix(P_op)
T_old_ref_op = np.linalg.inv(T_ref_old_op)

#endregion


#-Visualization
plot_scene_4Cam(combined_3D_points * scale_factor, colmap_3D, T_ref_cam1_op2 * scale_factor, T_ref_cam_gt['T_ref_cam5_gt'], T_ref_cam2_op2 * scale_factor, T_ref_cam_gt['T_ref_cam5_gt'], T_ref_old_op * scale_factor, T_ref_cam_gt['T_ref_old_gt'])


#region 7.3: Reprojection error

#- Camera OLD
print("\u25CF Reprojection error old camera (DLT + BA)")
P_old_ref_op_const= K_old_op @ np.eye(3,4) @ T_old_ref_op
rmse_old_DLT = visualizar_error_reproyeccion(combined_2D_points[best_indices, :], (combined_3D_points[:, best_indices]).T, P_old_ref_op_const, image_old_good)
print() 
#endregion
#endregion


# ------------------------- Detecting ------------------------- 
#region: Detecting changes
# =================================================================
# STEP 8: STRUCTURAL CHANGE DETECTION BETWEEN TIMELINES
# =================================================================
# We project our modern optimized 3D map coordinates onto the historical 
# camera view frame. If a point lands exactly where its feature match was 
# seen, the structure is "Unchanged". If it lands far away, it 
# means that part of the physical environment changed.
# =================================================================

points_points_3D_PnP_op_hom = np.vstack((points_3D_PnP_op, np.ones((1, points_3D_PnP_op.shape[1]))))

X_proj_old = P_old_ref @ points_points_3D_PnP_op_hom
X_proj_old = X_proj_old[:2, :] / X_proj_old[2, :]

#visualizar_puntos(image_old_good, X_proj_old.T)

#endregion


# ------------------------- Final checks ------------------------- 
#region 9: Final checks
# =================================================================
# STEP 9: FINAL CHECKS & QUANTITATIVE ACCURACY EVALUATION
# =================================================================
# We compute numerical metrics to check the errors of our estimations 
# against the absolute ground-truth measurements, then print summaries
# =================================================================
print("----------------Step 8: Final checks------------------")
print() 

#region 9.1:Parametrization

#-Parametrization of transformation matrices
t_ref_cam1, theta_ref_cam1 = parametrization_T(T_ref_cam1_op2 * scale_factor, polar=False)
t_ref_cam2, theta_ref_cam2 = parametrization_T(T_ref_cam2_op2 * scale_factor, polar=False)
t_ref_old, theta_ref_old = parametrization_T(T_ref_old_op * scale_factor, polar=False)

#-Parametrization of ground truth transformation matrices
t_ref_cam1_gt, theta_ref_cam1_gt = parametrization_T(T_ref_cam_gt['T_ref_cam5_gt'], polar=False)
t_ref_cam2_gt, theta_ref_cam2_gt = parametrization_T(T_ref_cam_gt['T_ref_cam9_gt'], polar=False)
t_ref_old_gt, theta_ref_old_gt = parametrization_T(T_ref_cam_gt['T_ref_old_gt'], polar=False)
#endregion


#region 9.2:Rotation error
rot_cam1 = compute_error(theta_ref_cam1, theta_ref_cam1_gt)
rot_cam2 = compute_error(theta_ref_cam2, theta_ref_cam2_gt)
rot_old = compute_error(theta_ref_old, theta_ref_old_gt)
#endregion


#region 9.3: Traslation error
tras_cam1 = compute_error(t_ref_cam1, t_ref_cam1_gt)
tras_cam2 = compute_error(t_ref_cam2, t_ref_cam2_gt)
tras_old = compute_error(t_ref_old, t_ref_old_gt)
#endregion


#region 9.4: Printing results

#-Transformation matrix errors
camera_data = [
    ["Camera 1", rot_cam1, tras_cam1],
    ["Camera 2", rot_cam2, tras_cam2],
    ["Camera old", rot_old, tras_old]
]

headers = ["Camera", "Rotational Error (radians)", "Translational Error (meters)"]

print(tabulate(camera_data, headers=headers, tablefmt="grid"))
print()

#-Transformation matrix errors
time_data = [
    ["SfM", t_SfM],
    ["BA of SfM", t_Bundle_SfM],
    ["PnP", t_PnP],
    ["BA of PnP", t_Bundle_PnP],
    ["DLT", t_DLT],
    ["BA of DLT", t_Bundle_DLT]
]

headers = ["Proccess", "Ex. Time (s)"]

print(tabulate(time_data, headers=headers, tablefmt="grid"))


#endregion
#endregion