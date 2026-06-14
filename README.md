# VISUAL-PLACE-RECOGNITION-AND-LOCALISATION
A Computer Vision pipeline capable of taking multi-view imagery, reconstructing sparse 3D point cloud structures, estimating exact camera positions across multiple modern perspectives, tracking camera poses from historical photos.

This project implements an incremental Structure from Motion (SfM) computer vision pipeline designed to solve a challenging spatial and historical alignment problem. 
In particular its main objective is to determine where a picture (the one used was taken around 150 years ago) was taken from.

## Technical Workflow and Architecture

The system executes sequentially through the following pipeline phases:

* **1. Initialization:** The script loads camera intrinsic calibrations ($K$ matrix), matching feature point coordinates pre-computed via SuperGlue, and ground-truth validation matrices from COLMAP.
* **2. Base Structure Estimation (SfM):** Feature point matches tracked between the *Reference Camera* and *Camera 1* are used to compute the Fundamental Matrix. By back-projecting these matching tracking lines, the system triangulates the initial raw 3D scene structure.
* **3. Core Workspace Bundle Adjustment:** To eliminate minor drift from algebraic triangulation, initial 3D positions and Camera 1's extrinsic pose are refined via non-linear least squares optimization (**Bundle Adjustment**), minimizing global pixel reprojection error.
* **4. Camera 2 Tracking (PnP):** The script isolates common features visible simultaneously across the *Reference Camera, Camera 1, and Camera 2*. Mapping these known optimized 3D coordinates to Camera 2’s 2D image plane allows the **Perspective-n-Point (PnP)** algorithm to localize Camera 2 within the workspace.
* **5. Multi-Camera Optimization Pass:** A global Bundle Adjustment runs simultaneously across all three modern camera frames to bundle their shared constraints.
* **6. Cross-Timeline Localization (DLT):** An unaligned historical view is introduced. By pairing its features with our modern structural map, the **Direct Linear Transform (DLT)** algorithm extracts the historical camera's raw projection matrix, which is then specifically refined through its own adjustment loop.


## Workspace

├── Data/
│   ├── Colmap/         # Validation transformation coordinates
│   ├── Images/         # Workspace camera source images (Ref, Cam1, Cam2, Old)
│   ├── Superglue/      # Feature descriptor packages (.npz arrays)
│   └── P_old_ref/      # Historical camera baseline configuration configurations
├── Calibration/
│   └── K_new_cam.txt   # Camera intrinsic sensor property matrix
├── main.py             # Main execution file
└── utils.py            # Computer vision and plotting helper functions
