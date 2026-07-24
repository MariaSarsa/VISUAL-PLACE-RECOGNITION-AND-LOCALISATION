# VISUAL-PLACE-RECOGNITION-AND-LOCALISATION
A Computer Vision pipeline capable of taking multi-view imagery, reconstructing sparse 3D point cloud structures, estimating exact camera positions across multiple modern perspectives, tracking camera poses from historical photos.

This project implements an incremental Structure from Motion (SfM) computer vision pipeline designed to solve a challenging spatial and historical alignment problem. 
In particular its main objective is to determine where a picture (the one used was taken around 150 years ago) was taken from.

<table>
  <tr>
    <td align="center">
      <img alt="Historical Query Image" src="https://github.com/user-attachments/assets/f20c1dfe-4367-4d54-898c-5ab8bc1e2015" width="60%"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <sub><b>Figure:</b> Pilar basilica picture dated before 1866.<br>
      <b>Source / Archive:</b> Obtained from <a href="https://commons.wikimedia.org/w/index.php?curid=24970366" target="_blank">Wikimedia Commons</a>.</sub>
    </td>
  </tr>
</table>

## Technical Workflow and Architecture

The system executes sequentially through the following pipeline phases:

* **1. Initialization:** The script loads camera intrinsic calibrations ($K$ matrix), matching feature point coordinates pre-computed via SuperGlue, and ground-truth validation matrices from COLMAP.
* **2. Base Structure Estimation (SfM):** Feature point matches tracked between the *Reference Camera* and *Camera 1* are used to compute the Fundamental Matrix. By back-projecting these matching tracking lines, the system triangulates the initial raw 3D scene structure.
* **3. Core Workspace Bundle Adjustment:** To eliminate minor drift from algebraic triangulation, initial 3D positions and Camera 1's extrinsic pose are refined via non-linear least squares optimization (**Bundle Adjustment**), minimizing global pixel reprojection error.
* **4. Camera 2 Tracking (PnP):** The script isolates common features visible simultaneously across the *Reference Camera, Camera 1, and Camera 2*. Mapping these known optimized 3D coordinates to Camera 2’s 2D image plane allows the **Perspective-n-Point (PnP)** algorithm to localize Camera 2 within the workspace.
* **5. Multi-Camera Optimization Pass:** A global Bundle Adjustment runs simultaneously across all three modern camera frames to bundle their shared constraints.
* **6. Cross-Timeline Localization (DLT):** An unaligned historical view is introduced. By pairing its features with our modern structural map, the **Direct Linear Transform (DLT)** algorithm extracts the historical camera's raw projection matrix, which is then specifically refined through its own adjustment loop.

## Workspace

The project files are organized into the following structure:

```text
├── Data/                
│   └── Colmap/  # Validation transformation coordinates
│   └── Images/ # Workspace camera source images (Ref, Cam1, Cam2, Old)
│   └── Superglue/ # Feature descriptor packages (.npz arrays)
│   └── P_old_ref/ # Historical camera baseline configuration
├── Calibration/  
│   └── K_new_cam.txt # Camera intrinsics matrix
├── main.py # Main execution file
└── utils.py # Computer vision and plotting helper functions 
```

## Visualization

This section shows the step-by-step visual outputs of the pipeline, from initial feature matching to the final localized historical picture.

### 1. Modern Reference Setup
Here we have the 3 modern photos of the Pilar basilica that are goint to be used to perfom the visual place recognition and localization. We will progresively incorporate them into our system. The order in which we include them is relevant, in particular we want to start with the most similar photo to the old one, which in this case is the orange one, that we will call REF (reference).

In the following Figure we can see where each of the photos was taken from, so that we have parallax between them.
<table>
  <tbody>
    <tr>
      <td width="65%" rowspan="3" align="center" valign="middle">
        <img alt="Modern photos location" src="https://github.com/user-attachments/assets/ae046caf-879c-4bf7-86ed-0de908665ed0" width="100%" />
      </td>
      <td width="35%" align="center">
        <img alt="A" src="https://github.com/user-attachments/assets/32610cc8-c5bb-4f44-9886-bb66ce319306" width="65%" />
      </td>
    </tr>
    <tr>
      <td width="35%" align="center">
        <img alt="B" src="https://github.com/user-attachments/assets/7a87dd50-56b0-4bb7-a73a-b78bc4ab0b79" width="65%" />
      </td>
    </tr>
    <tr>
      <td width="35%" align="center">
        <img alt="C" src="https://github.com/user-attachments/assets/91abd1b3-6c02-48d6-9caf-bdbe0fa01d0d" width="65%" />
      </td>
    </tr>
  </tbody>
</table>

---

### 2. Feature Extraction and Matching (SuperGlue)
We use SuperGlue to compute highly reliable and dense feature matches across the modern photos. In particular, we obtain matches between out REF viewpoint and the other two cameras.

<table>
  <tr>
    <td align="center">
      <img alt="Superglue matches between modern photos" src="https://github.com/user-attachments/assets/69d40267-1c2b-46c8-9d44-e9dc786eba37" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 3. Camera 1 Localization (SfM)
Using the 8-point algorithm on the tracked features between the REF camera and Camera 1 (blue), we estimate the relative pose of Camera 1 (respect to REF camera) and back-project the rays (triangulation) to form our first raw 3D point cloud.

<table>
  <tr>
    <td align="center">
      <img alt="Structure from Motion (8 points): estimating camera 1 pose" src="https://github.com/user-attachments/assets/e09fd028-aaf9-4937-b033-f6f6f813067e" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 4. Multi-View Feature Tracking
Next, we isolate features observed across all three camera systems to prepare for the estimation of the pose of Camera 2 (yellow).

<table>
  <tr>
    <td align="center">
      <img alt="Adding a third camera" src="https://github.com/user-attachments/assets/17f450ce-7668-45f2-a137-39173f652cee" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 5. Camera 2 Localization (PnP)
By projecting the known optimized 3D coordinates from our baseline onto the 2D image plane of Camera 2, the Perspective-n-Point (PnP) algorithm can localize Camera 2 in the space.

<table>
  <tr>
    <td align="center">
      <img alt="PnP: estimating camera 2 pose" src="https://github.com/user-attachments/assets/668cd8d4-2e74-43db-bde1-7653bc558508" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 6. Historical Query Matching
NOw, the historical image (pink) is introduced into the pipeline. We compute feature correspondences between the historical landmarks and our modern 3D map using robust RANSAC filtering to keep only consistent geometric inliers.

<table>
  <tr>
    <td align="center">
      <img alt="Adding the old camera" src="https://github.com/user-attachments/assets/f3fb0b27-c3a5-487f-a2c6-21dc45437100" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 7. Old Camera Localization (DLT)
Using the filtered 2D-3D correspondences, the Direct Linear Transform (DLT) calculates the projection matrix of the historical camera.

<table>
  <tr>
    <td align="center">
      <img alt="DLT: estimating camera pose old" src="https://github.com/user-attachments/assets/2eb54dc7-dd88-45ae-92ed-bc12175afe25" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 8. Physical Metric Scaling
So far, our model is up-to-sclae, because pure SfM is scale-invariant. So we calculate a scaling factor ($\alpha$) by correlating pixel distances in the 3D reconstruction with real physical distances measured in meters (using Google Earth's measurement tools as ground truth).

<table>
  <tr>
    <td align="center">
      <img alt="Real scale" src="https://github.com/user-attachments/assets/37df0716-ccc3-46eb-ba73-3c3f6af6407c" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 9. 3D Reconstruction
Here we can find a 3D reconstructions of the Pilar basilica generated with COLMAP, showing both the camera-tracking sparse cloud and the final dense surface model.

<table>
  <tr>
    <td align="center">
      <img alt="Reconstructions" src="https://github.com/user-attachments/assets/671849d9-adaf-4d84-a634-36b49f7722ec" width="60%"/>
      <br>
    </td>
  </tr>
</table>

---

### 10. Pipeline Metrics and Error Performance Tables
Below are the quantitative results of the execution passes. After every stage, a Bundle Adjustment is run to reduce the root-mean-square error (RMSE) of the pixel reprojection. 
<table>
  <tr>
    <td align="center">
      <img alt="Results table" src="https://github.com/user-attachments/assets/bd1c328d-09f4-4198-92b3-a0267644b664" width="60%"/>
      <br>
    </td>
  </tr>
</table>

The final table illustrates the physical error of our localized camera positions when validated against ground truth data (Colmap).

<table>
  <tr>
    <td align="center">
      <img alt="Cameras error table" src="https://github.com/user-attachments/assets/934bac70-87e1-4fb5-a659-7b6922c4e544" width="60%"/>
      <br>
    </td>
  </tr>
</table>



