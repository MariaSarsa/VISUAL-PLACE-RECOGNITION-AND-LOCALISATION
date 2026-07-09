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


## Visualization

<table>
  <tbody>
    <tr>
      <td width="65%" rowspan="3" align="center" valign="middle">
        <img alt="Modern photos location" src="https://github.com/user-attachments/assets/ae046caf-879c-4bf7-86ed-0de908665ed0" width="100%" />
        <br>
      </td>
      <td width="35%" align="center">
        <img alt="A" src="https://github.com/user-attachments/assets/32610cc8-c5bb-4f44-9886-bb66ce319306" width="65%" />
        <br>
      </td>
    </tr>
    <tr>
      <td width="35%" align="center">
        <img alt="B" src="https://github.com/user-attachments/assets/7a87dd50-56b0-4bb7-a73a-b78bc4ab0b79" width="65%" />
        <br>
      </td>
    </tr>
    <tr>
      <td width="35%" align="center">
        <img alt="C" src="https://github.com/user-attachments/assets/91abd1b3-6c02-48d6-9caf-bdbe0fa01d0d" width="65%" />
        <br>
      </td>
    </tr>
  </tbody>
</table>

<table>
  <tr>
    <td align="center">
      <img alt="Superglue matches between modern photos" src="https://github.com/user-attachments/assets/69d40267-1c2b-46c8-9d44-e9dc786eba37" width="60%"/>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img alt="Structure from Motion (8 points): estimating camera 1 pose" src="https://github.com/user-attachments/assets/e09fd028-aaf9-4937-b033-f6f6f813067e"
 width="60%"/>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img alt="Adding a third camera" src="https://github.com/user-attachments/assets/17f450ce-7668-45f2-a137-39173f652cee" width="60%"/>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img alt="PnP: estimating camera 2 pose" src="https://github.com/user-attachments/assets/668cd8d4-2e74-43db-bde1-7653bc558508" width="60%"/>
    </td>
  </tr>
</table>


## Workspace

The project files are organized into the following structure:

* ** Data/** 
  * ** Colmap/** — # Validation transformation coordinates
  * ** Images/** — # Workspace camera source images (Ref, Cam1, Cam2, Old)
  * ** Superglue/** —   # Feature descriptor packages (.npz arrays)
  * ** P_old_ref/** — # Historical camera baseline configuration configurations
* ** Calibration/** 
  * ** K_new_cam.txt** — # Camera intrinsic sensor property matrix
* ** main.py** —  # Main execution file
* ** utils.py** — # Computer vision and plotting helper functions 

