<div align="center">

# 🏸 Badminton Spatio-Temporal Inference Pipeline

  <p>
    An end-to-end computer vision engine that tracks player kinematics, analyzes pose trajectories, and classifies shot types in real-time broadcast footage.
  </p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/your-username/your-repo-name/stargazers">
      <img src="https://img.shields.io/github/stars/your-username/your-repo-name?style=for-the-badge&color=8A2BE2" alt="Stars" />
    </a>
    <a href="https://github.com/your-username/your-repo-name/network/members">
      <img src="https://img.shields.io/github/forks/your-username/your-repo-name?style=for-the-badge&color=8A2BE2" alt="Forks" />
    </a>
    <a href="https://github.com/your-username/your-repo-name/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/your-username/your-repo-name?style=for-the-badge&color=8A2BE2" alt="License" />
    </a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
    <img src="https://img.shields.io/badge/YOLOv8-000000?style=for-the-badge&logo=ultralytics&logoColor=white" alt="YOLO" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </p>

  <br />

  <!-- Demo Banner -->
  <img src="docs/demo.gif" alt="Pipeline Demo" width="85%" style="border-radius: 10px;" />

</div>

---

## 📌 Overview

**Badminton Spatio-Temporal Inference Pipeline** is a computer vision system designed to automatically classify badminton stroke types directly from video footage. 

Standard video classification models often struggle with complex athletic movements due to visual noise such as court lines, lighting shifts, and varied player apparel. This pipeline addresses these challenges by isolating human kinematics using **YOLO-Pose 26** keypoint extraction. By computing joint displacements relative to the player's hip center, the model evaluates movement trajectories independently of absolute court position.

The extracted spatial-temporal representations pass through a custom classification head to predict shot categories (smash, drop, clear, drive, lift) and export fully annotated video clips with real-time class predictions.

---

## ⚡ Key Highlights & Architecture
