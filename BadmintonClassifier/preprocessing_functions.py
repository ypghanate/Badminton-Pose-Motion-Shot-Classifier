import torch.nn as nn
import cv2
from ultralytics import YOLO
import numpy as np

def frame_to_features(result, player_idx):
  kp = result.keypoints.xy[player_idx].cpu().numpy()
  h, w = result.orig_shape

  hip_centre = (kp[11] + kp[12]) / 2
  shoulder_centre = (kp[5] + kp[6]) / 2
  torso = np.linalg.norm(shoulder_centre - hip_centre) + 1e-6

  pose = ((kp - hip_centre) / torso).flatten()
  position = hip_centre / np.array([w, h])

  return np.concatenate([pose, position])

def read_video(path, n=30):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, n).astype(int)   # 30 evenly spaced frame positions
    return [frames[i] for i in idx]


    


drive.mount('/content/drive')

# 2. Set working directory strictly to your badminton folder
folder_path = '/content/drive/MyDrive/training_badminton'

if os.path.exists(folder_path):
    os.chdir(folder_path)
    print("Working directory set to:", os.getcwd())
    print("Folder contents:", os.listdir('.'))
else:
    print("Folder not found. Please verify the directory name.")