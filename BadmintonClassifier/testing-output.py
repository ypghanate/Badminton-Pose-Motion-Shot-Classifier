import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load pre-extracted features and class labels from cache
X, y, classes = torch.load("/content/drive/MyDrive/training_badminton/cache.pt")
print(f"Loaded dataset from cache: X={X.shape}, y={y.shape}, {len(classes)} classes")

# 2. Split dataset for validation check
_, X_val, _, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y.numpy(),
    random_state=42,
)

val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# 3. Instantiate model architecture and load saved trained weights
model = ShotClassifier(in_channels=36, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("shot_classifier.pt", map_location=device))
model.eval()

# 4. Predict on validation split
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        out = model(xb.to(device))
        all_preds.append(out.argmax(dim=1).cpu())
        all_true.append(yb)

y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_true).numpy()

# 5. Print Evaluation Metrics
avg_method = 'weighted'

print(f"Accuracy:     {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision:    {precision_score(y_true, y_pred, average=avg_method):.3f}")
print(f"Recall:       {recall_score(y_true, y_pred, average=avg_method):.3f}")
print(f"F1 Score:     {f1_score(y_true, y_pred, average=avg_method):.3f}\n")

# Use loaded classes directly or override if needed
class_labels = classes if isinstance(classes, list) else list(classes)

print("--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 6. Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title('Badminton Shot Classification - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

from google.colab import files
uploaded = files.upload()
path = next(iter(uploaded))

import numpy as np, torch, cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, classes = torch.load("/content/drive/MyDrive/training_badminton/cache.pt")
print(classes)

model = ShotClassifier(in_channels=36, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("shot_classifier.pt", map_location=device))
model.eval()


def predict(path):
  frames = read_video(path, n=30)

  results = yolo(frames, verbose=False)
  feats = [frame_to_features(r, 0) for r in results]

  x = torch.tensor(np.stack(feats), dtype=torch.float32)
  x = x.T.unsqueeze(0).to(device)

  model.eval()
  with torch.no_grad():
      output = model(x)
      probs = torch.softmax(output, dim=1)[0].cpu()

  best = probs.argmax().item()
  ans, conf = classes[best], probs[best].item()
  if (ans == '05_Drop Shot'):
    ans = 'Net Shot'
  elif (ans == '07_Transitional Slice'):
    ans = 'Drop Shot'
  return ans

print(predict(path))

import cv2, subprocess

def annotate_video(path, ans, out_path="annotated.mp4", y_frac=0.12):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter("raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, w / 900, max(2, w // 500)
    (tw, th), base = cv2.getTextSize(ans, font, scale, thick)
    while tw > 0.9 * w:                                   # shrink long labels so they fit
        scale *= 0.9
        (tw, th), base = cv2.getTextSize(ans, font, scale, thick)

    x = (w - tw) // 2                                     # centred horizontally
    y = int(h * y_frac) + th                              # text baseline, a bit down from the top
    pad = th // 2

    while True:
      ok, frame = cap.read()
      if not ok:
          break
      # black outline first (thicker), then white text on top
      cv2.putText(frame, ans, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
      cv2.putText(frame, ans, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
      writer.write(frame)

    cap.release()
    writer.release()

    subprocess.run(["ffmpeg", "-y", "-i", "raw.mp4", "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", out_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

from IPython.display import Video

ans = predict(path)
out = annotate_video(path, ans, "annotated.mp4")
Video(out, embed=True, width=640)

from google.colab import files
files.download(out)