import cv2, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

yolo = YOLO("yolo26s-pose.pt")
N_FRAMES = 30

from tqdm.auto import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

class ShotDataset(Dataset):
    def __init__(self, model, root, limit=None):
        root = Path(root)
        self.classes = sorted(d.name for d in root.iterdir() if d.is_dir())

        # 1. list every video up front so the bar knows the total
        items = []
        for label_id, cls in enumerate(self.classes):
            paths = [p for p in (root / cls).iterdir() if p.suffix.lower() in VIDEO_EXTS]
            items += [(p, label_id) for p in paths[:limit]]
            print(f"{cls}: {len(paths)} videos found")
        print(f"total videos to process: {len(items)}")

        # 2. process them with one progress bar
        X, y = [], []
        skipped = 0
        bar = tqdm(items, desc="Extracting features", unit="clip")
        for path, label_id in bar:
            bar.set_postfix(cls=self.classes[label_id], kept=len(X), skipped=skipped)

            frames = read_video(str(path), n=N_FRAMES)
            if frames is None:
                skipped += 1
                continue
            try:
                results = model(frames, verbose=False)
                feats = [frame_to_features(r, 0) for r in results]
            except (IndexError, AttributeError):        # nobody detected in some frame
                skipped += 1
                continue
            X.append(np.stack(feats))
            y.append(label_id)

        print(f"kept {len(X)} clips, skipped {skipped}")
        assert len(X) > 0, "No clips loaded: check the folder path and file extensions"

        self.X = torch.tensor(np.stack(X), dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


dataset = ShotDataset(yolo, "/content/drive/MyDrive/training_badminton/VideoBadminton_Dataset")

X_train, X_val, y_train, y_val = train_test_split(
    dataset.X, dataset.y,
    test_size=0.2,
    stratify=dataset.y.numpy(),
    random_state=42,
)

X, y, classes = torch.load("/content/drive/MyDrive/training_badminton/cache.pt")
print(X.shape, y.shape, len(classes), "classes")

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y.numpy(),
    random_state=42,
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

model = ShotClassifier(in_channels=36, num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 50

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for i, (xb, yb) in enumerate(train_loader, start=1):
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Epoch {epoch} | batch {i}/{len(train_loader)} | loss so far (avg) {running_loss / i:.4f}", end="\r")

    print(f"Epoch {epoch} | average loss {running_loss / len(train_loader):.4f}" + " " * 30)

torch.save(model.state_dict(), "shot_classifier.pt")
