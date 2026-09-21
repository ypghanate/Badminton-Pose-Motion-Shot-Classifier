
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=36, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_features, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)


class ShotClassifier(nn.Module):
    def __init__(self, in_channels=36, num_classes=9):
        super().__init__()
        self.extractor = FeatureExtractor(in_channels)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.extractor(x))