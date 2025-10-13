import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    """
    Tiny CNN with configurable channel widths.
    - channels: tuple (c1, c2, c3) used for conv layers.
    - uses AdaptiveAvgPool + small FC to keep params low.
    """
    def __init__(self, num_classes=10, channels=(8,16,32), dropout=0.2):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def build_model(num_classes=10, device='cpu', channels=(8,16,32), dropout=0.2):
    """
    Build and return the TinyCNN moved to `device`.
    Accepts channels and dropout so callers (train.py) can change capacity.
    """
    model = TinyCNN(num_classes=num_classes, channels=channels, dropout=dropout)
    return model.to(device)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable