# -*- coding: utf-8 -*-
"""
Submission file for fruit classification challenge.
Final model: ImprovedCNN with depthwise separable convolutions.
Validation accuracy: 72.95%
"""

from challenge_test import Test
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ImprovedCNN(nn.Module):
    """
    Improved 4-layer CNN with depthwise separable convolutions.
    Architecture: 16→24→32→38 filters
    Parameters: 9,986 (under 10k limit)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial conv: 3→16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Depthwise separable: 16→24
        self.conv2_dw = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.conv2_pw = nn.Conv2d(16, 24, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        
        # Standard conv: 24→32
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Depthwise separable: 32→38
        self.conv4_dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False)
        self.conv4_pw = nn.Conv2d(32, 38, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(38)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(38, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        x = self.pool(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn4(self.conv4_pw(self.conv4_dw(x))))
        x = self.pool(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class Trainer(Test):
    
    def __init__(self):
        super(Test, self).__init__()
    
    def create_model(self):
        model = ImprovedCNN(num_classes=10)
        return model
    
    def create_transform(self):
        return transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def save_model(self, model):
        torch.save(model.state_dict(), "model.torch")


if __name__ == "__main__":
    trainer = Trainer()
    model = trainer.create_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ImprovedCNN")
    print(f"Total parameters: {total_params:,}")
    print(f"Status: {'PASS' if total_params <= 10000 else ' FAIL'}")
    
    try:
        checkpoint = torch.load("best_model_v2.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights: val_acc = {checkpoint.get('val_acc', 0):.4f}")
        trainer.save_model(model)
        print(f"Saved to model.torch")
    except FileNotFoundError:
        print("Warning: best_model_v2.pth not found")