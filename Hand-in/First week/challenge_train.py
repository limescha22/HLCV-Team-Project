# -*- coding: utf-8 -*-
"""
    Challenge Submission
"""

from challenge_test import Test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import torchvision


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        # Input: 3x150x150
        self.features = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1),  # -> 5x150x150
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # -> 5x75x75
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # -> 5x37x37
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # -> 5x18x18
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # -> 5x9x9
        )
        self.classifier = nn.Linear(5 * 9 * 9, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Trainer(Test):

    def __init__(self):
        super(Test, self).__init__()

    def create_model(self):
        """Return the CNN architecture (under 10k params)."""
        model = SmallCNN(num_classes=10)
        num_par = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {num_par} parameters.")
        return model

    def create_transform(self):
        """Return preprocessing steps for inference."""
        return torchvision.transforms.Compose([
            v2.Resize((150, 150)),
            v2.ToTensor(),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def save_model(self, model):
        """Save trained model weights for evaluation."""
        torch.save(model.state_dict(), "model.torch")


# --- Optional Training Example (NOT required by submission)
if __name__ == "__main__":
    trainer = Trainer()
    model = trainer.create_model()
    transform = trainer.create_transform()

    # Save untrained weights to demonstrate submission format
    trainer.save_model(model)
    print("✅ Model definition saved as 'model.torch'.")
