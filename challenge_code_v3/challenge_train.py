# -*- coding: utf-8 -*-
"""
    Training script with checkpointing and early stopping.
    Uses separate train/val directories.
"""

from challenge_test import Test
import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
import os


# -----------------------------
#  CNN Definition
# -----------------------------
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(6, 7, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(7, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(8, 7, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(7*9*9, 14),  # hidden layer with 16 neurons
            torch.nn.ReLU(),              # activation
            torch.nn.Linear(14, 10)  # output layer
        )


    def forward(self, x):
        return self.classifier(self.features(x))


# -----------------------------
#  Trainer Definition
# -----------------------------
class Trainer(Test):
    def __init__(self):
        super().__init__()

    def create_model(self):
        return SimpleCNN(num_classes=10)

    def create_transform(self):
        return v2.Compose([
            v2.RandomResizedCrop((150,150), scale=(0.7, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(0.3),
            v2.RandomRotation(20),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
            v2.ToTensor(),
            v2.Normalize([0.5]*3, [0.5]*3),
            v2.Resize((150, 150)),
            v2.ToImage(),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            v2.ToDtype(torch.float32, scale=True),  # scale 1./255
        ])

    def save_model(self, model):
        torch.save(model.state_dict(), "model.torch")

    # -----------------------------
    # Training Loop with Checkpoint + EarlyStopping
    # -----------------------------
    def train_model(
        self,
        train_dir="../data_submission/baseline",
        val_dir="../data/val",
        batch_size=100,
        epochs=100,
        lr=1e-3,
        patience=10,
        checkpoint_path="best_model.torch",
        device=None
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        transform = self.create_transform()

        # Load datasets
        train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

        # Initialize model, loss, optimizer
        model = self.create_model().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Print number of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized with {num_params} trainable parameters.")

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            train_loss, correct, total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_loss /= total

            # --- Validation ---
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * imgs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            val_loss /= val_total

            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
            )

            # --- Checkpoint ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"✅ Saved new best model: val_acc={best_val_acc:.3f}")
            else:
                epochs_no_improve += 1

            # --- Early stopping ---
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping after {epoch+1} epochs. Best val_acc={best_val_acc:.3f}")
                break

        # --- Load best model & save final weights ---
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.save_model(model)
        print("✅ Training complete. Best model saved as 'model.torch'.")


# -----------------------------
#  Main Entry Point
# -----------------------------
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_model(
        train_dir="../data_augmented/moderate_clean_mix_match", # moderate_clean_mix_matchmoderate_small_no_gray_mix_match
        val_dir="../data/val",
        batch_size=150,
        epochs=100,
        lr=1e-3,
        patience=10
    )
