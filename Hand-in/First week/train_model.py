import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as v2
import os
from torch import autocast

OUTPUT_DIR = "../../data_augmented/moderate_clean_mix_match"   # folder with your augmented training data
VAL_DATA_DIR = "../../data/val"             # validation folder


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Linear(5*9*9, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = v2.Compose([
        v2.Resize((150, 150)),
        v2.ToTensor(),
        v2.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    train_ds = torchvision.datasets.ImageFolder(root=OUTPUT_DIR, transform=transform)
    val_ds = torchvision.datasets.ImageFolder(root=VAL_DATA_DIR, transform=transform)

    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)


    model = SmallCNN(num_classes=10).to(device)

    # Optionally, load the random weights from your submission
    if os.path.exists("model.torch"):
        model.load_state_dict(torch.load("model.torch"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50  # or 100
    best_val_loss = float('inf')

    # --- Early Stopping Parameters ---
    patience = 10  # number of epochs to wait for improvement
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        running_corrects = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()
        
        train_loss = running_loss / len(train_ds)
        train_acc = running_corrects / len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()
        
        val_loss /= len(val_ds)
        val_acc = val_corrects / len(val_ds)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        
        # --- Save best model and early stopping check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model.torch")
            print("✅ Best model updated!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping triggered after {epoch+1} epochs")
            break
