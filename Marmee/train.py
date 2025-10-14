"""
Train + diagnostics script.
- Trains TinyCNN (configurable channels up to ~10k params)
- Saves best checkpoint
- After training, loads best checkpoint and computes:
    - confusion matrix heatmap
    - per-class accuracy bar chart
    - confidence histogram
    - loss & accuracy plots
- Requirements: scikit-learn, seaborn
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from model import build_model, count_params

# ----------------- Config (edit as needed) -----------------
cfg = {
    "img_size": 150,
    "batch_size": 32,
    "lr": 5e-3,
    "weight_decay": 1e-5,
    "epochs": 100,
    "seed": 42,
    "dropout": 0.2,
    "train_dir": "train",
    "val_dir": "val",
    "save_path": "tiny_cnn_best.pth",
    "plots_path": "train_plots.png",
    "aug_examples_path": "aug_examples.png",
    "history_path": "train_history.json",
    "conf_matrix_path": "confusion_matrix.png",
    "per_class_acc_path": "per_class_accuracy.png",
    "confidence_hist_path": "confidence_hist.png",
    "config_out": "train_config.json",
    # model capacity: tune channels until params <= 10000
    # default below is a good starting point; you can change to (8,16,32) etc.
    "channels": (14, 23, 30),
    # Whether to use OneCycleLR with SGD (set False to use Adam+StepLR)
    "use_onecycle": False,
    # OneCycleLR peak LR (only used if use_onecycle=True)
    "onecycle_max_lr": 0.03,
    # Whether to enable MixUp training (False by default)
    "use_mixup": False,
    "mixup_alpha": 0.4
}

# ----------------- Reproducibility -----------------
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
torch.cuda.manual_seed_all(cfg["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------- Transforms -----------------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(cfg["img_size"], scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
])

val_tf = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ----------------- Data -----------------
train_ds = datasets.ImageFolder(cfg["train_dir"], transform=train_tf)
val_ds = datasets.ImageFolder(cfg["val_dir"], transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

print("Classes:", train_ds.classes)
print("Train samples:", len(train_ds), "Val samples:", len(val_ds))

# ----------------- Save config -----------------
Path(cfg["config_out"]).write_text(json.dumps(cfg, indent=2))

# ----------------- Visualize a few augmented samples -----------------
try:
    xb, yb = next(iter(train_loader))
    grid = utils.make_grid(xb[:16], nrow=4, padding=2)
    ndarr = (grid.numpy().transpose(1, 2, 0) * 0.5) + 0.5
    ndarr = np.clip(ndarr, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(ndarr)
    plt.axis('off')
    plt.title("Augmented training samples")
    plt.savefig(cfg["aug_examples_path"], bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved augmented examples to", cfg["aug_examples_path"])
except Exception as e:
    print("Could not save augmented examples:", e)

# ----------------- Build model -----------------
model = build_model(num_classes=len(train_ds.classes), device=device,
                    channels=tuple(cfg["channels"]), dropout=cfg["dropout"])
total_params, trainable_params = count_params(model)
print(f"Model channels: {cfg['channels']}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
if trainable_params > 10000:
    print("WARNING: trainable params exceed 10k budget. Adjust cfg['channels'].")

# ----------------- Criterion, optimizer, scheduler -----------------
criterion = nn.CrossEntropyLoss()

if cfg["use_onecycle"]:
    # use SGD + OneCycleLR (step per batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["onecycle_max_lr"],
                                                    steps_per_epoch=steps_per_epoch, epochs=cfg["epochs"],
                                                    pct_start=0.3, anneal_strategy='cos')
    onecycle_mode = True
    print("Using OneCycleLR (SGD). Max LR:", cfg["onecycle_max_lr"])
else:
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    onecycle_mode = False
    print("Using Adam + StepLR")

# MixUp helper (optional)
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ----------------- Training loop -----------------
best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, cfg["epochs"] + 1):
    # --- train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        if cfg["use_mixup"]:
            inputs, targets_a, targets_b, lam = mixup_data(xb, yb, alpha=cfg["mixup_alpha"])
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            preds = outputs.argmax(1)
            # approximate correct count for mixed labels (not exact)
            correct += (lam * (preds == targets_a).sum().item() + (1-lam) * (preds == targets_b).sum().item())
        else:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            preds = outputs.argmax(1)
            correct += (preds == yb).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if onecycle_mode:
            scheduler.step()  # step per batch for OneCycle

        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # --- eval on val set ---
    model.eval()
    running_loss, correct, vtotal = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            vtotal += xb.size(0)
    val_loss = running_loss / vtotal
    val_acc = correct / vtotal

    # step scheduler per-epoch if not OneCycle
    if not onecycle_mode:
        scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_ds.classes,
            "img_size": cfg["img_size"],
            "channels": cfg["channels"],
            "dropout": cfg["dropout"],
        }, cfg["save_path"])
        print("  Saved best model:", cfg["save_path"])

# ----------------- Save history & plots -----------------
with open(cfg["history_path"], "w") as f:
    json.dump({"cfg": cfg, "history": history, "best_val_acc": best_val_acc}, f, indent=2)
print("Saved history to", cfg["history_path"])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")

plt.tight_layout()
plt.savefig(cfg["plots_path"], dpi=150)
plt.close()
print("Saved training plots to", cfg["plots_path"])

# ----------------- Diagnostics using the BEST checkpoint -----------------
if os.path.isfile(cfg["save_path"]):
    print("Loading best checkpoint for diagnostics:", cfg["save_path"])
    ckpt = torch.load(cfg["save_path"], map_location=device)
    best_channels = tuple(ckpt.get("channels", cfg["channels"]))
    best_img_size = ckpt.get("img_size", cfg["img_size"])
    # build model with same channels and load weights
    best_model = build_model(num_classes=len(train_ds.classes), device=device, channels=best_channels, dropout=ckpt.get("dropout", cfg["dropout"]))
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model.eval()

    # prepare val loader with correct resize (in case img_size changed)
    val_transform_diag = transforms.Compose([
        transforms.Resize((best_img_size, best_img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_ds_diag = datasets.ImageFolder(cfg["val_dir"], transform=val_transform_diag)
    val_loader_diag = DataLoader(val_ds_diag, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for xb, yb in val_loader_diag:
            xb = xb.to(device)
            logits = best_model(xb)
            probs = softmax(logits).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_ds.classes, yticklabels=train_ds.classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (best model)")
    plt.tight_layout()
    plt.savefig(cfg["conf_matrix_path"], dpi=150)
    plt.close()
    print("Saved confusion matrix to", cfg["conf_matrix_path"])

    # per-class accuracy
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    plt.figure(figsize=(10, 4))
    plt.bar(train_ds.classes, per_class_acc)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Per-class accuracy"); plt.title("Per-class Accuracy (best model)")
    plt.tight_layout()
    plt.savefig(cfg["per_class_acc_path"], dpi=150)
    plt.close()
    print("Saved per-class accuracy to", cfg["per_class_acc_path"])

    # confidence histogram
    pred_probs = all_probs[np.arange(len(all_preds)), all_preds]
    correct_mask = (all_preds == all_labels)
    plt.figure(figsize=(8, 4))
    plt.hist(pred_probs[correct_mask], bins=20, alpha=0.6, label='correct')
    plt.hist(pred_probs[~correct_mask], bins=20, alpha=0.6, label='incorrect')
    plt.xlabel("Predicted probability (confidence)"); plt.ylabel("Count")
    plt.legend(); plt.title("Confidence histogram (best model)")
    plt.tight_layout()
    plt.savefig(cfg["confidence_hist_path"], dpi=150)
    plt.close()
    print("Saved confidence histogram to", cfg["confidence_hist_path"])

    # classification report json
    report = classification_report(all_labels, all_preds, target_names=train_ds.classes, output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved classification_report.json")

else:
    print("No checkpoint found at", cfg["save_path"], "— skipping diagnostics.")

print("Training & diagnostics complete. Best validation accuracy:", best_val_acc)