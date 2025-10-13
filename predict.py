import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

from model import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True, help="Path to checkpoint (must contain model_state_dict, classes, img_size)")
parser.add_argument("--input_folder", required=True, help="Folder with images to predict")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

device = torch.device(args.device)
ckpt = torch.load(args.weights, map_location=device)

# Retrieve recorded classes and image size from checkpoint (saved by train.py)
classes = ckpt.get("classes", None)
img_size = ckpt.get("img_size", 80)

# Build model and load weights
model = build_model(num_classes=len(classes) if classes else 10, device=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Preprocessing: resize -> tensor -> normalize (must match training)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

with torch.no_grad():
    for fname in sorted(os.listdir(args.input_folder)):
        path = os.path.join(args.input_folder, fname)
        if not os.path.isfile(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(1).item()
            label = classes[pred] if classes else str(pred)
            print(fname, label)
        except Exception as e:
            print("Could not process", path, ":", e)