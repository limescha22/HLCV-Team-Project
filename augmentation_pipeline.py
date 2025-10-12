import os
import torch, torchvision, torchvision.transforms.v2 as transforms
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Defining paths

TRAIN_DATA_DIR = os.path.join("data/train")
VAL_DATA_DIR = os.path.join("data/val")

CLEAN_TRAIN_DATA_DIR = os.path.join("clean/train")
CLEAN_VAL_DATA_DIR = os.path.join("clean/val")

OUTPUT_BASE = "data_augmented"
os.makedirs(OUTPUT_BASE, exist_ok=True)


sizes = [120, 150, 200]
strengths = ["moderate", "strong", "grayscale"]

# --- AUGMENTATION PIPELINES ---
def get_augmentations(size, strength="moderate", grayscale=False):
    """Return augmentation transform for saving images."""

    augment_list = [
        transforms.Resize((size, size)),
        transforms.RandomRotation(10 if strength=="moderate" else 25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.1) if strength=="moderate" else (0.15, 0.2),
            shear=5 if strength=="moderate" else 15,
            scale=(0.9, 1.1) if strength=="moderate" else (0.8, 1.2)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(
            brightness=0.2 if strength=="moderate" else 0.4,
            contrast=0.2 if strength=="moderate" else 0.4
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.2 if strength=="moderate" else 0.4,
            scale=(0.02, 0.1) if strength=="moderate" else (0.05, 0.2),
            ratio=(0.3, 3.3)
        )
    ]
    if grayscale:
        augment_list.insert(-2, transforms.Grayscale(num_output_channels=3))

    return transforms.Compose(augment_list)


# --- GENERATE AND SAVE AUGMENTED DATASETS ---
def create_augmented_dataset(size, strength, grayscale=False):
    """Creates and saves augmented dataset for one configuration."""
    tag = f"{strength}_{size}"
    if grayscale and "gray" not in tag:
        tag = f"grayscale_{strength}_{size}"

    print(f"\n🔧 Generating dataset: {tag}")

    out_dir = os.path.join(OUTPUT_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)

    transform = get_augmentations(size, strength, grayscale)
    base_dataset = datasets.ImageFolder(TRAIN_DATA_DIR)

    for class_idx, class_name in enumerate(base_dataset.classes):
        class_input_dir = os.path.join(TRAIN_DATA_DIR, class_name)
        class_output_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        image_files = os.listdir(class_input_dir)
        for img_file in tqdm(image_files, desc=f"{class_name} ({tag})"):
            img_path = os.path.join(class_input_dir, img_file)

            # Load and apply augmentation
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                aug_img = transform(img)

            # Save augmented image
            base_name = os.path.splitext(img_file)[0]
            save_path = os.path.join(class_output_dir, f"{base_name}_aug.png")
            save_image(aug_img, save_path)


# --- MAIN LOOP ---
# for size in sizes:
#     create_augmented_dataset(size, strength="moderate", grayscale=False)
#     create_augmented_dataset(size, strength="strong", grayscale=False)
#     create_augmented_dataset(size, strength="moderate", grayscale=True)
#     create_augmented_dataset(size, strength="strong", grayscale=True)
create_augmented_dataset(170, strength="moderate", grayscale=False)# 170 as most images have height around 170. Medium RGB bc that performed best
print("\n✅ All augmented datasets generated and saved!")
