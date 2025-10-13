import numpy as np
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

import random

# Defining paths

TRAIN_DATA_DIR = os.path.join("data/train")
VAL_DATA_DIR = os.path.join("data/val")

CLEAN_TRAIN_DATA_DIR = os.path.join("clean/train")
CLEAN_VAL_DATA_DIR = os.path.join("clean/val")

OUTPUT_BASE = "data_augmented"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# --- CONFIG ---
IMG_SIZE = (150, 150)
OUTPUT_BASE = "data_augmented"
SINGLE_DIR = os.path.join(OUTPUT_BASE, "single_various_augmentation")
MULTI_DIR = os.path.join(OUTPUT_BASE, "multiple_various_augmentation")

# make sure base directories exist
os.makedirs(SINGLE_DIR, exist_ok=True)
os.makedirs(MULTI_DIR, exist_ok=True)

# --- Moderate augmentation parameters ---
MODERATE_PARAMS = {
    "rotation": (-30, 30),              # degrees
    "translation": (-0.2, 0.2),         # fraction of image size
    "scaling": (0.9, 1.1),              # zoom in/out
    "shear": (-15, 15),                 # degrees
    "h_flip": 0.5,                       # probability
    "v_flip": 0.3,                       # probability
    "brightness": (0.8, 1.2),            # only RGB
    "contrast": (0.8, 1.2),              # only RGB
    "noise_std": (0.01, 0.05),           # Gaussian noise
    "blur_radius": (0, 1)                # Gaussian blur
}

def augment_image(img, apply_prob=0.6, grayscale_prob=0.5):
    """Apply moderate, probabilistic augmentations on a single image.
    img: PIL Image
    apply_prob: probability to apply each augmentation
    grayscale_prob: probability to convert to grayscale
    """
    img_aug = img.copy()
    
    # Decide grayscale
    is_grayscale = random.random() < grayscale_prob
    if is_grayscale:
        img_aug = img_aug.convert("L").convert("RGB")  # keep 3 channels
    
    width, height = img_aug.size
    img_np = np.array(img_aug)

    # --- Geometric / spatial transformations ---
    # Rotation
    if random.random() < apply_prob:
        angle = random.uniform(*MODERATE_PARAMS["rotation"])
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        img_np = cv2.warpAffine(img_np, M, (width, height), borderMode=cv2.BORDER_REFLECT)

    # Translation
    if random.random() < apply_prob:
        tx = random.uniform(*MODERATE_PARAMS["translation"]) * width
        ty = random.uniform(*MODERATE_PARAMS["translation"]) * height
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img_np = cv2.warpAffine(img_np, M, (width, height), borderMode=cv2.BORDER_REFLECT)

    # Scaling
    if random.random() < apply_prob:
        scale = random.uniform(*MODERATE_PARAMS["scaling"])
        img_np = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # Crop or pad to original size
        h, w = img_np.shape[:2]
        if h > height:
            start = (h - height)//2
            img_np = img_np[start:start+height, :]
        elif h < height:
            pad_top = (height - h)//2
            pad_bottom = height - h - pad_top
            img_np = cv2.copyMakeBorder(img_np, pad_top, pad_bottom, 0, 0, cv2.BORDER_REFLECT)
        if w > width:
            start = (w - width)//2
            img_np = img_np[:, start:start+width]
        elif w < width:
            pad_left = (width - w)//2
            pad_right = width - w - pad_left
            img_np = cv2.copyMakeBorder(img_np, 0, 0, pad_left, pad_right, cv2.BORDER_REFLECT)

    # Shear
    if random.random() < apply_prob:
        shear_angle = np.deg2rad(random.uniform(*MODERATE_PARAMS["shear"]))
        M = np.array([[1, np.tan(shear_angle), 0],
                      [0, 1, 0]], dtype=np.float32)
        img_np = cv2.warpAffine(img_np, M, (width, height), borderMode=cv2.BORDER_REFLECT)

    # Flips
    if random.random() < MODERATE_PARAMS["h_flip"]:
        img_np = cv2.flip(img_np, 1)
    if random.random() < MODERATE_PARAMS["v_flip"]:
        img_np = cv2.flip(img_np, 0)

    # --- Color / photometric transformations (skip if grayscale) ---
    img_aug = Image.fromarray(img_np)
    if not is_grayscale:
        # Brightness
        if random.random() < apply_prob:
            factor = random.uniform(*MODERATE_PARAMS["brightness"])
            img_aug = ImageEnhance.Brightness(img_aug).enhance(factor)
        # Contrast
        if random.random() < apply_prob:
            factor = random.uniform(*MODERATE_PARAMS["contrast"])
            img_aug = ImageEnhance.Contrast(img_aug).enhance(factor)

    img_np = np.array(img_aug)

    # --- Noise ---
    if random.random() < apply_prob:
        noise_std = random.uniform(*MODERATE_PARAMS["noise_std"])
        noise = np.random.normal(0, noise_std*255, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np.astype(np.float32)+noise, 0, 255).astype(np.uint8)

    # --- Blur ---
    if random.random() < apply_prob:
        radius = random.randint(*MODERATE_PARAMS["blur_radius"])
        img_aug = Image.fromarray(img_np)
        img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius))
        img_np = np.array(img_aug)

    return Image.fromarray(img_np)

def augment_and_save_dataset(TRAIN_DATA_DIR, OUTPUT_DIR, variants_per_image=6, apply_prob=0.6, grayscale_prob=0.5, img_size=(150,150)):
    """Augment all images in TRAIN_DATA_DIR and save to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    class_names = [d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))]

    for cls in class_names:
        cls_input_dir = os.path.join(TRAIN_DATA_DIR, cls)
        cls_output_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(cls_output_dir, exist_ok=True)

        img_files = [f for f in os.listdir(cls_input_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]

        for img_file in tqdm(img_files, desc=f"Class {cls}"):
            img_path = os.path.join(cls_input_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)

            for i in range(variants_per_image):
                aug_img = augment_image(img, apply_prob=apply_prob, grayscale_prob=grayscale_prob)
                base_name = os.path.splitext(img_file)[0]
                save_path = os.path.join(cls_output_dir, f"{base_name}_aug_{i+1}.jpg")
                aug_img.save(save_path, "JPEG")

# --- Define output directory ---
OUTPUT_DIR = "data_augmented/moderate_small_mix_match"

# --- Call the augmentation function ---
augment_and_save_dataset(
    TRAIN_DATA_DIR=TRAIN_DATA_DIR,    # your existing training data
    OUTPUT_DIR=OUTPUT_DIR,            # where to save augmented images
    variants_per_image=20,             # how many augmented versions per original
    apply_prob=0.4,                   # probability of each augmentation being applied
    grayscale_prob=0.2,               # probability of converting to grayscale
    img_size=(150, 150)               # final resize dimension
)

print(f"\n✅ Augmentation complete! All images saved under: {OUTPUT_DIR}")

# --- CONFIG ---
BATCH_SIZE = 500
EPOCHS = 100
NUM_CLASSES = 10
DATA_AUG_DIR = OUTPUT_BASE


# --- Load training dataset ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    OUTPUT_DIR,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# --- Resize validation dataset to match training ---
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DATA_DIR,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- Define CNN model ---
# 150x150x3 input size
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(5, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(5, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(5, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(5, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CLASSES)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# --- Callbacks ---
checkpoint_path = os.path.join(f"best_model_{OUTPUT_DIR}.keras")
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# --- Train ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=2
)

# --- Store best validation accuracy ---
best_val_loss = max(history.history['val_loss'])