import os
import sys
import subprocess

# --- Install required packages if not already installed ---
def install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["opencv-python", "albumentations", "tqdm", "numpy"]:
    install(pkg)

# --- Now import after installation ---
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from glob import glob

# Input and output directories
input_root = "/home/hutlab_int/Akhil_yolo/Research/BraTS21"
output_root = "/home/hutlab_int/Akhil_yolo/Research/BraTS21_preprocessed2"
os.makedirs(output_root, exist_ok=True)

# Augmentation + preprocessing pipeline (based on paper)
transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def is_black(img, threshold=1):
    """Check if the slice is nearly black (empty)."""
    return np.mean(img) < threshold

for split in ["train", "val"]:
    img_dir = os.path.join(input_root, split, "images")
    lbl_dir = os.path.join(input_root, split, "labels")
    out_img_dir = os.path.join(output_root, split, "images")
    out_lbl_dir = os.path.join(output_root, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img_files = glob(os.path.join(img_dir, "*.png"))

    for img_path in tqdm(img_files, desc=f"Processing {split}"):
        fname = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, fname.replace(".png", ".txt"))

        # Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Skip black slices
        if is_black(img):
            continue

        # Normalize intensities (0–255)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Read YOLO labels if exists
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                labels = [l.strip().split() for l in f.readlines()]
            bboxes = [[float(x) for x in l[1:]] for l in labels]
            class_labels = [int(l[0]) for l in labels]
        else:
            bboxes, class_labels = [], []

        # Apply augmentation (keeps labels consistent)
        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_classes = augmented["class_labels"]

        # Save processed image
        out_img_path = os.path.join(out_img_dir, fname)
        cv2.imwrite(out_img_path, aug_img)

        # Save updated labels
        out_lbl_path = os.path.join(out_lbl_dir, fname.replace(".png", ".txt"))
        with open(out_lbl_path, "w") as f:
            for cls, box in zip(aug_classes, aug_bboxes):
                f.write(f"{cls} " + " ".join(map(str, box)) + "\n")

print("✅ Preprocessing completed. Output saved at:", output_root)
