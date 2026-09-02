import os
import cv2
import numpy as np

from src.config import CLASSES


def list_image_paths(data_dir):
    samples = []
    for label in CLASSES:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(".jpg"):
                img_path = os.path.join(class_dir, fname)
                mask_path = os.path.join(class_dir, fname[:-4] + ".png")
                mask_path = mask_path if os.path.isfile(mask_path) else None
                samples.append((img_path, mask_path, label))
    return samples


def load_image(path, size=None, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def load_dataset(data_dir, size=None):
    samples = list_image_paths(data_dir)
    if not samples:
        raise RuntimeError(
            f"No images found under {data_dir}. "
            f"Run download_data.sh first, or point to data/sample for a quick test."
        )
    X, y, paths = [], [], []
    for img_path, _mask_path, label in samples:
        img = load_image(img_path, size=size)
        X.append(img)
        y.append(label)
        paths.append(img_path)
    return X, y, paths
