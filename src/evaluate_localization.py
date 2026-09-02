import argparse
import os

import cv2
import numpy as np

from src.config import DATA_DIR_FULL, IMG_SIZE, OUTPUTS_DIR
from src.data_loader import list_image_paths
from src.preprocessing import preprocess
from src.defect_localization import localize_defects, boxes_to_mask, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DIR_FULL)
    args = parser.parse_args()

    samples = list_image_paths(args.data)
    scores = []
    per_class = {}

    for img_path, mask_path, label in samples:
        if mask_path is None:
            continue

        gt_check = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_check is None or not np.any(gt_check > 0):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        pre = preprocess(img, size=IMG_SIZE)
        gt_resized = cv2.resize(gt_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        cleaned, _boxes = localize_defects(pre)
        score = iou(cleaned, gt_resized)
        scores.append(score)
        per_class.setdefault(label, []).append(score)

    print(f"Evaluated {len(scores)} defect images with ground-truth masks.")
    print(f"Mean IoU (all defect classes): {np.mean(scores):.3f}")
    print("\nPer-class mean IoU:")
    for label, s in per_class.items():
        print(f"  {label:<12s} {np.mean(s):.3f}  (n={len(s)})")

    with open(os.path.join(OUTPUTS_DIR, "localization_iou.txt"), "w") as f:
        f.write(f"Mean IoU: {np.mean(scores):.3f}\n")
        for label, s in per_class.items():
            f.write(f"{label}: {np.mean(s):.3f} (n={len(s)})\n")


if __name__ == "__main__":
    main()
