import cv2
import numpy as np

MIN_DEFECT_AREA = 25


def localize_defects(img_gray, min_area=MIN_DEFECT_AREA):
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            boxes.append(cv2.boundingRect(c))

    return cleaned, boxes


def draw_boxes(img_gray, boxes, color=255):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis


def boxes_to_mask(boxes, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for (x, y, w, h) in boxes:
        mask[y:y + h, x:x + w] = 255
    return mask


def iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union
