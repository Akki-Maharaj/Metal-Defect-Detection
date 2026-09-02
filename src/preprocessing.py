import cv2
import numpy as np

from src.config import IMG_SIZE


def preprocess(img_gray, size=IMG_SIZE):
    img = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)

    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img
