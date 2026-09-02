import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from src.config import (
    HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_ORIENTATIONS,
    LBP_RADIUS, LBP_N_POINTS, LBP_METHOD,
    GLCM_DISTANCES, GLCM_ANGLES,
    CANNY_LOW, CANNY_HIGH,
)


def extract_hog(img):
    features = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_lbp_histogram(img):
    lbp = local_binary_pattern(img, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_glcm_features(img):
    img_q = (img / 8).astype(np.uint8)
    glcm = graycomatrix(
        img_q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        levels=32, symmetric=True, normed=True,
    )
    props = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        props.extend(graycoprops(glcm, prop).ravel())
    return np.array(props)


def extract_edge_contour_features(img):
    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)
    edge_density = np.count_nonzero(edges) / edges.size

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        perims = [cv2.arcLength(c, True) for c in contours]
        largest_area = max(areas)
        largest_perim = perims[int(np.argmax(areas))]
        mean_area = float(np.mean(areas))
    else:
        largest_area = largest_perim = mean_area = 0.0

    return np.array([edge_density, num_contours, largest_area, largest_perim, mean_area])


def extract_intensity_stats(img):
    flat = img.astype(np.float64).ravel()
    return np.array([flat.mean(), flat.std(), skew(flat)])


def extract_features(img):
    hog_feat = extract_hog(img)
    lbp_feat = extract_lbp_histogram(img)
    glcm_feat = extract_glcm_features(img)
    edge_feat = extract_edge_contour_features(img)
    stat_feat = extract_intensity_stats(img)

    return np.concatenate([hog_feat, lbp_feat, glcm_feat, edge_feat, stat_feat])


FEATURE_BLOCK_NAMES = ["HOG", "LBP", "GLCM", "Edge/Contour", "Intensity stats"]
