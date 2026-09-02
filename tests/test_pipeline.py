import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR_SAMPLE, IMG_SIZE
from src.data_loader import list_image_paths, load_image
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.defect_localization import localize_defects, iou, boxes_to_mask


def _sample_image():
    samples = list_image_paths(DATA_DIR_SAMPLE)
    assert samples, "sample dataset is missing -- did you clone with data/sample intact?"
    img_path, _mask, _label = samples[0]
    return load_image(img_path)


def test_preprocess_shape():
    img = _sample_image()
    out = preprocess(img, size=IMG_SIZE)
    assert out.shape == IMG_SIZE
    assert out.dtype == np.uint8


def test_feature_vector_is_fixed_length_and_finite():
    img = _sample_image()
    pre = preprocess(img, size=IMG_SIZE)
    feat1 = extract_features(pre)
    feat2 = extract_features(pre)
    assert feat1.shape == feat2.shape
    assert np.all(np.isfinite(feat1))


def test_localize_defects_returns_boxes_in_image_bounds():
    img = _sample_image()
    pre = preprocess(img, size=IMG_SIZE)
    mask, boxes = localize_defects(pre)
    assert mask.shape == IMG_SIZE
    for (x, y, w, h) in boxes:
        assert 0 <= x < IMG_SIZE[1]
        assert 0 <= y < IMG_SIZE[0]
        assert x + w <= IMG_SIZE[1]
        assert y + h <= IMG_SIZE[0]


def test_iou_identical_masks_is_one():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    assert iou(mask, mask) == 1.0


def test_iou_disjoint_masks_is_zero():
    a = np.zeros((50, 50), dtype=np.uint8)
    a[0:10, 0:10] = 255
    b = np.zeros((50, 50), dtype=np.uint8)
    b[40:50, 40:50] = 255
    assert iou(a, b) == 0.0


def test_boxes_to_mask_shape():
    mask = boxes_to_mask([(0, 0, 5, 5)], (50, 50))
    assert mask.shape == (50, 50)
    assert mask[2, 2] == 255
    assert mask[30, 30] == 0
