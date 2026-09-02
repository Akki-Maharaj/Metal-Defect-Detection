import argparse
import os

import cv2
import joblib
import numpy as np

from src.config import MODELS_DIR, OUTPUTS_DIR, IMG_SIZE, CLASS_DISPLAY_NAMES
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.defect_localization import localize_defects, draw_boxes


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    return model, scaler, le


def predict_image(img_gray, model=None, scaler=None, le=None):
    if model is None:
        model, scaler, le = load_artifacts()

    pre = preprocess(img_gray, size=IMG_SIZE)
    feat = extract_features(pre).reshape(1, -1)
    feat_s = scaler.transform(feat)

    pred_idx = model.predict(feat_s)[0]
    label = le.inverse_transform([pred_idx])[0]

    probs = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_s)[0]
        probs = {le.classes_[i]: float(p) for i, p in enumerate(proba)}

    mask, boxes = localize_defects(pre)

    return {
        "predicted_label": label,
        "predicted_display_name": CLASS_DISPLAY_NAMES.get(label, label),
        "class_probabilities": probs,
        "mask": mask,
        "boxes": boxes,
        "preprocessed_image": pre,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--out", default=os.path.join(OUTPUTS_DIR, "prediction.png"),
                         help="Where to save the annotated visualization")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.image)

    result = predict_image(img)

    print(f"Predicted class: {result['predicted_display_name']}")
    if result["class_probabilities"]:
        print("Class probabilities:")
        for cls, p in sorted(result["class_probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {CLASS_DISPLAY_NAMES.get(cls, cls):<12s} {p:.3f}")
    print(f"Detected {len(result['boxes'])} defect region(s).")

    vis = draw_boxes(result["preprocessed_image"], result["boxes"])
    cv2.putText(vis, result["predicted_display_name"], (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(args.out, vis)
    print(f"Saved visualization -> {args.out}")


if __name__ == "__main__":
    main()
