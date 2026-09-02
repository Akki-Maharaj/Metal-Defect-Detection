import argparse
import time
import os
import json

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import (
    DATA_DIR_FULL, DATA_DIR_SAMPLE, MODELS_DIR, OUTPUTS_DIR,
    IMG_SIZE, RANDOM_STATE, TEST_SIZE, CLASSES, CLASS_DISPLAY_NAMES,
)
from src.data_loader import load_dataset
from src.preprocessing import preprocess
from src.feature_extraction import extract_features


def build_feature_matrix(data_dir):
    print(f"Loading images from {data_dir} ...")
    X_imgs, y_labels, paths = load_dataset(data_dir, size=None)
    print(f"Loaded {len(X_imgs)} images across {len(set(y_labels))} classes.")

    print("Extracting features (preprocess -> HOG/LBP/GLCM/edge/stats) ...")
    t0 = time.time()
    feats = []
    for img in X_imgs:
        pre = preprocess(img, size=IMG_SIZE)
        feats.append(extract_features(pre))
    X = np.vstack(feats)
    print(f"Done in {time.time() - t0:.1f}s. Feature vector length: {X.shape[1]}")
    return X, np.array(y_labels), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DIR_FULL,
                         help="Path to a class-subfoldered image directory "
                              "(default: data/full; use data/sample for a quick test)")
    args = parser.parse_args()

    X, y, paths = build_feature_matrix(args.data)

    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "svm": SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                    random_state=RANDOM_STATE, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        ),
    }

    results = {}
    best_name, best_model, best_acc = None, None, -1

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, n_jobs=-1)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0
        )
        print(f"{name} 5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        print(f"{name} held-out test accuracy: {acc:.3f}")
        print(report)

        results[name] = {
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
            "test_accuracy": float(acc),
        }

        with open(os.path.join(OUTPUTS_DIR, f"classification_report_{name}.txt"), "w") as f:
            f.write(report)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[CLASS_DISPLAY_NAMES[c] for c in le.classes_],
            yticklabels=[CLASS_DISPLAY_NAMES[c] for c in le.classes_],
        )
        plt.title(f"Confusion Matrix ({name}) - test accuracy {acc:.3f}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f"confusion_matrix_{name}.png"), dpi=150)
        plt.close()

        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} (test accuracy {best_acc:.3f})")
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    with open(os.path.join(OUTPUTS_DIR, "results_summary.json"), "w") as f:
        json.dump({"best_model": best_name, **results}, f, indent=2)

    print(f"\nSaved model -> {MODELS_DIR}/best_model.joblib")
    print(f"Saved plots/reports -> {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
