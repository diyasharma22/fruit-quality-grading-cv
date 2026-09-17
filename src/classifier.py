"""
src/classifier.py
Module 4: Grading.

Provides two grading approaches:
  - rule_based_grade(): defect_ratio thresholded directly against
    config.GRADE_THRESHOLDS. Fast, interpretable, no training needed.
  - KNN classifier: trained on full feature vectors to predict
    fresh/rotten, for comparison against the rule-based approach.

train_and_save() trains the KNN + scaler and bundles them with the
already-fitted PCA (loaded from disk) into a single joblib file so
grade_image() can score a brand-new image standalone at inference time.
"""

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import config


def rule_based_grade(defect_ratio):
    """
    Grade purely from defect_ratio, no model needed.
    <= Good threshold -> "Good"
    <= Medium threshold -> "Medium"
    otherwise -> "Poor"
    """
    if defect_ratio <= config.GRADE_THRESHOLDS["Good"]:
        return "Good"
    elif defect_ratio <= config.GRADE_THRESHOLDS["Medium"]:
        return "Medium"
    else:
        return "Poor"


def train_knn(X, y):
    """
    Scale features and train a KNN classifier.
    Returns (model, scaler, test accuracy, report string) so callers
    can inspect performance before deciding to save the bundle.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBORS)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, scaler, acc, report


def save_model_bundle(model, scaler, pca, path=config.MODEL_BUNDLE_PATH):
    """
    Bundle the KNN model, scaler, and PCA together -- all three are
    needed together to grade a new image from scratch at inference time.
    """
    joblib.dump({"model": model, "scaler": scaler, "pca": pca}, path)


def load_model_bundle(path=config.MODEL_BUNDLE_PATH):
    return joblib.load(path)


def predict_grade_from_vector(bundle, feature_vector):
    """
    Predict fresh/rotten from a single (unscaled) feature vector using
    a loaded model bundle. feature_vector should already have PCA
    applied (i.e. built via features.build_feature_vector using the
    same PCA stored in this bundle).
    """
    scaled = bundle["scaler"].transform(feature_vector.reshape(1, -1))
    prediction = bundle["model"].predict(scaled)[0]
    return prediction
