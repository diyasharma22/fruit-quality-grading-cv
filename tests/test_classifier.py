import numpy as np
from src.classifier import rule_based_grade, train_knn
import config


def test_rule_based_good():
    threshold = config.GRADE_THRESHOLDS["Good"]
    assert rule_based_grade(0) == "Good"
    assert rule_based_grade(threshold) == "Good"


def test_rule_based_medium():
    good = config.GRADE_THRESHOLDS["Good"]
    medium = config.GRADE_THRESHOLDS["Medium"]

    value = (good + medium) / 2
    assert rule_based_grade(value) == "Medium"


def test_rule_based_poor():
    medium = config.GRADE_THRESHOLDS["Medium"]
    assert rule_based_grade(medium + 0.01) == "Poor"


def test_knn_training():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0.2],
        [0.2, 0],
        [10, 10],
        [10, 11],
        [11, 10],
        [11, 11],
        [10, 10.2],
        [10.2, 10],
    ])

    y = np.array([
        "fresh", "fresh", "fresh", "fresh", "fresh", "fresh",
        "rotten", "rotten", "rotten", "rotten", "rotten", "rotten"
    ])

    model, scaler, accuracy, report = train_knn(X, y)

    assert model is not None
    assert scaler is not None
    assert 0 <= accuracy <= 1
    assert report is not None
