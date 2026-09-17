"""
main.py
End-to-end CLI: given an image path, run the full pipeline
(preprocess -> segment -> detect blemishes -> extract features -> grade)
and produce both a console summary and an annotated output image.

Usage:
    python main.py --image data/sample/apple/rotten/img_000.jpg
"""

import argparse

import config
from src.preprocessing import preprocess_image
from src.segmentation import segment_foreground, detect_blemishes
from src.features import extract_raw_features, build_feature_vector, load_pca
from src.classifier import rule_based_grade, load_model_bundle, predict_grade_from_vector
from src.report import draw_annotations, save_report_image


def grade_image(image_path):
    """
    Run the full pipeline on a single image and return a result dict
    with both grading approaches plus the annotated image.
    """
    img = preprocess_image(image_path)
    fruit_mask = segment_foreground(img)
    blemish_mask, defect_ratio, blemish_contours = detect_blemishes(img, fruit_mask)

    rule_grade = rule_based_grade(defect_ratio)

    bundle = load_model_bundle()
    raw = extract_raw_features(img, fruit_mask, defect_ratio)
    feature_vector = build_feature_vector(raw, bundle["pca"])
    knn_prediction = predict_grade_from_vector(bundle, feature_vector)

    text_lines = [
        f"Rule-based: {rule_grade}",
        f"KNN: {knn_prediction}",
        f"Defect ratio: {defect_ratio:.4f}",
    ]
    annotated = draw_annotations(img, fruit_mask, blemish_contours, text_lines)
    out_path = save_report_image(annotated, image_path)

    return {
        "rule_grade": rule_grade,
        "knn_prediction": knn_prediction,
        "defect_ratio": defect_ratio,
        "output_image": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Grade a fruit image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    result = grade_image(args.image)

    print(f"Rule-based grade : {result['rule_grade']}")
    print(f"KNN prediction   : {result['knn_prediction']}")
    print(f"Defect ratio     : {result['defect_ratio']:.4f}")
    print(f"Annotated output : {result['output_image']}")


if __name__ == "__main__":
    main()
