import argparse
import csv
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import grade_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_actual_condition(image_path):
    """
    Extract actual condition from paths such as:
    data/sample/apple/fresh/image.jpg
    data/sample/apple/rotten/image.jpg
    """
    parts = [part.lower() for part in image_path.parts]

    if "fresh" in parts:
        return "fresh"

    if "rotten" in parts:
        return "rotten"

    return ""


def get_fruit_name(image_path):
    """
    Extract fruit name from:
    data/sample/<fruit>/<condition>/image.jpg
    """
    parts = image_path.parts

    for i, part in enumerate(parts):
        if part.lower() in {"fresh", "rotten"} and i > 0:
            return parts[i - 1]

    return ""


def batch_grade(input_dir, output_csv):
    input_dir = Path(input_dir)

    image_paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    rows = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Grading {image_path}")

        try:
            result = grade_image(str(image_path))

            actual_condition = get_actual_condition(image_path)
            fruit = get_fruit_name(image_path)

            rows.append({
                "image": str(image_path),
                "fruit": fruit,
                "actual_condition": actual_condition,
                "rule_grade": result["rule_grade"],
                "knn_prediction": result["knn_prediction"],
                "defect_ratio": result["defect_ratio"],
                "output_image": result["output_image"],
            })

        except Exception as error:
            print(f"ERROR: {image_path}")
            print(error)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "fruit",
        "actual_condition",
        "rule_grade",
        "knn_prediction",
        "defect_ratio",
        "output_image",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("BATCH GRADING COMPLETE")
    print("=" * 60)
    print(f"Images processed: {len(rows)}")
    print(f"CSV report: {output_csv}")

    # Calculate KNN accuracy when actual labels are available
    labeled_rows = [
        row for row in rows
        if row["actual_condition"] in {"fresh", "rotten"}
    ]

    if labeled_rows:
        correct = sum(
            row["actual_condition"] == row["knn_prediction"]
            for row in labeled_rows
        )

        accuracy = correct / len(labeled_rows)

        print(f"KNN accuracy: {accuracy:.2%}")
        print(f"Correct predictions: {correct}/{len(labeled_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Grade all fruit images in a directory."
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing fruit images."
    )

    parser.add_argument(
        "--output",
        default="outputs/batch_report.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    batch_grade(args.dir, args.output)


if __name__ == "__main__":
    main()
