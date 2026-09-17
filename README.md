# Fruit Quality Grader

A computer vision pipeline that grades fruit images (apple, banana, orange)
as **fresh** or **rotten**, and assigns a quality grade (**Good / Medium /
Poor**) based on visible surface blemish coverage.

Built entirely with classical computer vision (OpenCV) and a lightweight
KNN classifier — no deep learning / neural networks involved.

## How it works

The pipeline runs in five stages:

1. **Preprocessing** (`src/preprocessing.py`) — load, resize, denoise
   (Gaussian + bilateral filter), and enhance contrast (CLAHE on the L
   channel of LAB color space).
2. **Segmentation** (`src/segmentation.py`) — isolate the fruit from its
   background using Otsu's automatic thresholding on the HSV saturation
   channel, then detect dark/brown blemish regions inside the fruit mask
   using HSV value-channel thresholding.
3. **Feature extraction** (`src/features.py`) — build an 18-dimensional
   feature vector per image: a PCA-reduced HSV color histogram (6
   components), Canny edge density, Harris corner density, K-Means
   dominant fruit colors, and the blemish defect ratio.
4. **Grading** (`src/classifier.py`) — two independent grading methods:
   - **Rule-based**: defect ratio thresholded directly into Good / Medium
     / Poor.
   - **KNN classifier**: trained on the full feature vector to predict
     fresh / rotten.
5. **Reporting** (`src/report.py`) — draws the fruit outline (green) and
   blemish contours (red) on the image, overlays both grades and the
   defect ratio, and saves the result to `outputs/`.

These two grading outputs answer different questions and can disagree by
design: the rule-based grade is a **quality scale** based purely on
visible blemish coverage, while the KNN prediction is a **binary
freshness classification** based on the full feature vector. A fruit with
a small visible blemish area can land as "Medium" quality while still
being correctly classified as "rotten" in origin.

## Project structure

```text
fruit-quality-grader/
  main.py                        CLI entrypoint: grade a single image
  config.py                      All tunable parameters in one place
  src/
    preprocessing.py             Module 1
    segmentation.py              Module 2
    features.py                  Module 3
    classifier.py                Module 4
    report.py                    Module 5
  tools/
    generate_sample_dataset.py   Synthetic dataset generator
    build_dataset.py             Builds feature matrix + fits PCA
  data/sample/                   Generated synthetic images (gitignored)
  models/                        Trained PCA + KNN bundle (gitignored)
  outputs/                       Annotated graded images (gitignored)
  tests/                         Unit tests
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Usage

**1. Generate synthetic sample data** (no internet / real dataset needed):

```bash
python tools/generate_sample_dataset.py --out data/sample --per-class 40
```

**2. Build the feature dataset and fit PCA:**

```bash
python tools/build_dataset.py --data data/sample
```

**3. Train the KNN classifier:**

```bash
python -c "
import joblib
from src.classifier import train_knn, save_model_bundle

data = joblib.load('data/features.joblib')
model, scaler, acc, report = train_knn(data['X'], data['y'])
print('Test accuracy:', acc)

pca = joblib.load('models/pca.joblib')
save_model_bundle(model, scaler, pca)
"
```

**4. Grade an image:**

```bash
python main.py --image data/sample/apple/rotten/img_000.jpg
```

This prints both grades to the console and saves an annotated image to
`outputs/`.

## A note on accuracy

The KNN classifier scores 100% on the synthetic dataset. This is **not**
representative of real-world performance — the synthetic fresh/rotten
classes are deliberately easy to separate (plain circle vs. circle with
dark blobs). Real fruit photos will have subtler, noisier blemishes,
inconsistent lighting, and background clutter.

## Using a real dataset

The folder layout (`data/<root>/<fruit>/<condition>/*.jpg`) matches the
Mendeley "Fruits Dataset for Classification" layout, so real data can be
substituted by pointing `tools/build_dataset.py --data <path>` at a real
dataset directory instead of `data/sample` — no code changes needed.
