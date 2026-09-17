# Problem Statement

Manual fruit quality inspection in agriculture and food retail is slow, inconsistent, and subjective — different inspectors can grade the same fruit differently, and manual checks don't scale to high produce volumes. There is a need for an automated, repeatable way to assess fruit condition from a simple photo.

## Scope

This project builds a classical computer vision pipeline (no deep learning) that:

- Accepts a single fruit image or a directory of images as input
- Isolates the fruit from its background
- Detects visible surface blemishes
- Extracts a fixed set of visual features (color, texture, edge/corner density, blemish coverage)
- Classifies the fruit as fresh or rotten using a trained KNN model
- Assigns an interpretable quality grade (Good / Medium / Poor) using a rule-based defect ratio threshold
- Produces an annotated output image and, for batch runs, a CSV report

The current implementation is validated against a synthetic dataset (generated in-repo) rather than real photographs, since no real labelled dataset was available at development time. The data folder layout matches the real Mendeley "Fruits Dataset for Classification", so the pipeline can be pointed at real data without code changes.

Out of scope for this version: deep learning models, a web/GUI interface, and multi-angle/3D fruit analysis.

## Target Users

- Small-scale fruit vendors or farmers who want a low-cost, no-internet-required first-pass quality check
- Students/researchers evaluating classical CV techniques for agricultural defect detection
- Developers who want a working reference pipeline to swap in a real dataset and extend

## High-Level Features

- Single-image grading via CLI (`main.py`)
- Batch grading over a directory with CSV export (`tools/batch_grade.py`)
- Synthetic dataset generator for development/testing without external data
- Dual grading output: rule-based quality grade + KNN freshness classification
- Annotated visual output showing detected fruit boundary and blemishes
- Automated test suite covering preprocessing, segmentation, and classification
