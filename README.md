# MNIST Digit Classifier via Hand-Crafted Statistical Features

> Classifying handwritten digits (0–9) from the MNIST dataset using 14 custom statistical image features fed into a Random Forest — no deep learning, no convolutions, just signal processing fundamentals.

---

## Overview

This project explores whether classical statistical descriptors extracted from raw pixel intensities are sufficient to distinguish handwritten digits. Instead of training a neural network, each image is compressed into a 14-dimensional feature vector capturing brightness, spread, shape, texture, and contrast. A Random Forest classifier is then trained on these vectors.

It serves as a clean demonstration of the feature engineering → model training → evaluation pipeline for image classification.

---

## How It Works

```
MNIST Image (28×28 px)
        ↓
  Grayscale Conversion
        ↓
  Flatten to 1D Pixel Array (784 values)
        ↓
  Extract 14 Statistical Features
        ↓
  Random Forest Classifier (100 trees)
        ↓
  Predict Digit Label (0–9)
```

---

## Feature Vector (14 Features)

Each image is described by the following hand-crafted features, all computed from its raw pixel intensities:

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Mean** | Average pixel brightness |
| 2 | **Median** | Middle value of sorted pixel list |
| 3 | **Variance** | Spread of pixel intensities |
| 4 | **Standard Deviation** | Square root of variance |
| 5 | **Skewness** | Asymmetry of intensity distribution |
| 6 | **Kurtosis** | Tailedness (excess kurtosis; normal = 0) |
| 7 | **Entropy** | Shannon entropy (bits) — texture complexity |
| 8 | **Minimum Intensity** | Darkest pixel value |
| 9 | **Maximum Intensity** | Brightest pixel value |
| 10 | **Median Absolute Deviation (MAD)** | Robust spread measure |
| 11 | **Local Contrast** | Michelson contrast: (max−min)/(max+min) |
| 12 | **Local Probability** | Frequency of the most common pixel value |
| 13 | **25th Percentile** | Lower quartile of intensity |
| 14 | **75th Percentile** | Upper quartile of intensity |

All statistical functions are implemented from scratch (no NumPy/SciPy for the math).

---

## Dataset

**MNIST** — 70,000 grayscale images of handwritten digits (28×28 pixels), loaded via Hugging Face `datasets`.

- 60,000 training images + 10,000 test images combined, then re-split 70/30
- 10 classes: digits 0 through 9
- Labels are balanced across classes

---


## Setup & Usage

### Requirements

```bash
pip install datasets Pillow scikit-learn pandas numpy opencv-python
```

### Run

Open the notebook in [Google Colab](https://colab.research.google.com) or locally in Jupyter and run all cells top to bottom. The notebook will:

1. Install dependencies
2. Download MNIST via Hugging Face
3. Extract features from all 70,000 images
4. Train a Random Forest on 70% of the data
5. Evaluate on the remaining 30% and print metrics

---

## Results

The model is evaluated on the 30% held-out test set using weighted averages across all 10 digit classes:

| Metric | Score |
|--------|-------|
| Accuracy | 30.49% |
| Precision | 28.86% |
| Recall | 30.49% |
| F1-Score | 29.46% |


---

## Key Design Decisions

- **No libraries for statistics** — Mean, median, variance, skewness, kurtosis, entropy, MAD, percentiles, contrast, and probability are all implemented manually in pure Python to demonstrate understanding of the underlying math.
- **Random Forest** — Chosen for its robustness to noisy/correlated features and interpretability via feature importances.
- **Stratified split** — `train_test_split(..., stratify=y)` ensures each digit class is proportionally represented in both train and test sets.

---

## Author

**Laiba Munir**  
