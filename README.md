# PCA-Based Face Recognition using Yale Dataset

## Overview
This project implements Person Identification and Facial Expression Recognition using Principal Component Analysis (PCA) on the Yale Face Dataset.

The system performs classification based on reconstruction error. A test image is projected onto different PCA subspaces, reconstructed, and the label corresponding to the minimum reconstruction error is selected.

---

## Key Concepts
- Principal Component Analysis (PCA)
- Dimensionality Reduction
- Eigenfaces (via SVD)
- Reconstruction-based classification
- Euclidean distance

---

## Dataset
This project uses the Yale Face Dataset:
- 15 subjects
- 11 images per subject
- Variations in lighting and facial expressions

Download the dataset from:
https://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html

After downloading, organize it as follows:

```
yale/
 ├── yale1/
 ├── yale2/
 ├── test1/
 ├── test2/
```

---

## Installation

Install required dependencies:

```
pip install numpy opencv-python
```

---

## How to Run

Run the following command:

```
python bscs24093_AI_MiniProj2.py
```

---

## Tasks Implemented

### 1. Person Identification
- PCA is computed separately for each person
- Principal components preserving 99% variance are retained
- Test images are classified using reconstruction error

### 2. Facial Expression Recognition
- PCA is computed for each expression category
- Tested on unseen subjects
- Classification is based on minimum reconstruction loss

---

## Results
- Person Identification Accuracy: 80%
- Facial Expression Recognition Accuracy: 25.8%

---

## Implementation Details
- Images are converted to grayscale and flattened into vectors
- PCA is implemented using Singular Value Decomposition (SVD)
- Covariance matrix is avoided due to high memory requirements
- Reconstruction error is computed using Euclidean norm

---

## Notes
- The dataset is not included due to size and licensing considerations
- Ensure correct folder structure before running the code
- Only NumPy and OpenCV are used as required

---

## Conclusion
This project demonstrates the use of PCA for:
- Face recognition
- Facial expression classification
- Efficient handling of high-dimensional image data

The results show that person identification performs significantly better than expression recognition due to stronger identity-specific features and less variation across samples.
