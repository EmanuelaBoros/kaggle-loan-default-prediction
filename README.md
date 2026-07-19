# Kaggle Loan Default Prediction

Legacy experiment code for the Kaggle **Loan Default Prediction** competition. The repository contains scripts for feature selection, binary default detection, regression over positive losses, threshold tuning, and Kaggle submission-file generation.

The code reflects an exploratory competition workflow rather than a packaged library. It was written for an older Python/scikit-learn stack and uses hard-coded paths and filenames in several places.

## Project Idea

The main approach is a two-stage prediction pipeline:

1. **Binary classification**: predict whether the loan loss is zero or non-zero.
2. **Regression on non-zero losses**: estimate the loss amount only for examples predicted as non-zero.
3. **Submission formatting**: combine zero and non-zero predictions and write a Kaggle-style `id,loss` CSV file.

Several scripts explore alternative classifiers, regressors, feature subsets, and thresholds.
