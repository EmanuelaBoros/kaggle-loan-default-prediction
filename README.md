# Kaggle Loan Default Prediction

Legacy experiment code for the Kaggle **Loan Default Prediction** competition. The repository contains scripts for feature selection, binary default detection, regression over positive losses, threshold tuning, and Kaggle submission-file generation.

The code reflects an exploratory competition workflow rather than a packaged library. It was written for an older Python/scikit-learn stack and uses hard-coded paths and filenames in several places.

## Project Idea

The main approach is a two-stage prediction pipeline:y

1. **Binary classification**: predict whether the loan loss is zero or non-zero.
2. **Regression on non-zero losses**: estimate the loss amount only for examples predicted as non-zero.
3. **Submission formatting**: combine zero and non-zero predictions and write a Kaggle-style `id,loss` CSV file.

Several scripts explore alternative classifiers, regressors, feature subsets, and thresholds.


## Repository Contents

| File / folder | Purpose |
|---|---|
| `benchmark.py` | Baseline end-to-end pipeline using imputation, scaling, feature selection, binary classification, regression, and submission writing. |
| `competition_prediction.py` | Main competition prediction experiments with hand-selected features, logistic classification, quantile regression, and submission generation. |
| `regression_models.py` | Early two-stage logistic/regression model using selected features. |
| `regression_models_roc_visualisation.py` | Variant using sample data and ROC-oriented analysis. |
| `binary_model_selection.py` | Cross-validation experiments for binary default/non-default classifiers. |
| `regression_model_selection.py` | Cross-validation experiments for non-zero-loss regression models. |
| `find_best_threshold.py` | Searches probability thresholds for converting classifier outputs into default/non-default decisions. |
| `iterative_feature_selection.py` | Iterative feature-selection experiments for AUC, Pearson correlation, and PR area. |
| `iterative_feature_selection_last.py` | Later feature-selection variant with target-specific selection outputs. |
| `liblinear_feature_elimination.py` | Exports selected features in LIBLINEAR/LIBSVM format and supports feature-elimination experiments. |
| `linear_model_over_prediction_scores.py` | Fits linear models over saved prediction scores from cross-validation splits. |
| `linear_model_over_prediction_scores_with_CV.py` | Cross-validation variant of the prediction-score modeling workflow. |
| `glmsklearn/` | Vendored scikit-learn-style wrappers around statsmodels GLM models. |
| `glm-sklearn-master/` | Original vendored `glm-sklearn` source snapshot. |

## Expected Data Layout

The scripts expect competition files under a local `data/` directory:

```text
data/
âââ train_v2.csv
âââ test_v2.csv
âââ sampleSubmission.csv
âââ train_v2_sample_10k.csv          # optional sample file used by some scripts
âââ test_v2_sample_10k.csv           # optional sample file used by some scripts
âââ *.pkl                            # intermediate feature-selection and CV outputs
```

Submission files are usually written under:

```text
predictions/
âââ predictions_competition_*.csv
```

Some scripts also expect or write:

```text
models/
âââ *.pkl
```

Create these folders before running scripts that write outputs:

```bash
mkdir -p data predictions models
```

## Requirements

This is legacy Python 2 code. It uses APIs that were later removed or renamed, such as:

- `cPickle`
- `sklearn.cross_validation`
- `sklearn.preprocessing.Imputer`
- Python 2 `print` statements

Recommended legacy environment:

```bash
python --version  # Python 2.7 expected
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

For modern Python 3, the scripts will need updates such as:

- Replace `cPickle` with `pickle`.
- Replace `sklearn.cross_validation` with `sklearn.model_selection`.
- Replace `sklearn.preprocessing.Imputer` with `sklearn.impute.SimpleImputer`.
- Convert `print` statements to Python 3 syntax.

## Typical Workflow

### 1. Add Kaggle Data

Download the competition files from Kaggle and place them in `data/`:

```text
data/train_v2.csv
data/test_v2.csv
data/sampleSubmission.csv
```

The scripts assume a `loss` column in the training file and feature columns named like `f527`, `f528`, etc.

### 2. Explore Binary Thresholds

Use threshold search to estimate a useful cutoff for default/non-default prediction:

```bash
python find_best_threshold.py
```

The script reports cross-validation AUC and mean absolute error for thresholded predictions.

### 3. Select Binary or Regression Models

Binary model experiments:

```bash
python binary_model_selection.py
```

Regression model experiments:

```bash
python regression_model_selection.py
```

These scripts evaluate feature subsets and model choices using cross-validation.

### 4. Run Competition Prediction

Generate a submission-oriented prediction file:

```bash
python competition_prediction.py
```

The main prediction functions:

- Read `data/train_v2.csv` and `data/test_v2.csv`.
- Impute missing values.
- Standardize features.
- Predict whether each test instance has non-zero loss.
- Predict loss values for non-zero cases.
- Write raw predictions to `predictions/predictions.csv`.
- Convert raw predictions to a Kaggle submission file using `data/sampleSubmission.csv`.

### 5. Optional: Feature Selection

Run iterative feature-selection experiments:

```bash
python iterative_feature_selection.py
python iterative_feature_selection_last.py
```

These scripts save intermediate dictionaries in `data/`, such as selected feature sets and metric histories.

### 6. Optional: LIBLINEAR/LIBSVM Export

```bash
python liblinear_feature_elimination.py
```

This script exports selected features in a sparse text format for external linear-model tools.

## Outputs

Common outputs include:

| Output | Description |
|---|---|
| `predictions/predictions.csv` | Raw predicted loss values. |
| `predictions/predictions_competition_*.csv` | Kaggle-formatted submission files with `id,loss`. |
| `models/*.pkl` | Pickled fitted models. |
| `data/*generation_dict*.pkl` | Pickled feature-selection histories. |
| `data/LogisticRegression_*.pkl` | Saved cross-validation objects or prediction probabilities. |

