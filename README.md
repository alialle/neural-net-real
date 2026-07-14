# Neural Net Playground

A reusable, from-scratch neural network training pipeline built with scikit-learn. Designed to plug in personal CSV data (e.g. SQL exports) and train small regression or classification models with proper train/test evaluation and checkpointing.

## What's in this repo

- `nn_template.py` — Regression template (predicts a **number**) using `MLPRegressor`
- `nn_template_classifier.py` — Classification template (predicts a **category**) using `MLPClassifier`
- `your_data.csv` — Your local dataset (excluded from git via `.gitignore` — see note below)

## Features

- Configurable input/output columns — no hardcoded topic, just rename to fit your data
- 80/20 train/test split
- 2 hidden layers, ReLU activation
- Epoch-by-epoch checkpointing: automatically keeps the model version with the **lowest test loss**, not just the last epoch trained
- Clear side-by-side print of training loss vs test loss (to catch overfitting)
- Ready-to-edit pr- Ready-to-edit pat the bottom of each script

## Setup

```bash
pip3 install pandas scikit-learn
```

## Usage

1. Export your data from SQL (or Excel) as a CSV file
2. Save it in this folder as `your_data.csv`
3. Open either script and edit these lines to match your CSV's column names:
```python
   INPUT_COLUMNS   INPUT_COLUMNS   INture_2", "feature_3"]
   OUTPUT_COLUMN = "target_value"   # or "target_category" for the classifier
```
4. Run it:
```bash
   python3 nn_template.py
   python3 nn_template_classifier.py
```

## Roadmap / ideas for later

- [ ] Add cross-validation instead of a single train/test split
- [ ] Add a config file- [ ] Add a config file- [ ] Add a config file- [ ] Add a config file- [ ] Add a config epochs
- [ ] Add support f- [ ] Add support f- [ ] Ars as a config option

## Notes

This project is for personal learning — using it to build intuition around neural networks (epochs, loss, overfitting, train/test splits) hands-on with the TensorFlow Playground
