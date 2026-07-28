# Trained Models

This folder contains trained machine learning models generated for the **Multi-Label Email Risk Detection Using Machine Learning** project.

Due to file size limitations, some large model artifacts (such as transformer checkpoints) may be stored externally using Google Colab/Google Drive storage.

---

## Available Models

### Traditional Machine Learning Models

| Model                        | File                            |
| ---------------------------- | ------------------------------- |
| Logistic Regression          | `logistic_regression_model.pkl` |
| Support Vector Machine (SVM) | `svm_model.pkl`                 |
| Random Forest                | `random_forest_model.pkl`       |
| XGBoost                      | `xgboost_model.pkl`             |

### Deep Learning Models

| Model                        | File                               |
| ---------------------------- | ---------------------------------- |
| Multi-Layer Perceptron (MLP) | `mlp_model.pkl`                    |
| Neural Network (Keras)       | `multi_label_email_risk_mlp.keras` |

### Transformer Models

| Model      | Location                         |
| ---------- | -------------------------------- |
| DistilBERT | `distilbert_email_risk/`         |


---

## Model Artifacts

The trained models may include:

* TF-IDF vectorizers
* Machine learning classifiers
* Neural network weights
* Transformer model checkpoints
* Supporting files required for evaluation

---

## Usage

The models are trained using:

```
notebooks/03_model_training.ipynb
```

The saved models are loaded and evaluated using:

```
notebooks/04_evaluation.ipynb
```

---

## Storage Notes

* Smaller machine learning models (`.pkl` files) are included in this folder.
* Larger artifacts, such as transformer checkpoints, may be stored externally due to GitHub storage limitations.
* Dataset files are not included due to licensing and privacy restrictions.

---

## Reproducibility

To recreate these models, run the training notebook and follow the preprocessing and feature engineering pipeline described in the project repository.
