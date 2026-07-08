# Multi-Label Email Risk Detection Using Machine Learning

## Overview

This project develops a **multi-label email risk detection system** that identifies multiple security and privacy risks within email content using machine learning and natural language processing.

Unlike traditional email filtering systems that assign a single label, this approach allows an email to be classified into multiple risk categories simultaneously, including phishing, spam, confidential information, financial risks, and legal risks.

The project evaluates traditional machine learning models and explores transformer-based approaches for improved text understanding.

---

## Project Goals

* Build a multi-label classification pipeline for email risk detection.
* Detect multiple risk categories within a single email.
* Compare traditional ML models with transformer-based NLP models.
* Handle challenges such as class imbalance and limited labeled data.
* Evaluate model performance using multi-label classification metrics.

---

## Risk Categories

| Risk Category | Description                                                |
| ------------- | ---------------------------------------------------------- |
| Spam          | Unwanted or unsolicited email messages                     |
| Phishing      | Emails designed to steal information or credentials        |
| Confidential  | Potential exposure of sensitive or proprietary information |
| Financial     | Fraud, payment, or financial manipulation risks            |
| Legal         | Contractual, compliance, or legal-related risks            |

---

## Methodology

### Data Processing

* Dataset collection and integration
* Text cleaning and preprocessing
* Multi-label encoding

### Feature Engineering

* TF-IDF text representation
* Text-based feature extraction
* Email metadata analysis

### Models

**Traditional Machine Learning**

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost

**Deep Learning / Transformer Models**

* Neural Networks
* DistilBERT
* RoBERTa

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Hamming Loss

---

## Repository Structure

```
multi-label-email-risk-detection/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── data/
│   └── README.md
│
├── figures/
├── models/
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses publicly available email and text datasets, including:

* Phishing email datasets
* Enron email corpus
* Spam datasets
* Financial text datasets
* Legal document datasets

Due to dataset size and licensing restrictions, datasets are not included in this repository.

---

## Technologies

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* PyTorch
* Transformers

---

## Reproducibility

The notebooks provide an end-to-end workflow:

1. Data preparation
2. Feature engineering
3. Model training
4. Model evaluation

Large datasets and trained models are stored separately due to repository size limitations.

---

## Future Work

* Fine-tune transformer-based language models.
* Add email metadata features.
* Develop real-time email risk scoring.
* Deploy an interactive risk detection application.

---

## Course Project

**DATA 780: Machine Learning**

**Project:** Multi-Label Email Risk Detection
