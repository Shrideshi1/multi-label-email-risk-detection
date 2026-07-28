# Multi-Label Email Risk Detection Using Machine Learning

## Overview

This project develops a **multi-label email risk detection system** that identifies multiple security and privacy risks within email content using machine learning and natural language processing.

Unlike traditional email filtering systems that assign a single label, this approach allows an email to be classified into multiple risk categories simultaneously, including phishing, spam, confidential information, financial risks, and legal risks.

The project evaluates traditional machine learning models and explores transformer-based approaches for improved text understanding.
---
### Running the Demo Application

1. Install cloudflared:
https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

2. Start Flask app:

python app.py

3. Create a public tunnel:

cloudflared tunnel --url http://localhost:5000

---
## Development Environment

The initial development, experimentation, and model training for this project were conducted using Google Colab. The Colab notebooks contain the original workflow, including data preprocessing, feature engineering, model training, and evaluation steps.

For reference, the Google Colab project files are available here:

[Google Colab Development Folder](https://drive.google.com/drive/folders/1SLhiH7VulyiPZ7L846kJBEbN1pa4c4zU?usp=sharing)

Note: The repository contains the finalized project files and results. The Google Colab folder is provided as a reference to demonstrate the initial development and experimentation process.

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

```text
multi-label-email-risk-detection/
│
├── data/
│   ├── combined_dataset.csv              # Unified multi-label email risk dataset
│   ├── raw/                              # Original source datasets
│   └── processed/                        # Cleaned and transformed datasets
│
├── notebooks/
│   ├── 01_data_preparation.ipynb         # Data loading and preprocessing
│   ├── 02_feature_engineering.ipynb      # TF-IDF and metadata feature extraction
│   ├── 03_model_training.ipynb            # Machine learning model training
│   ├── 04_evaluation.ipynb                # Model evaluation and analysis
│   └── demo.ipynb                         # Project demonstration notebook
│
├── models/
│   ├── distilbert_email_risk/             # Fine-tuned DistilBERT model files
│   ├── svm_model.pkl                      # Trained SVM model
│   ├── mlp_model.pkl                      # Trained MLP classifier
│   ├── xgboost_model.pkl                  # Trained XGBoost model
│   ├── logistic_regression_model.pkl      # Trained Logistic Regression model
│   ├── random_forest_model.pkl            # Trained Random Forest model
│   └── multi_label_email_risk_mlp.keras   # Saved neural network model
│
├── reports/                               # Additional reports
│   └── final_report.pdf                   # Final project report 
│
├── figures/
│   ├── overall_model_performance.png      # Model comparison visualizations
│   ├── hamming_loss_comparison.png        # Multi-label error analysis
│   └── ...                                # Additional evaluation figures
│
├── app.py                                 # Application entry point
├── cloudflared/                           # Cloudflare tunnel configuration/files
├── requirements.txt                       # Python package dependencies
│
└── README.md                              # Project documentation
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
