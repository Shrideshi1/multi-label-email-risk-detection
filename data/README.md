# Dataset Information

This folder contains information about the datasets used for the **Multi-Label Email Risk Detection** project.

Due to dataset size limitations and licensing restrictions, the actual datasets are not stored in this repository.

## Dataset Sources

The project uses publicly available datasets, including:

* Phishing email datasets
* Enron email corpus
* Spam email datasets
* Financial text datasets
* Legal document datasets

## Data Organization

After downloading and preparing the datasets, the expected structure is:

```
data/
│
├── raw/
│   └── README.md
│
├── processed/
│   └── README.md
│
├── combined_dataset.csv
│
└── README.md
```


---

## Data Processing

The data preparation pipeline includes:

- Text cleaning and preprocessing
- Dataset integration
- Label transformation for multi-label classification
- Risk category mapping
- Synthetic data integration
- Feature preparation for machine learning models

The preprocessing pipeline can be reproduced using:
notebooks/01_data_preprocessing.ipynb


Feature engineering is performed using:
notebooks/02_feature_engineering.ipynb


---

## Risk Categories

The processed dataset contains multi-label risk annotations for:

1. Financial Risk  
2. Credential Risk  
3. Customer Information Risk  
4. Proprietary Risk  
5. Legal Risk  
6. Attachment Risk  
7. Phishing/Spam Risk  

Each email can contain one or more risk categories simultaneously.

---

## Combined Dataset

The `combined_dataset.csv` file represents the final unified dataset created by combining multiple public email and text datasets.

The dataset contains:

- Email text
- Source information
- Multi-label risk categories
- Metadata features used for classification

The combined dataset was created through:

- Data cleaning
- Label normalization
- Risk category mapping
- Synthetic data integration
- Dataset consolidation

---

## Reproducibility

The complete dataset preparation workflow can be reproduced by running:
notebooks/01_data_preprocessing.ipynb

The generated processed data is then used for model training and evaluation.
