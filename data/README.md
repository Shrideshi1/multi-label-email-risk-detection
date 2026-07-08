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
├── raw/
│   └── Original datasets
│
└── processed/
    └── combined_dataset.csv
```

## Processing

The data preparation pipeline includes:

* Text cleaning and preprocessing
* Dataset integration
* Label transformation for multi-label classification
* Feature preparation for machine learning models

## Risk Categories

The processed dataset contains labels for:

* Spam
* Phishing
* Confidential Information
* Financial Risk
* Legal Risk

For details on data preparation, see:

`notebooks/01_data_preparation.ipynb`
