# Reports and Evaluation Results

This folder contains project reports and generated evaluation outputs for the **Multi-Label Email Risk Detection Using Machine Learning** project.

The files in this folder include the final report and CSV files containing model performance results generated during evaluation.

## Folder Structure

```text
reports/
│
├── final_report.pdf
│
├── all_model_comparison.csv
├── nn_overall_evaluation.csv
├── nn_per_class_evaluation.csv
├── neural_network_binary_cross_entropy_loss.csv
│
└── README.md
```

## Report Files

The final report contains:

- Project motivation
- Related work
- Dataset description
- Data preprocessing
- Feature engineering
- Model development
- Experimental results
- Error analysis
- Limitations
- Future improvements

## Evaluation CSV Files

The generated CSV files contain model evaluation results.



The report files and evaluation outputs can be regenerated using:

```
notebooks/04_evaluation.ipynb
```

The evaluation notebook loads trained models, generates predictions, calculates performance metrics, and exports the results as CSV files.


```
