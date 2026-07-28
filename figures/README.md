
# Figures and Visualizations

All generated evaluation figures are stored in the `figures/` folder.

The visualizations include:

## Model Comparison
- `advanced_model_comparison.png` - Advanced comparison of evaluated models
- `overall_model_performance.png` - Overall model performance comparison
- `neural_network_vs_xgboost_comparison.png` - Neural network vs XGBoost comparison
- `hamming_loss_comparison.png` - Hamming loss comparison across models

## Error Analysis
- `false_negative_comparison_all_models.png` - False negative comparison across models
- `false_positive_comparison_all_models.png` - False positive comparison across models
- `total_prediction_errors_across_models.png` - Total prediction errors across models
- `overall_error_rate_comparison.png` - Overall error rate comparison

## Per-Class Risk Category Evaluation
- `per_class_f1_across_models.png` - F1-score comparison by risk category
- `per_class_precision_across_models.png` - Precision comparison by risk category
- `per_class_recall_across_models.png` - Recall comparison by risk category
- `logistic_regression_performance_by_risk_category.png`
- `random_forest_performance_by_risk_category.png`
- `svm_performance_by_risk_category.png`
- `xgboost_performance_by_risk_category.png`
- `neural_network_performance_by_risk_category.png`

## Model-Specific Analysis
- `logistic_regression_false_negative_analysis.png`
- `random_forest_false_negative_analysis.png`
- `svm_false_negative_analysis.png`
- `xgboost_false_negative_analysis.png`
- `neural_network_false_negative_analysis.png`

## Feature Importance Analysis
- `xgboost_feature_importance.png` - Important features identified by XGBoost
- `random_forest_feature_importance.png` - Important features identified by Random Forest

## Neural Network Training Analysis
- `neural_network_binary_cross_entropy_loss.png` - Neural network training loss curve

## Generation

Figures are generated from the analysis and evaluation notebooks:

```text
notebooks/
├── 02_feature_engineering.ipynb
└── 04_evaluation.ipynb
```

## Storage

Generated figures are stored here to support project documentation, analysis, and reporting.
