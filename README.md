# phase3project

phase3project_regression_models_lasso_ridge

Interpretation of the F1 score - model accuracy test
Class 0 → “Not churn” (customers who stayed)

- Precision = 0.88 → When the model predicts a customer will stay, it’s correct 88% of the time.
- Recall = 0.97 → It successfully identifies 97% of the customers who actually stayed.
- F1-score = 0.92 → Strong balance between precision and recall.
- Support = 1425 → Out of 1667 customers, 1425 really did not churn.
  The model is excellent at recognizing loyal customers.

Class 1 → “Churn” (customers who left)

- Precision = 0.57 → When the model predicts churn, only 57% of those predictions are correct.
- Recall = 0.21 → It only catches 21% of the actual churners — meaning most churners are missed.
- F1-score = 0.31 → Weak balance, showing poor performance on churn detection.
- Support = 242 → There were 242 actual churn cases.
  The model struggles to detect churners, which is critical because these are the customers you want to retain.

Overall

- Accuracy = 0.86 → 86% of predictions are correct, but this is misleading because churners are a minority.
- Macro avg (0.73 precision, 0.59 recall, 0.62 F1) → Treats both classes equally, showing the imbalance clearly.
- Weighted avg (0.83 precision, 0.86 recall, 0.83 F1) → Weighted by class size, looks better because non-churn dominates.

Business Interpretation

- The model is biased toward predicting “not churn” because most customers stay.
- It misses the majority of churners, which is dangerous: you’d fail to intervene with customers at risk of leaving.
- In churn prediction, recall for class 1 (churn) is often more important than overall accuracy, because missing churners means lost revenue.

Next Steps to Improve Churn Detection

- Resampling techniques: Oversample churners (SMOTE) or undersample non-churners.
- Class weights: Penalize misclassifying churn more heavily in the model.
- Alternative metrics: Focus on ROC-AUC, precision-recall curves, or recall for churn rather than accuracy.
- Feature engineering: Add stronger predictors of churn (usage patterns, complaints, tenure, etc.).
