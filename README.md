
# RetainIQ – Customer Churn Prediction & Retention Intelligence

## 1. Project Overview

RetainIQ is a machine learning-based customer churn prediction system designed to identify customers who may discontinue a telecom service.

The project analyses customer demographics, service subscriptions, contract details, tenure, billing information, and other available features to predict churn risk.

It also provides an interactive Streamlit dashboard for exploring customer data and testing individual customer profiles.

## 2. Objectives

- Analyse customer churn patterns using exploratory data analysis.
- Clean and preprocess customer data.
- Train and compare multiple machine learning models.
- Evaluate models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Provide interactive churn-risk predictions through a dashboard.
- Support data-driven customer retention planning.

## 3. Dataset

**Dataset:** Telco Customer Churn

**Source:** Kaggle  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains telecom customer information, including tenure, contract type, monthly charges, internet services, payment method, and churn status.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## 5. Methodology

1. Load and inspect the dataset.
2. Convert TotalCharges into numeric format and handle missing values.
3. Explore churn distribution and monthly charges.
4. Separate input features from the target variable.
5. Apply numerical scaling and categorical encoding.
6. Split the data into training and testing sets using an 80:20 ratio.
7. Train Logistic Regression and Random Forest models.
8. Evaluate and compare model performance.
9. Save the selected model and create an interactive dashboard.

## 6. Model Performance

The following results were obtained from the project's test set.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.81% | 78.28% |
| Precision | 50.43% | 58.29% |
| Recall | 78.34% | 63.90% |
| F1 Score | 61.36% | 60.97% |
| ROC-AUC | 84.13% | 83.36% |

Random Forest achieved higher accuracy and precision, while Logistic Regression achieved higher recall, F1 Score, and ROC-AUC.

The appropriate model depends on the business objective and the relative cost of missed churn cases and false alarms.

## 7. Dashboard Features

- Customer statistics and historical churn overview.
- Churn distribution visualisation.
- Churn analysis by contract type.
- Monthly charge comparison.
- Individual customer churn-risk estimation.
- Suggested retention actions based on estimated risk.

## 8. Project Files

- `train_model.py` – Data preprocessing, model training, and evaluation.
- `app.py` – Interactive Streamlit dashboard.
- `feature_analysis.py` – Model feature analysis.
- `model_comparison.csv` – Evaluation results.
- `feature_importance.csv` – Feature analysis results.
- `feature_importance.png` – Feature analysis visualisation.
- `retainiq_model.joblib` – Saved machine learning pipeline.
- `churn_distribution.png` – Churn distribution chart.
- `monthly_charges_churn.png` – Monthly charges analysis.
- `logistic_regression_confusion.png` – Logistic Regression evaluation.
- `random_forest_confusion.png` – Random Forest evaluation.

## 9. How to Run the Project

Install the required libraries:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

Train the models:

```bash
python train_model.py
```

Generate feature analysis:

```bash
python feature_analysis.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

Ensure the dataset CSV file is available in the project directory before running the scripts.

## 10. Limitations

- Predictions depend on the quality and representativeness of the available dataset.
- The dashboard uses default values for some customer attributes.
- Estimated churn probabilities are not guarantees of future behaviour.
- The system requires further validation before real-world business deployment.

## 11. Conclusion

RetainIQ demonstrates how machine learning can be applied to customer churn analysis. By comparing two classification models and presenting customer insights through an interactive dashboard, the project provides a foundation for understanding churn patterns and exploring potential retention strategies.

## 12. References

1. Kaggle – Telco Customer Churn Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Scikit-learn Documentation: https://scikit-learn.org/
3. Streamlit Documentation: https://docs.streamlit.io/
4. Pandas Documentation: https://pandas.pydata.org/docs/
