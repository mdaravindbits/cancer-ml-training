# breast cancer-ml-training
a. Problem Statement

The objective of this project is to build Machine Learning classification models to predict whether a breast tumor is Malignant (M) or Benign (B) using diagnostic measurement features. Multiple ML models are trained and evaluated to identify the best performing model for accurate medical diagnosis support.

b. Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset contains features computed from digitized images of breast mass cell nuclei.

First Column: ID Number (not used for training)

Second Column: Diagnosis

M → Malignant

B → Benign

Remaining Columns: Numeric diagnostic features

The dataset was split into:

Training Data: 85%

Testing Data: 15%

c. Models Used

The following six Machine Learning models were implemented:

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

Comparison Table of Evaluation Metrics
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9884	1.0000	1.0000	0.9688	0.9841	0.9753
Decision Tree	0.9535	0.9566	0.9118	0.9688	0.9394	0.9028
KNN	0.9535	0.9959	0.9667	0.9063	0.9355	0.9003
Naive Bayes	0.9535	0.9977	1.0000	0.8750	0.9333	0.9026
Random Forest (Ensemble)	0.9884	0.9983	1.0000	0.9688	0.9841	0.9753
XGBoost (Ensemble)	0.9884	1.0000	1.0000	0.9688	0.9841	0.9753
Observations on Model Performance
ML Model Name	Observation about model performance
Logistic Regression	Achieved very high accuracy and perfect precision. Shows strong performance even as a simple linear model.
Decision Tree	Lower accuracy compared to ensemble models. More prone to overfitting and less stable.
KNN	Good performance but slightly lower recall and F1 score compared to top models. Sensitive to data distribution.
Naive Bayes	Very high precision but lower recall, meaning some malignant cases were missed.
Random Forest (Ensemble)	Excellent performance with very high accuracy and strong generalization ability.
XGBoost (Ensemble)	One of the best performing models with near perfect metrics across all evaluation measures.
