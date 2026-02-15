# 📌 Breast Cancer Classification – ML + Streamlit Deployment

## 🔗 Project Overview
This project builds multiple Machine Learning classification models to predict whether a tumor is Malignant or Benign using the Breast Cancer dataset.

The goal is to demonstrate an end-to-end ML workflow:
- Data preprocessing
- Model training
- Model evaluation
- Web app development
- Cloud deployment

---

## 🎯 Problem Statement
Breast cancer is one of the most common cancers worldwide. Early detection helps improve survival rates.

This project predicts:
- Malignant (Cancerous)
- Benign (Non-Cancerous)

using machine learning classification models.

---

## 📊 Dataset Description

### Dataset Details
- Total Instances: 569
- Total Features: 30+ numerical features
- Target Column: Diagnosis  
  - M → Malignant  
  - B → Benign  

### Example Features
- Radius Mean  
- Texture Mean  
- Perimeter Mean  
- Area Mean  
- Smoothness Mean  
- Compactness Mean  

---

## 🤖 Machine Learning Models Used

Six classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## 📈 Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9884 | 1.0000 | 1.0000 | 0.9688 | 0.9841 | 0.9753 |
| XGBoost | 0.9884 | 1.0000 | 1.0000 | 0.9688 | 0.9841 | 0.9753 |
| Random Forest | 0.9884 | 0.9983 | 1.0000 | 0.9688 | 0.9841 | 0.9753 |
| Decision Tree | 0.9535 | 0.9566 | 0.9118 | 0.9688 | 0.9394 | 0.9028 |
| Naive Bayes | 0.9535 | 0.9977 | 1.0000 | 0.8750 | 0.9333 | 0.9026 |
| KNN | 0.9535 | 0.9959 | 0.9667 | 0.9063 | 0.9355 | 0.9003 |

---

## 🧠 Observations

### Logistic Regression
- Very high accuracy and perfect AUC  
- Stable and reliable baseline model  

### XGBoost
- Matches Logistic Regression performance  
- Very strong ensemble learning capability  

### Random Forest
- Very strong performance  
- Slightly lower AUC than top models  

### Decision Tree
- Fast and interpretable  
- Lower overall performance  

### Naive Bayes
- Very high precision  
- Lower recall (misses some malignant cases)  

### KNN
- Balanced performance  
- Slightly lower than top ensemble models  
