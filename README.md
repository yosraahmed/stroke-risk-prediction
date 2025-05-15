# 🧠 Stroke Risk Prediction Using Machine Learning in R

This project uses patient medical data to predict stroke risk and assist early detection efforts. Three classification models — Logistic Regression, Decision Tree, and Random Forest — were built and evaluated to identify the most accurate method for predicting stroke occurrences.

---

## 📌 Project Objective

To develop a predictive model that identifies individuals at higher risk of stroke using healthcare data, enabling better preventive care and resource allocation.

---

## 🧠 Key Features

- Data preprocessing: handling missing values, encoding categorical variables
- Feature engineering and exploratory analysis (age, gender, hypertension, heart disease, etc.)
- Trained and evaluated three ML models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest
- Used accuracy, AUC, and F1-score for evaluation
- Random Forest achieved the best performance (76% accuracy)

---

## 🧰 Tools & Libraries

- **R**
- `caret`, `mice`, `ggplot2`, `rpart`, `randomForest`, `pROC`

---

## 📁 Files Included

- `stroke_prediction.R` – R script containing full model development pipeline
- `/screenshots/` – Visuals of model output and evaluation

---

## 📝 Note

> This is an academic machine learning project based on healthcare data. It is not intended for medical use.
