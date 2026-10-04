# 🚀 IIT Roorkee ML Hackathon 2026 — Coupon Acceptance Prediction

Machine learning project developed for the **IIT Roorkee AI & ML Certification Hackathon 2026** to predict whether a customer will accept a recommended coupon.

## 🎯 Business Problem

The objective is to predict whether a customer will accept or reject a recommended coupon based on customer demographics, travel context, weather, time, coupon type, and behavioral features.

This can help businesses improve targeted coupon recommendations and reduce ineffective offers.

## 🔍 What I Did

- Performed data cleaning and exploratory data analysis (EDA)
- Performed Chi-Square hypothesis testing on categorical features
- Selected relevant features for modeling
- Applied One-Hot Encoding
- Compared:
  - Logistic Regression — 68.27%
  - Random Forest — 74.49%
  - XGBoost — 74.64%
- Performed XGBoost hyperparameter tuning using RandomizedSearchCV
- Achieved **76.32% best 3-fold cross-validation accuracy**
- Saved the trained XGBoost model and encoder using Joblib
- Built a **FastAPI REST API** for real-time prediction
- Manually deployed the API on **AWS EC2 (Ubuntu)**
- Tested the `/predict` endpoint successfully

## 🌐 Model Deployment

```text
XGBoost Model
     ↓
FastAPI
     ↓
Uvicorn
     ↓
AWS EC2 (Ubuntu)
     ↓
/predict
