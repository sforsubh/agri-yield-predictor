# Agriculture Yield Predictor

## Overview

This project presents a machine learning model to accurately predict agricultural crop yields at a granular level using historical data from India. The model incorporates advanced data preprocessing, feature engineering, and gradient boosting techniques to capture complex relationships between crop types, seasons, districts, and temporal trends.

The goal is to empower farmers, agricultural cooperatives, policymakers, and agri-businesses with reliable yield forecasts to optimize resource allocation, reduce financial risks, and improve food supply chain management.

---

## Features

- Data cleaning and handling of missing values and outliers
- Feature engineering including ANOVA F-value encoding and crop-season interaction features
- Handling of nonlinear relationships using tree-based models (XGBoost, Random Forest)
- Hyperparameter tuning with RandomizedSearchCV
- Model interpretability with SHAP values
- Comprehensive evaluation using R², RMSE, and MAE metrics

---

## Dataset

The dataset contains historical agricultural statistics with features such as:

- State, District, Crop, Season, Year
- Area cultivated, Production, and Yield values

