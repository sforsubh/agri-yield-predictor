# agri-yield-predictor
A Streamlit-based web app for predicting agricultural crop yield using an XGBoost regression pipeline trained on Indian crop data.
# Agricultural Yield Prediction App

This is a Streamlit web application (ongoing project) that predicts the **crop yield (in tonnes/hectare)** based on the selected **crop**, **season**, and **district**. It uses a supervised machine learning model (XGBoost) trained on real Indian agriculture data.

## 🔍 Features

- Predicts yield for multiple crops across Indian districts
- Interactive dropdowns for crop, district, and season
- Pretrained XGBoost pipeline with supervised F-value encoding
- Fully deployable on Streamlit Cloud or locally (ongoing)

## 🧠 Machine Learning

- Model: XGBoost Regressor
- Preprocessing: Custom F-value encoder (supervised)
- Evaluation Metric: RMSE, R-squared
- Input features: Crop, Season, District, Crop_Season
- Output: Predicted yield in **tonnes/hectare**
