import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.feature_selection import f_regression

# -----------------------------
# Re-define your custom encoder
# -----------------------------
class FValueEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.maps_ = {}
    def fit(self, X, y):
        for col in self.cols:
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_cat = enc.fit_transform(X[[col]])
            F_vals, _ = f_regression(X_cat, y)
            cats = enc.categories_[0]
            self.maps_[col] = dict(zip(cats, F_vals))
        return self
    def transform(self, X):
        X2 = X.copy()
        out = pd.DataFrame(index=X2.index)
        for col in self.cols:
            out[f"{col}_F"] = X2[col].map(self.maps_[col]).fillna(0)
        return out

# -----------------------------
# Load your trained pipeline
# -----------------------------
pipeline = joblib.load("yield_xgb_pipeline.pkl")

# -----------------------------
# Load and clean dataset for dropdowns
# -----------------------------
df = pd.read_csv("Indian Agriculture Dataset.csv")[
    ['State','Crop','Season','District','Yield']
].dropna()

for c in ['State','Crop','Season','District']:
    df[c] = df[c].astype(str).str.strip().str.lower()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Yield Predictor", layout="centered")
st.markdown("<h1 style='text-align:center;'>🌾Agricultural Yield Predictor</h1>",
            unsafe_allow_html=True)
st.write("Predict the **crop yield (t/ha)** for any valid Crop–Season–District combination in Indian sub-continent.")

# 1️⃣ State dropdown (for filtering)
all_states = sorted(df['State'].unique())
selected_state = st.selectbox("Select State", all_states)

# 2️⃣ District filtered by State
districts = sorted(df[df['State']==selected_state]['District'].unique())
selected_district = st.selectbox("Select District", districts)

# 3️⃣ Season filtered by State+District
seasons = sorted(
    df[(df['State']==selected_state) & 
       (df['District']==selected_district)]['Season']
    .unique()
)
selected_season = st.selectbox("Select Season", seasons)

# 4️⃣ Crop filtered by State+District+Season
crops = sorted(
    df[(df['State']==selected_state) &
       (df['District']==selected_district) &
       (df['Season']==selected_season)]['Crop']
    .unique()
)
selected_crop = st.selectbox("Select Crop", crops)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Yield"):
    # Build the single-row DataFrame that your pipeline expects
    row = {
        'Crop': selected_crop,
        'Season': selected_season,
        'District': selected_district,
        'crop_season': f"{selected_crop}_{selected_season}"
    }
    input_df = pd.DataFrame([row])

    # ➊ Defensive check: ensure encoder saw all categories
    encoder = pipeline.named_steps['fval']
    missing = [
        col for col in encoder.cols
        if row[col] not in encoder.maps_[col]
    ]
    if missing:
        st.error(f"❌ Cannot predict: missing categories for {missing}")
        st.stop()

    # ➋ Predict and clamp at zero
    raw_pred = pipeline.predict(input_df)[0]
    pred = max(0.0, raw_pred)

    st.success(f"🌾Predicted Yield: **{pred:.2f} t/ha**")
