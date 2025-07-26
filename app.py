import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.feature_selection import f_regression

# ────────────────────────────────────────────────────
# 1) CACHE THE DATA LOAD
# ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel(
        "Indian Agriculture Dataset.xlsx",
        engine="openpyxl",
        usecols=["State","District","Crop","Season","Year","Yield"]
    ).dropna()
    for c in ['State','Crop','Season','District']:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df

# ────────────────────────────────────────────────────
# 2) CACHE THE MODEL LOAD
# ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("yield_xgb_pipeline.pkl")

# ────────────────────────────────────────────────────
# 3) CACHE THE CASCADING OPTION BUILD
# ────────────────────────────────────────────────────
@st.cache_data
def build_options(df):
    # state → districts
    state2d = df.groupby("State")["District"] \
                .unique().apply(sorted).to_dict()
    # (state,district) → seasons
    sd2s = df.groupby(["State","District"])["Season"] \
              .unique().apply(sorted).to_dict()
    # (state,district,season) → crops
    sds2c = df.groupby(["State","District","Season"])["Crop"] \
               .unique().apply(sorted).to_dict()
    return state2d, sd2s, sds2c

# ────────────────────────────────────────────────────
# 4) REDefine your custom encoder for unpickling
# ────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────
# 5) MAIN APP
# ────────────────────────────────────────────────────
st.set_page_config(page_title="Yield Predictor", layout="centered")

# Load once
df     = load_data()
pipeline = load_model()
state2d, sd2s, sds2c = build_options(df)

# UI Header
st.markdown("<h1 style='text-align:center;'>🌾 Agricultural Yield Predictor</h1>",
            unsafe_allow_html=True)
st.write("Predict the **crop yield (t/ha)** for any valid Crop–Season–District combination in India.")

# Cascading dropdowns
col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_state = st.selectbox("Select State", list(state2d.keys()))
with col2:
    districts = state2d.get(selected_state, [])
    selected_district = st.selectbox("Select District", districts)
with col3:
    seasons = sd2s.get((selected_state, selected_district), [])
    selected_season = st.selectbox("Select Season", seasons)
with col4:
    crops = sds2c.get((selected_state, selected_district, selected_season), [])
    selected_crop = st.selectbox("Select Crop", crops)

# Predict
if st.button("Predict Yield"):
    row = {
        'Crop': selected_crop,
        'Season': selected_season,
        'District': selected_district,
        'crop_season': f"{selected_crop}_{selected_season}"
    }
    input_df = pd.DataFrame([row])

    # Defensive check
    encoder = pipeline.named_steps['fval']
    missing = [c for c in encoder.cols if row[c] not in encoder.maps_[c]]
    if missing:
        st.error(f"❌ Cannot predict: missing categories for {missing}")
    else:
        raw_pred = pipeline.predict(input_df)[0]
        pred = max(0.0, raw_pred)
        st.success(f"🌾 Predicted Yield: **{pred:.2f} t/ha**")
