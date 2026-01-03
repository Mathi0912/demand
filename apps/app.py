import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Demand Prediction (XGBoost)", layout="centered")
st.title("📊 Demand Prediction – Use Case 1")

st.write("Enter values to predict customer demand")

# -------------------------------
# TRAIN SAMPLE MODEL (INLINE)
# -------------------------------
np.random.seed(42)

train_df = pd.DataFrame({
    "Price": np.random.uniform(50, 500, 300),
    "Discount": np.random.uniform(0, 0.5, 300),
    "Inventory_Level": np.random.randint(50, 1000, 300),
    "Promotion": np.random.randint(0, 2, 300),
    "Competitor_Pricing": np.random.uniform(50, 500, 300),
    "Seasonality": np.random.randint(0, 2, 300),
    "Epidemic": np.random.randint(0, 2, 300)
})

train_df["Demand"] = (
    0.5 * train_df["Inventory_Level"]
    - 0.4 * train_df["Price"]
    + 250 * train_df["Discount"]
    + 180 * train_df["Promotion"]
    - 0.3 * train_df["Competitor_Pricing"]
    + 120 * train_df["Seasonality"]
    - 150 * train_df["Epidemic"]
)

X = train_df.drop("Demand", axis=1)
y = train_df["Demand"]

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X, y)

# -------------------------------
# USER INPUTS
# -------------------------------
st.sidebar.header("Input Features")

price = st.sidebar.number_input("Price", min_value=1.0, value=100.0)
discount = st.sidebar.slider("Discount (0–0.5)", 0.0, 0.5, 0.1)
inventory = st.sidebar.number_input("Inventory Level", min_value=0, value=500)
promotion = st.sidebar.selectbox("Promotion (0 = No, 1 = Yes)", [0, 1])
competitor_price = st.sidebar.number_input("Competitor Pricing", min_value=1.0, value=95.0)
seasonality = st.sidebar.selectbox("Seasonality (0 = No, 1 = Yes)", [0, 1])
epidemic = st.sidebar.selectbox("Epidemic (0 = No, 1 = Yes)", [0, 1])

input_df = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory_Level": [inventory],
    "Promotion": [promotion],
    "Competitor_Pricing": [competitor_price],
    "Seasonality": [seasonality],
    "Epidemic": [epidemic]
})

st.subheader("User Input")
st.write(input_df)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Demand"):
    pred = model.predict(input_df)[0]
    st.success(f"📈 Predicted Demand: {int(pred)}")
