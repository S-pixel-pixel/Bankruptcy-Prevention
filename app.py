import streamlit as st
import pickle
import numpy as np

# Load trained Gradient Boosting model
with open("gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Bankruptcy Prediction App")
st.write("Enter company risk factors to predict bankruptcy.")

# Define feature names
features = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk"
]

# Collect user inputs
user_input = []
for feature in features:
    val = st.selectbox(
        f"{feature.replace('_', ' ').title()}",
        options=[0.0, 0.5, 1.0],
        index=0
    )
    user_input.append(val)

# Predict button
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    result = "🚨 Bankrupt" if prediction == 1 else "✅ Not Bankrupt"
    st.subheader(f"Prediction: {result}")