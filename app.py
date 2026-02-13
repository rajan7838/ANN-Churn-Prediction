import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model + encoders
model = load_model("models/best_model.h5")

scaler = pickle.load(open("models/scaler.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
onehot = pickle.load(open("models/onehot_encoder.pkl", "rb"))

st.title("Customer Churn Prediction App")

# Inputs
credit_score = st.number_input("Credit Score", 300, 900)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure", 0, 10)
balance = st.number_input("Balance")
products = st.number_input("Num Of Products", 1, 4)
credit_card = st.selectbox("Has Credit Card?", [0, 1])
active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary")

if st.button("Predict Churn"):

    # Encode Gender
    gender_encoded = label_encoder.transform([gender])[0]

    # Encode Geography safely
    geo_encoded = onehot.transform([[geography]])
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()

    # Combine all inputs in correct order
    input_data = np.array([[credit_score,
                            gender_encoded,
                            age,
                            tenure,
                            balance,
                            products,
                            credit_card,
                            active,
                            salary]])

    # Merge geo + numerical features
    final_input = np.concatenate([geo_encoded, input_data], axis=1)

    # Scale input
    final_scaled = scaler.transform(final_input)

    # Prediction
    prediction = model.predict(final_scaled)[0][0]

    if prediction > 0.5:
        st.error("Customer will EXIT (Churn)")
    else:
        st.success("Customer will STAY")