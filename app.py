import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np


# Load model and tools
model = tf.keras.models.load_model('model.h5')

with open("one_hot.pkl", "rb") as file:
    one_hot = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =============== TITLE ===============
st.title("Bank Customer Churn Prediction")
st.write("Fill the details and click Predict")

# =============== INPUTS ===============
credit_score = st.number_input("Credit Score", 300, 850, 600)
geography = st.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 95, 40)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 0.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# =============== PREDICT BUTTON ===============
if st.button("Predict Churn"):

    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary
    }

    df = pd.DataFrame([input_data])

    # Encoding
    geo_encoded = one_hot.transform(df[["Geography"]])
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot.get_feature_names_out(["Geography"])
    )

    df["Gender"] = label_encoder.transform(df["Gender"])

    final_df = pd.concat([df.drop("Geography", axis=1), geo_df], axis=1)

    # Scaling
    scaled_input = scaler.transform(final_df)

    # Prediction
    probability = model.predict(scaled_input, verbose=0)[0][0]

    # Output
    st.markdown("---")
    if probability > 0.5:
        st.error("Customer will LEAVE the bank")
    else:
        st.success("Customer will STAY")

    st.write(f"**Churn Probability: {probability:.2%}**")
    st.progress(float(probability))   # ✅ FIXED