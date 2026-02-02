# 🏦 Bank Customer Churn Prediction (ANN)

This project predicts whether a bank customer is likely to **leave (churn)** or **stay** using an **Artificial Neural Network (ANN)** built with TensorFlow and deployed using **Streamlit**.

It is an end-to-end Machine Learning project covering:
- Data preprocessing
- Feature encoding & scaling
- ANN model training
- Model serialization
- Web app deployment

---

## 📌 Problem Statement
Customer churn is a major concern for banks. The goal of this project is to predict whether a customer will exit the bank based on demographic and financial features.

---

## 🚀 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas, NumPy**
- **Streamlit**
- **Pickle**
- **Git & GitHub**

---

## 🧠 Model Used
- Artificial Neural Network (ANN)
- Input layer → Hidden layers (ReLU) → Output layer (Sigmoid)
- Loss: Binary Crossentropy  
- Optimizer: Adam  

---

## 🖥️ Streamlit App Features
- User-friendly input form
- Real-time churn prediction
- Probability score visualization
- Clean UI


## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
