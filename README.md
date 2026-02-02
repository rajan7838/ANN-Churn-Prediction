ANN-Based Customer Churn Prediction
📌 Project Overview

Customer churn is a critical business problem where companies lose customers over time. This project builds an Artificial Neural Network (ANN) model to predict whether a bank customer is likely to churn (leave the bank) based on demographic, financial, and account-related features.

The goal is to help businesses identify at-risk customers early and take preventive actions.

🎯 Problem Statement

Given historical customer data, predict whether a customer will churn (1) or stay (0).

This is a binary classification problem solved using a supervised learning approach with an Artificial Neural Network.

🧠 Solution Approach

Data preprocessing (encoding categorical variables, feature scaling)

Build and train an ANN using Keras/TensorFlow

Evaluate model performance

Save the trained model for reuse

(Optional) Deploy prediction using a Streamlit web app

📂 Dataset Description

The dataset contains customer information such as:

Credit Score

Geography

Gender

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

Target Variable:

Exited (1 = Customer churned, 0 = Customer stayed)

⚙️ Tech Stack

Programming Language: Python

Libraries: NumPy, Pandas, Scikit-learn

Deep Learning: TensorFlow / Keras

Model Persistence: Pickle / H5

Version Control: Git & GitHub

🏗️ Model Architecture (ANN)

Input Layer: Customer features

Hidden Layers: Fully connected Dense layers with ReLU activation

Output Layer: 1 neuron with Sigmoid activation

Loss Function: Binary Crossentropy
Optimizer: Adam
Evaluation Metric: Accuracy

📊 Model Workflow
Raw Data
   ↓
Data Cleaning & Encoding
   ↓
Feature Scaling
   ↓
ANN Model Training
   ↓
Model Evaluation
   ↓
Model Saving & Prediction
🚀 How to Run the Project

Clone the repository

git clone https://github.com/rajan7838/ANN-Churn-Prediction.git
cd ANN-Churn-Prediction

Install dependencies

pip install -r requirements.txt

Run the application (if Streamlit is used)

streamlit run app.py
✅ Results

The ANN model successfully learns non-linear relationships in customer data and predicts churn with good accuracy.

This model can be used as a decision-support tool for customer retention strategies.

📌 Key Learnings

Practical implementation of Artificial Neural Networks

Importance of feature scaling for ANN performance

Model training, evaluation, and persistence

Structuring ML projects professionally using GitHub

🔮 Future Improvements

Compare ANN with Logistic Regression and XGBoost

Hyperparameter tuning

Model monitoring and retraining

Full MLOps pipeline integration

👤 Author

A. Rajan
GitHub: https://github.com/rajan7838
