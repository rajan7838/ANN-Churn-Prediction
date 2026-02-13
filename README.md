# 🔥 ANN Customer Churn Prediction (MLOps Project)

An end-to-end **Artificial Neural Network (ANN)** based Machine Learning project to predict whether a bank customer will **exit (churn)** or stay.

This project follows an **MLOps-level modular pipeline structure** including:

✅ Data Ingestion  
✅ Data Preprocessing  
✅ ANN Model Training  
✅ Model Evaluation  
✅ Model Saving & Pushing  
✅ Streamlit Web Deployment  
✅ CI/CD with GitHub Actions  
✅ Docker Container Support  

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges in banking and subscription businesses.

This project predicts:

- **Will the customer leave the bank?**
- **Will the customer stay?**

Using an ANN classification model trained on customer demographics and account details.

---

## 📂 Project Folder Structure (MLOps Standard)

ANN-Churn-Prediction/
│
├── .github/workflows/
│ └── main.yml # CI/CD Pipeline
│
├── artifacts/ # Pipeline Outputs
│ ├── raw_data.csv
│ ├── train.csv
│ ├── test.csv
│ ├── scaler.pkl
│ ├── label_encoder.pkl
│ ├── onehot_encoder.pkl
│ └── model.h5
│
├── data/
│ └── Churn_Modelling.csv # Original Dataset
│
├── models/
│ └── best_model.h5 # Final Model
│
├── notebooks/
│ └── EDA_ModelTraining.ipynb # Experiments & EDA
│
├── src/
│ ├── data_ingestion.py
│ ├── data_preprocessing.py
│ ├── model_trainer.py
│ ├── model_evaluation.py
│ └── model_pusher.py
│
├── app.py # Streamlit Web App
├── train.py # Training Pipeline Runner
├── Dockerfile # Containerization
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ Tech Stack Used

- Python
- TensorFlow / Keras (ANN Model)
- Scikit-learn (Preprocessing + Metrics)
- Pandas & NumPy (Data Handling)
- Streamlit (Web Deployment)
- GitHub Actions (CI/CD)
- Docker (Containerization)

---

## 🚀 How to Run This Project

### ✅ 1. Clone Repository

```bash
git clone https://github.com/rajan7838/ANN-Churn-Prediction.git
cd ANN-Churn-Prediction
✅ 2. Create Conda Environment
conda create -p venv python=3.11 -y
conda activate venv/
✅ 3. Install Requirements
pip install -r requirements.txt
🏋️ Run Training Pipeline
This command runs complete MLOps workflow:

python train.py
It will automatically generate:

Processed datasets

Encoders + Scaler

ANN Model (model.h5)

Final pushed model inside models/

🌐 Run Streamlit Web App
After training, start deployment:

streamlit run app.py
Then open in browser:

http://localhost:8501
📊 Model Output
The model predicts:

✅ Customer Will Stay

⚠️ Customer Will Exit (Churn)

🔁 CI/CD Pipeline
GitHub Actions automatically runs:

Dependency installation

Model training pipeline test

Configured in:

.github/workflows/main.yml
🐳 Docker Support
Build Docker Image:

docker build -t churn-app .
Run Container:

docker run -p 8501:8501 churn-app
📌 Future Improvements
MLflow Experiment Tracking

DVC Pipeline Versioning

Hyperparameter Tuning

Deployment on AWS / Render

Model Explainability (SHAP, LIME)

👨‍💻 Author
Rajan Kumar
📌 GitHub: https://github.com/rajan7838


