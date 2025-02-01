---
title: "Step-by-Step Guide to Building a Smart Crop Selection App Using Streamlit"
date: 2025-02-01
categories:
  - Data Science Projects
tags:
  - Data Science
  - ML Engineer
---

Agriculture plays a crucial role in feeding the world, and selecting the right crop for a given soil type can significantly improve productivity. With **Machine Learning and Streamlit**, we can build a **Smart Crop Selection App** that predicts the most suitable crop based on **soil properties** such as Nitrogen (N), Phosphorus (P), Potassium (K), and pH levels.

In this blog post, we will walk you through **how to use Streamlit** to build and deploy a **Smart Crop Selection** web application.

---

## 🚀 What is Streamlit?
[Streamlit](https://streamlit.io/) is an **open-source Python library** that makes it easy to build and share **interactive web applications** for data science and machine learning projects. It requires minimal code and allows quick visualization of models and data.

### ✅ **Why Use Streamlit for Smart Crop Selection?**
- **Easy to use** – Minimal front-end coding required.
- **Interactive UI** – Users can input soil data and get real-time predictions.
- **Fast Deployment** – Easily deployable on **Streamlit Community Cloud**.
- **Open-source & Free** – No need for expensive infrastructure.

---

## **🛠️ Step 1: Set Up Your Environment**

Before building the app, install the required dependencies.

### **📥 Install Required Libraries**
Run the following command in your terminal:
```bash
pip install streamlit pandas numpy scikit-learn joblib xgboost
```
This installs:
- **Streamlit** → To create the web app.
- **Pandas & NumPy** → For handling data.
- **scikit-learn** → For preprocessing and machine learning.
- **XGBoost** → The machine learning model for crop classification.
- **Joblib** → To save and load trained models.

---

## **📂 Step 2: Organize Your Project**
Create a directory structure:
```plaintext
smart_crop_selection/
├── app.py                # Streamlit web app
├── data/
│   ├── soil_data.csv      # Dataset
├── models/
│   ├── crop_model.pkl     # Trained ML model
├── scripts/
│   ├── preprocess.py      # Data preprocessing functions
│   ├── train_model.py     # Model training script
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
```

---

## **📊 Step 3: Prepare and Preprocess Data**

We need to preprocess the soil dataset before training the model.

### **🔹 Create `preprocess.py`**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(df, fit_scaler=True, fit_encoder=True):
    X = df.drop(columns=['Crop'])
    y = df['Crop']
    
    label_encoder = LabelEncoder()
    if fit_encoder:
        y_encoded = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, 'models/label_encoder.pkl')
    else:
        label_encoder = joblib.load('models/label_encoder.pkl')
        y_encoded = label_encoder.transform(y)
    
    scaler = StandardScaler()
    numerical_features = ['N', 'P', 'K', 'ph']
    if fit_scaler:
        X_scaled = X.copy()
        X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
        joblib.dump(scaler, 'models/scaler.pkl')
    else:
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = X.copy()
        X_scaled[numerical_features] = scaler.transform(X[numerical_features])
    
    return X_scaled, y_encoded
```

---

## **🤖 Step 4: Train the Machine Learning Model**

We use **XGBoost** to predict the best crop based on soil parameters.

### **🔹 Create `train_model.py`**
```python
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from scripts.preprocess import preprocess_data

# Load dataset
df = pd.read_csv('data/soil_data.csv')
X_scaled, y_encoded = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, 'models/crop_model.pkl')
```

---

## **🖥️ Step 5: Build the Streamlit Web App**

### **🔹 Create `app.py`**
```python
import streamlit as st
import pandas as pd
import joblib
from scripts.preprocess import preprocess_data

crop_model = joblib.load('models/crop_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

st.title("🌱 Smart Crop Selection App")

N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=250, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=250, value=50)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=250, value=50)
pH = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)

input_data = pd.DataFrame([[N, P, K, pH]], columns=['N', 'P', 'K', 'ph'])
test_input = preprocess_data(input_data, fit_scaler=False)
predicted_crop = crop_model.predict(test_input)[0]
recommended_crop = label_encoder.inverse_transform([predicted_crop])[0]

st.subheader(f"🌾 Recommended Crop: {recommended_crop}")
```

---

## **🚀 Step 6: Run and Deploy the App**
### **🔹 Run Locally**
```bash
streamlit run app.py
```
### **🔹 Deploy on Streamlit Community Cloud**
1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Select your repository and set `app.py` as the main file.
4. Click **Deploy**.

---

## **✅ Conclusion**
With **Streamlit and Machine Learning**, we have successfully built a **Smart Crop Selection App** that predicts the best crop based on soil parameters. 🚜🌱

Try deploying your own version and improve it by adding weather data or more soil parameters!

---

