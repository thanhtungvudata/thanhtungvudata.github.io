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
│   ├── soil_measures.csv      # Dataset
├── models/
│   ├── crop_model.pkl     # Trained ML model
│   ├── label_encoder.pkl  # Encode data labels
│   ├── scaler.pkl         # Scale data inputs
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
    """
    Preprocesses the input dataset:
    - Encodes the target variable
    - Standardizes numerical features
    
    Parameters:
    df (pd.DataFrame): Input data containing soil parameters and crop labels
    fit_scaler (bool): Whether to fit a new StandardScaler (True for training, False for inference)
    fit_encoder (bool): Whether to fit a new LabelEncoder (True for training, False for inference)
    
    Returns:
    tuple: (Preprocessed feature matrix X, Encoded target variable y, scaler, label_encoder)
    """
    # Separate features and target
    X = df.drop(columns=['crop'])  # Ensure 'crop' is the target variable
    y = df['crop']
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    if fit_encoder:
        y_encoded = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, 'models/label_encoder.pkl')  # Save encoder
    else:
        label_encoder = joblib.load('models/label_encoder.pkl')
        y_encoded = label_encoder.transform(y)
    
    # Standardize numerical features
    numerical_features = ['N', 'P', 'K', 'ph']
    X[numerical_features] = X[numerical_features].astype('float64')
    scaler = StandardScaler()
    if fit_scaler:
        X_scaled = X.copy()
        X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])
        joblib.dump(scaler, 'models/scaler.pkl')  # Save scaler
    else:
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = X.copy()
        X_scaled[numerical_features] = scaler.transform(X[numerical_features])
    
    return X_scaled, y_encoded, scaler, label_encoder

def preprocess_input(input_df):
    """
    Prepares new input data for model prediction.
    
    Parameters:
    input_df (pd.DataFrame): User input containing soil parameters
    
    Returns:
    pd.DataFrame: Scaled input ready for prediction
    """
    scaler = joblib.load('models/scaler.pkl')  # Load pre-trained scaler
    numerical_features = ['N', 'P', 'K', 'ph']
    input_df[numerical_features] = input_df[numerical_features].astype('float64')
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    return input_df
```

---

## **🤖 Step 4: Train the Machine Learning Model**

We use **XGBoost** to predict the best crop based on soil parameters.

### **🔹 Create `train_model.py`**
```python
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from .preprocess import preprocess_data

# Load dataset
df = pd.read_csv('data/soil_measures.csv')

# Preprocess data
X_scaled, y_encoded, scaler, label_encoder = preprocess_data(df, fit_scaler=True, fit_encoder=True)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Define parameter grids for XGBoost
xgb_params = {
    'max_depth': [10, 20],
    'learning_rate': [0.001, 0.1, 1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.7, 0.9]
}

# Perform hyperparameter tuning using cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost tuning
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', verbosity=1)
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=kf, scoring='accuracy', n_jobs=1, verbose=1)
xgb_grid.fit(X_train, y_train)

# Best model from GridSearchCV
best_xgb_model = xgb_grid.best_estimator_

# Evaluate model
y_pred = best_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best XGBoost Model Accuracy: {accuracy:.4f}')


# Save the best model
joblib.dump(best_xgb_model, 'models/crop_model.pkl')
```

---

## **🖥️ Step 5: Build the Streamlit Web App**

### **🔹 Create `app.py`**
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scripts.preprocess import preprocess_input
from sklearn.metrics import accuracy_score

# Load trained model and label encoder
crop_model = joblib.load('models/crop_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Streamlit UI
st.title("🌱 Smart Crop Selection Web App")
st.markdown("Enter soil parameters to get the best crop recommendation.")

# Sidebar inputs
st.sidebar.header("Input Soil Parameters")
N = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=160.0, value=30.0)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=160.0, value=60.0)
K = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=50.0)
pH = st.sidebar.number_input("pH Level", min_value=0.0, max_value=10.0, value=6.0)

# Convert inputs to DataFrame
input_data = pd.DataFrame([[N, P, K, pH]], columns=['N', 'P', 'K', 'ph'])

# Preprocess input
test_input = preprocess_input(input_data)

# Prediction
if st.sidebar.button("Predict Crop"):
    predicted_crop_index = crop_model.predict(test_input)[0]
    recommended_crop = label_encoder.inverse_transform([predicted_crop_index])[0]
    
    # Calculate accuracy on training data
    training_data = pd.read_csv('data/soil_measures.csv')
    X_train, y_train, _, _ = preprocess_input(training_data.drop(columns=['crop'])), training_data['crop'], None, None
    y_train_encoded = label_encoder.transform(y_train)
    y_train_pred = crop_model.predict(X_train)
    accuracy = accuracy_score(y_train_encoded, y_train_pred) * 100
    
    # Display results
    st.subheader("🌾 Recommended Crop:")
    st.write(f"**{recommended_crop}** with an accuracy of **{accuracy:.2f}%**")
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

## Live Demo 
You can access the deployed version of this app: [https://smartcropselection.streamlit.app/](https://github.com/thanhtungvudata/smart_crop_selection_web_app).

---

## Deployment Code & Model Development Insights

The deployment code of the project is available [here](https://github.com/thanhtungvudata/smart_crop_selection_web_app).

The data exploratory analysis and model development insights are discussed in detail in my previous [post](https://thanhtungvudata.github.io/data%20science%20projects/crop-selection/).

---

## **✅ Conclusion**
With **Streamlit and Machine Learning**, we have successfully built a **Smart Crop Selection App** that predicts the best crop based on soil parameters. 🚜🌱

Try deploying your own version and improve it by adding weather data or more soil parameters!