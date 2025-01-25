---
title: "Smart Crop Selection: Using Machine Learning to Optimize Farming Decisions"
date: 2025-01-25
categories:
  - Data Science Projects
tags:
  - Data Science
  - Multiclass Classification
---

Choosing the right crop to plant each season is a crucial decision for farmers aiming to maximize their yield and profitability. Soil health plays a significant role in crop growth, with factors such as nitrogen, phosphorus, potassium levels, and soil pH having a direct impact on productivity. However, assessing soil conditions can be expensive and time-consuming, leading farmers to prioritize certain metrics based on budget constraints.

In this project, we leverage machine learning to assist farmers in making data-driven decisions about which crop to plant based on essential soil characteristics. The goal is to develop an accurate predictive model that can analyze soil composition and recommend the most suitable crop for optimal yield.

## Dataset Description
The dataset used for this project, [**`soil_measures.csv`**](https://drive.google.com/file/d/12pCK-DKKWbeuPGdrMbDZATrctdfEJtYt/view?usp=sharing), contains key soil metrics collected from various fields. Each row in the dataset represents a set of soil measurements and the corresponding crop that is best suited for those conditions. The features included in the dataset are:

- **`N`** (Nitrogen): Nitrogen content ratio in the soil, a vital nutrient for plant growth.
- **`P`** (Phosphorous): Phosphorous content ratio, which supports root development and flowering.
- **`K`** (Potassium): Potassium content ratio, essential for disease resistance and water uptake.
- **`ph`**: The acidity or alkalinity level of the soil, impacting nutrient availability.
- **`crop`:** The target variable representing the ideal crop for the given soil composition.

```
# All required libraries are imported here.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
```


