---
title: "Smart Crop Selection: Using Machine Learning to Optimize Farming Decisions"
date: 2025-01-25
categories:
  - Data Science Projects
tags:
  - Data Science
  - Multiclass Classification
---
In this post, we will work on a project that leverages machine learning to assist farmers in making data-driven decisions about which crop to plant based on essential soil characteristics. 

<img src="/assets/images/crop_selection_smart_farming.png" alt="smart farming" width="600">

To run the project, we follow the [essential steps](https://thanhtungvudata.github.io/data%20science%20insights/project-steps/) of a data science project as follows.

### 1. Problem Definition and Business Understanding
Choosing the right crop to plant each season is a crucial decision for farmers aiming to maximize their yield and profitability. Soil health plays a significant role in crop growth, with factors such as nitrogen, phosphorus, potassium levels, and soil pH having a direct impact on productivity. However, assessing soil conditions can be expensive and time-consuming, leading farmers to prioritize certain metrics based on budget constraints.

The goal is to develop an accurate predictive model that can analyze soil composition and recommend the most suitable crop for optimal yield. 

### 2. Dataset Description
The dataset used for this project, [**`soil_measures.csv`**](https://drive.google.com/file/d/12pCK-DKKWbeuPGdrMbDZATrctdfEJtYt/view?usp=sharing), contains key soil metrics collected from various fields. Each row in the dataset represents a set of soil measurements and the corresponding crop that is best suited for those conditions. The features included in the dataset are:

- **`N`** (Nitrogen): Nitrogen content ratio in the soil, a vital nutrient for plant growth.
- **`P`** (Phosphorous): Phosphorous content ratio, which supports root development and flowering.
- **`K`** (Potassium): Potassium content ratio, essential for disease resistance and water uptake.
- **`ph`**: The acidity or alkalinity level of the soil, impacting nutrient availability.
- **`crop`:** The target variable representing the ideal crop for the given soil composition.

### 3. Data Exploration and Cleaning

```python
# All required libraries are imported here
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('soil_measures.csv')

# Feature and target selection
X = df[['N', 'P', 'K', 'ph']]
y = df['crop']

# Check for crop types
unique_crop_type = df['crop'].unique()
print(unique_crop_type)
```
Output:
```
['rice' 'maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans'
 'mungbean' 'blackgram' 'lentil' 'pomegranate' 'banana' 'mango' 'grapes'
 'watermelon' 'muskmelon' 'apple' 'orange' 'papaya' 'coconut' 'cotton'
 'jute' 'coffee']
```

```python
# Check for missing values
missing_values_count = df.isna().sum()
print(missing_values_count)
```
Output:
```
N       0
P       0
K       0
ph      0
crop    0
dtype: int64
```

```python
# Visualize feature distributions using box plots
numerical_features = ['N', 'P', 'K', 'ph']

# Box plots
plt.figure(figsize=(12, 6))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, len(numerical_features), i)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Analyze the correlation between features using a heatmap
correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.show()
```
Output:

<img src="/assets/images/crop_data_box_plot.png" alt="Box Plot" width="600">

<img src="/assets/images/crop_data_correlation.png" alt="Box Plot" width="600">

Key Insights:
- The presence of outliers in `P` and `K` suggests data points with unusually high values, which might require further investigation or potential transformation. 
- Skewed distributions in `P` and `K` might indicate the need for data normalization or transformation before modeling.
- The strong correlation between `P` and `K` suggests they might be interdependent, possibly due to similar soil management practices or sources.
- Since `ph` has weak correlations, it may be treated as an independent factor in subsequent analysis.
- Multicollinearity might be an issue in modeling due to the high correlation between `P` and `K`.

Since `P` and `K` contain extreme outliers, it is unclear whether the observed strong correlation between them is driven by these outliers or if `P` and `K` are genuinely correlated. To investigate this, we conduct an experiment by simultaneously removing the outliers from both `P` and `K`, and then re-evaluate their correlation.

```python
# Define numerical features to check for outliers
numerical_features = ['P', 'K']

# Function to remove outliers using IQR
def remove_outliers_iqr(data, columns):
    df_cleaned = data.copy()
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

# Remove outliers from all numerical features
df_cleaned = remove_outliers_iqr(df, numerical_features)

# Plot boxplots for cleaned data
plt.figure(figsize=(12, 6))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, len(numerical_features), i)
    sns.boxplot(y=df_cleaned[feature], color='green')
    plt.title(f'Boxplot of {feature} (Cleaned)')
plt.tight_layout()
plt.show()

# Analyze the correlation between features using a heatmap
correlation_matrix = df_cleaned[numerical_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.show()
```

Output:
<img src="/assets/images/crop_data_cleaned_box_plot.png" alt="crop_data_cleaned_box_plot" width="600">
<img src="/assets/images/crop_data_cleaned_correlation.png" alt="crop_data_cleaned_correlation" width="600">

Key Actionable Insights:
- The correlation between features `P` and `K` becomes minor after removing outliers, it indicates that the observed multicollinearity was primarily influenced by the presence of extreme or anomalous data points. Outliers inflate correlation values, making it seem like two variables are highly related when they are not.
- With outliers removed, regression models should yield more reliable coefficients and predictions, reducing the risk of overfitting due to anomalous points.


### 4. Data Preprocessing
From the insights from the previous step, we will use the cleaned data (in which the outliers from both `P` and `K` are simultaneously removed) for data preprocessing steps. 

```python
# Feature and target selection
X = df_cleaned[['N', 'P', 'K', 'ph']]
y = df_cleaned['crop']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessing: Standardize numerical features
numerical_features = ['N', 'P', 'K', 'ph']
X[numerical_features] = X[numerical_features].astype('float64')
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
```

### 5. Model Selection and Training
Key Observations from the Correlation Matrix:
- The correlation values among the features (N, P, K, ph) are very low, mostly in the range of -0.07 to 0.30, indicating weak or no linear relationships between the features. 
- The highest correlation observed is 0.30 between N and K, which is still considered weak. 
- Most values are close to zero, suggesting that the features are nearly independent of one another.

```python
# Print the number of rows of Cleaned dataset
print(f"Cleaned dataset size: {df_cleaned.shape[0]}")
```
Output:
```
Cleaned dataset size: 2000
```

Recommended Models Based on Low Correlation and Small Dataset:
- Since the correlation among features is minor, models that do not rely heavily on linear relationships and can handle non-linear patterns would be more appropriate.
- Since the dataset is small, we will start with XGBoost (an advanced tree-based model) and compare its performance with a baseline using a simple K-Nearest Neighbors (KNN) model. 

```python
# Define parameter grids for XGBoost and LightGBM
xgb_params = {
    'max_depth': [10, 20],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.9]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Perform hyperparameter tuning using cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost tuning
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss',verbosity=1)
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=kf, scoring='accuracy', n_jobs=1, verbose=1)
xgb_grid.fit(X_train, y_train)

# Hyperparameter tuning for KNN
knn_model = KNeighborsClassifier()
knn_grid_search = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=kf, scoring='accuracy', n_jobs=1, verbose=1)
knn_grid_search.fit(X_train, y_train)
```

### 6. Model Evaluation
To choose a suitable model, we first explore whether data is balanced or not.

```python
target_counts = df_cleaned['crop'].value_counts()

# Plot the class distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=target_counts.index, y=target_counts.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('Crop Class Distribution')
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.show()
```
Output:

<img src="/assets/images/crop_data_cleaned_class_distribution.png" alt="crop_data_cleaned_class_distribution" width="600">

Since the numbers of data sample of each crop class are the same, the data is balanced. Thus, accuracy and marco F1-Score (unweighted average F1-Score) are commonly good metrics. However, considering the multi-class nature of the crop classification problem and the potential impact of both false positives and false negatives, we choose macro F1-Score to evaluate the model performance.  

```python
# Evaluate models
xgb_best_model = xgb_grid.best_estimator_
xgb_pred = xgb_best_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_pred, average='macro')

best_knn_model = knn_grid_search.best_estimator_
knn_pred = best_knn_model.predict(X_test)
knn_f1 = f1_score(y_test, knn_pred, average='macro')

# Plot comparison of macro F1-scores
models = ['XGBoost', 'KNN']
f1_scores = [xgb_f1, knn_f1]

plt.figure(figsize=(8, 5))
plt.bar(models, f1_scores, color=['blue', 'orange'])
plt.xlabel('Model')
plt.ylabel('Macro F1-Score')
plt.title('Comparison of Model Performance')
plt.ylim(0, 1)
plt.show()
```
Output:

<img src="/assets/images/crop_data_cleaned_model_evaluation.png" alt="crop_data_cleaned_model_evaluation" width="600">

Key Actionable Insights:
- XGBoost outperformed KNN in terms of macro F1-score, making it the preferred model for deployment.
- While XGBoost slightly outperforms KNN in terms of predictive accuracy, the decision on which model to use should also consider computational efficiency and resource constraints.
- If computational resources are limited, and the slight reduction in performance is acceptable, KNN would be the preferred choice due to its simplicity and efficiency.
- However, if accuracy is the top priority and computational resources allow, XGBoost should be used.

### Conclusion

The project has demonstrated that machine learning, particularly tree-based models like XGBoost, can provide valuable insights into agricultural decision-making. With further improvements and expanded datasets, such models can significantly contribute to precision agriculture.

The code of this project is available [here](https://github.com/thanhtungvudata/Smart-Crop-Selection). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
