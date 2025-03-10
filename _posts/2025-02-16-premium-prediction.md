---
title: "Building a Machine Learning Model to Estimate Insurance Premiums"
date: 2025-02-16
categories:
  - Data Science Projects
tags:
  - Data Science
  - Regression
---
Insurance pricing is a complex process where companies determine premiums based on various risk factors such as age, health, income, claims history, and lifestyle. However, the exact pricing models used by insurers are proprietary, making it difficult for customers and businesses to understand premium calculations.

In this project, we build a machine learning model using EDA, feature engineering, and XGBoost to predict insurance premium amounts. By mimicking the insurer's pricing strategy, our goal is to uncover key factors affecting premiums and develop a data-driven premium estimation tool.

<img src="/assets/images/Building a Machine Learning Model to Estimate Insurance Premiums.jpg" alt="Estimate Insurance Premiums" width="600">

To run the project, we follow the [essential steps](https://thanhtungvudata.github.io/data%20science%20insights/project-steps/) of a data science project as follows.

### 1. Problem Definition and Business Understanding
Insurance companies assess risk factors to price policies, but customers often lack transparency on why they are charged certain premiums. A predictive model can:
- Help customers estimate their premiums based on their profile.
- Allow insurers to optimize pricing and detect anomalies.
- Ensure fairness and regulatory compliance in premium calculations.

Our goal is to predict the insurance premium amount a customer would be charged based on their attributes using machine learning models. The model will identify the most important risk factors and help in estimating premiums for new customers.

### 2. Dataset Description
The dataset [**`train.csv`**](https://www.kaggle.com/competitions/playground-series-s4e12/data) includes various customer attributes that influence premium pricing. Key features include:

Numerical Features
- Age (Float) – Customer’s age.
- Annual Income (Float) – Earnings per year.
- Health Score (Float) – Health risk indicator.
- Previous Claims (Float) – Number of past claims.
- Vehicle Age (Float) – Age of insured vehicle.
- Credit Score (Float) – Financial risk measure.
- Insurance Duration (Float) – Policy period.

Categorical Features
- Gender (Male/Female)
- Marital Status (Single/Married/Other)
- Education Level (High School, Bachelor’s, Master’s, etc.)
- Occupation (Job type, with significant missing values)
- Smoking Status (Yes/No)
- Exercise Frequency (Regular, Occasional, None)
- Property Type (Owned, Rented)

Target Variable
- Premium Amount (Float) – The insurance premium charged by the insurer.

### 3. Data Exploration Analysis (EDA)

```python
# All required libraries are imported here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import scipy.stats as stats

# 📌 Load Dataset
file_path = "train.csv"  
df = pd.read_csv(file_path)

# Missing Values Analysis ###
print("\nMissing Values:\n")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
print(missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percentage', ascending=False))
```
Output:
```
                      Missing Values  Percentage
Previous Claims               364029   30.335750
Occupation                    358075   29.839583
Credit Score                  137882   11.490167
Number of Dependents          109672    9.139333
Customer Feedback              77824    6.485333
Health Score                   74076    6.173000
Annual Income                  44949    3.745750
Age                            18705    1.558750
Marital Status                 18529    1.544083
Vehicle Age                        6    0.000500
Insurance Duration                 1    0.000083
```

```python
# Distribution of Target Variable (Premium Amount) ###
plt.figure(figsize=(8, 5))
sns.histplot(df['Premium Amount'], bins=50, kde=True)
plt.title("Distribution of Premium Amount")
plt.xlabel("Premium Amount")
plt.ylabel("Frequency")
plt.show()
```
Output:

<img src="/assets/images/premium_prediction_distribution.png" alt="distribution" width="600">

```python
# Correlation Matrix for Numerical Features ###
numeric_df = df.select_dtypes(include=['number'])

# Compute the correlation matrix
corr_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.show()
```
Output:

<img src="/assets/images/premium_prediction_heatmap.png" alt="heatmap" width="600">

```python
# 📌 Convert Date Features
df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"])
df["year"] = df["Policy Start Date"].dt.year.astype("float32")
df["month"] = df["Policy Start Date"].dt.month.astype("float32")
df["day"] = df["Policy Start Date"].dt.day.astype("float32")
df["dow"] = df["Policy Start Date"].dt.dayofweek.astype("float32")
df.drop(columns=["Policy Start Date", "id"], inplace=True, errors="ignore")  # Remove ID and date column

# 📌 Identify Categorical & Numerical Features
cat_features = df.select_dtypes(include=["object"]).columns.tolist()
num_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 📌 ANOVA Test to Check Correlation Between Categorical & Numerical Features
anova_results = {}
significance_level = 0.05  # Default p-value threshold

print("\n📊 ANOVA Results:")
print("=" * 50)

for num_col in num_features:
    for cat_col in cat_features:
        groups = [df[num_col][df[cat_col] == category] for category in df[cat_col].dropna().unique()]

        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results[(cat_col, num_col)] = p_value

            # Print significant results
            if p_value < significance_level:
                print(f"✔ {cat_col} vs {num_col} | p-value: {p_value:.6f} (Significant ✅)")
            else:
                print(f"❌ {cat_col} vs {num_col} | p-value: {p_value:.6f} (Not Significant)")

# 📌 Convert Results to DataFrame
anova_df = pd.DataFrame(anova_results.items(), columns=["Feature Pair", "p-value"])
anova_df = anova_df.sort_values(by="p-value")

# 📌 Visualizing the Top Significant Relationships
significant_results = anova_df[anova_df["p-value"] < significance_level]

if not significant_results.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=-np.log10(significant_results["p-value"]), 
        y=significant_results["Feature Pair"].astype(str), 
        palette="coolwarm"
    )
    plt.xlabel("-log10(p-value)")
    plt.ylabel("Feature Pairs")
    plt.title("Significant ANOVA Relationships (Categorical vs Numerical)")
    plt.show()
else:
    print("\n🚀 No significant relationships found!")
```
Output:
```
📊 ANOVA Results:
==================================================
❌ Gender vs Age | p-value: nan (Not Significant)
❌ Marital Status vs Age | p-value: nan (Not Significant)
❌ Education Level vs Age | p-value: nan (Not Significant)
❌ Occupation vs Age | p-value: nan (Not Significant)
❌ Location vs Age | p-value: nan (Not Significant)
❌ Policy Type vs Age | p-value: nan (Not Significant)
❌ Customer Feedback vs Age | p-value: nan (Not Significant)
❌ Smoking Status vs Age | p-value: nan (Not Significant)
❌ Exercise Frequency vs Age | p-value: nan (Not Significant)
❌ Property Type vs Age | p-value: nan (Not Significant)
❌ Gender vs Annual Income | p-value: nan (Not Significant)
❌ Marital Status vs Annual Income | p-value: nan (Not Significant)
❌ Education Level vs Annual Income | p-value: nan (Not Significant)
❌ Occupation vs Annual Income | p-value: nan (Not Significant)
❌ Location vs Annual Income | p-value: nan (Not Significant)
❌ Policy Type vs Annual Income | p-value: nan (Not Significant)
❌ Customer Feedback vs Annual Income | p-value: nan (Not Significant)
❌ Smoking Status vs Annual Income | p-value: nan (Not Significant)
❌ Exercise Frequency vs Annual Income | p-value: nan (Not Significant)
❌ Property Type vs Annual Income | p-value: nan (Not Significant)
❌ Gender vs Number of Dependents | p-value: nan (Not Significant)
❌ Marital Status vs Number of Dependents | p-value: nan (Not Significant)
❌ Education Level vs Number of Dependents | p-value: nan (Not Significant)
❌ Occupation vs Number of Dependents | p-value: nan (Not Significant)
❌ Location vs Number of Dependents | p-value: nan (Not Significant)
❌ Policy Type vs Number of Dependents | p-value: nan (Not Significant)
❌ Customer Feedback vs Number of Dependents | p-value: nan (Not Significant)
❌ Smoking Status vs Number of Dependents | p-value: nan (Not Significant)
❌ Exercise Frequency vs Number of Dependents | p-value: nan (Not Significant)
❌ Property Type vs Number of Dependents | p-value: nan (Not Significant)
❌ Gender vs Health Score | p-value: nan (Not Significant)
❌ Marital Status vs Health Score | p-value: nan (Not Significant)
❌ Education Level vs Health Score | p-value: nan (Not Significant)
❌ Occupation vs Health Score | p-value: nan (Not Significant)
❌ Location vs Health Score | p-value: nan (Not Significant)
❌ Policy Type vs Health Score | p-value: nan (Not Significant)
❌ Customer Feedback vs Health Score | p-value: nan (Not Significant)
❌ Smoking Status vs Health Score | p-value: nan (Not Significant)
❌ Exercise Frequency vs Health Score | p-value: nan (Not Significant)
❌ Property Type vs Health Score | p-value: nan (Not Significant)
❌ Gender vs Previous Claims | p-value: nan (Not Significant)
❌ Marital Status vs Previous Claims | p-value: nan (Not Significant)
❌ Education Level vs Previous Claims | p-value: nan (Not Significant)
❌ Occupation vs Previous Claims | p-value: nan (Not Significant)
❌ Location vs Previous Claims | p-value: nan (Not Significant)
❌ Policy Type vs Previous Claims | p-value: nan (Not Significant)
❌ Customer Feedback vs Previous Claims | p-value: nan (Not Significant)
❌ Smoking Status vs Previous Claims | p-value: nan (Not Significant)
❌ Exercise Frequency vs Previous Claims | p-value: nan (Not Significant)
❌ Property Type vs Previous Claims | p-value: nan (Not Significant)
❌ Gender vs Vehicle Age | p-value: nan (Not Significant)
❌ Marital Status vs Vehicle Age | p-value: nan (Not Significant)
❌ Education Level vs Vehicle Age | p-value: nan (Not Significant)
❌ Occupation vs Vehicle Age | p-value: nan (Not Significant)
❌ Location vs Vehicle Age | p-value: nan (Not Significant)
❌ Policy Type vs Vehicle Age | p-value: nan (Not Significant)
❌ Customer Feedback vs Vehicle Age | p-value: nan (Not Significant)
❌ Smoking Status vs Vehicle Age | p-value: nan (Not Significant)
❌ Exercise Frequency vs Vehicle Age | p-value: nan (Not Significant)
❌ Property Type vs Vehicle Age | p-value: nan (Not Significant)
❌ Gender vs Credit Score | p-value: nan (Not Significant)
❌ Marital Status vs Credit Score | p-value: nan (Not Significant)
❌ Education Level vs Credit Score | p-value: nan (Not Significant)
❌ Occupation vs Credit Score | p-value: nan (Not Significant)
❌ Location vs Credit Score | p-value: nan (Not Significant)
❌ Policy Type vs Credit Score | p-value: nan (Not Significant)
❌ Customer Feedback vs Credit Score | p-value: nan (Not Significant)
❌ Smoking Status vs Credit Score | p-value: nan (Not Significant)
❌ Exercise Frequency vs Credit Score | p-value: nan (Not Significant)
❌ Property Type vs Credit Score | p-value: nan (Not Significant)
❌ Gender vs Insurance Duration | p-value: nan (Not Significant)
❌ Marital Status vs Insurance Duration | p-value: nan (Not Significant)
❌ Education Level vs Insurance Duration | p-value: nan (Not Significant)
❌ Occupation vs Insurance Duration | p-value: nan (Not Significant)
❌ Location vs Insurance Duration | p-value: nan (Not Significant)
❌ Policy Type vs Insurance Duration | p-value: nan (Not Significant)
❌ Customer Feedback vs Insurance Duration | p-value: nan (Not Significant)
❌ Smoking Status vs Insurance Duration | p-value: nan (Not Significant)
❌ Exercise Frequency vs Insurance Duration | p-value: nan (Not Significant)
❌ Property Type vs Insurance Duration | p-value: nan (Not Significant)
❌ Gender vs Premium Amount | p-value: 0.860021 (Not Significant)
❌ Marital Status vs Premium Amount | p-value: 0.620333 (Not Significant)
❌ Education Level vs Premium Amount | p-value: 0.329249 (Not Significant)
❌ Occupation vs Premium Amount | p-value: 0.677483 (Not Significant)
❌ Location vs Premium Amount | p-value: 0.508437 (Not Significant)
❌ Policy Type vs Premium Amount | p-value: 0.624434 (Not Significant)
❌ Customer Feedback vs Premium Amount | p-value: 0.072483 (Not Significant)
❌ Smoking Status vs Premium Amount | p-value: 0.858504 (Not Significant)
❌ Exercise Frequency vs Premium Amount | p-value: 0.694050 (Not Significant)
❌ Property Type vs Premium Amount | p-value: 0.349525 (Not Significant)

🚀 No significant relationships found!
```

Key Actionable Insights:
- There are significant missing values in the dataset. 
- Tree-based models (XGBoost, LightGBM, CatBoost) handle missing values better than linear models.
- The distribution of Premium Amount is highly skewed. 
- A log transformation is necessary to reduce skewness and improve model performance.
- Numerical features show little to no correlation with each other.
- No significant relationships were found between categorical features and numerical features.
- Since there are no strong relationships, models like XGBoost, CatBoost, or LightGBM may better capture complex interactions.


### 4 . Data Preprocessing 
From the insights from the previous step, we will use log transformation for data preprocessing steps. 

```python
# 📌 Define Target Variable (Log Transformation to Reduce Skewness)
df["Premium Amount"] = np.log1p(df["Premium Amount"])  # log(1 + x) transformation

# 📌 Convert Date Features
df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"])
df["year"] = df["Policy Start Date"].dt.year.astype("float32")
df["month"] = df["Policy Start Date"].dt.month.astype("float32")
df["day"] = df["Policy Start Date"].dt.day.astype("float32")
df["dow"] = df["Policy Start Date"].dt.dayofweek.astype("float32")
df.drop(columns=["Policy Start Date", "id"], inplace=True, errors="ignore")  # Remove ID and date column

# 📌 Identify Categorical and Numerical Features
cat_features = df.select_dtypes(include=["object"]).columns.tolist()
num_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_features.remove("Premium Amount")  # Exclude target variable

# 📌 Convert Categorical Features to "category" dtype for XGBoost
for col in cat_features:
    df[col] = df[col].astype("category")

# 📌 Define Features and Target
X = df.drop(columns=["Premium Amount"])
y = df["Premium Amount"]
```

### 5. Model Selection and Training
From the insights from the EDA step, we will use XGBoost for the predictive model.

```python
# 📌 Cross-Validation Setup (5-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))  # Out-of-Fold Predictions
feature_importance_df = pd.DataFrame(index=X.columns)
rmsle_per_fold = []  # Store RMSLE per fold

# 📌 Train XGBoost with Cross-Validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"🚀 Training Fold {fold + 1}...")

    # Split Data
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Train XGBoost Model with Native Categorical Handling
    model = XGBRegressor(
        enable_categorical=True,
        tree_method="hist",  # Optimized for speed
        max_depth=8,
        learning_rate=0.01,
        n_estimators=2000,
        colsample_bytree=0.9,
        subsample=0.9,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100
    )

    # Out-of-Fold Predictions
    fold_preds = model.predict(X_valid)
    oof_preds[valid_idx] = fold_preds

    # ✅ Calculate RMSLE for This Fold
    fold_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(fold_preds)))
    rmsle_per_fold.append(fold_rmsle)
    print(f"✔ Fold {fold + 1} RMSLE: {fold_rmsle:.5f}")

    # ✅ Store Feature Importance
    feature_importance_df[f"Fold_{fold + 1}"] = model.feature_importances_
```
Output:
```
🚀 Training Fold 1...
[0]	validation_0-rmse:1.09602
[100]	validation_0-rmse:1.05936
[200]	validation_0-rmse:1.04985
[300]	validation_0-rmse:1.04745
[400]	validation_0-rmse:1.04674
[500]	validation_0-rmse:1.04652
[600]	validation_0-rmse:1.04640
[700]	validation_0-rmse:1.04635
[800]	validation_0-rmse:1.04631
[900]	validation_0-rmse:1.04630
[1000]	validation_0-rmse:1.04628
[1100]	validation_0-rmse:1.04628
[1116]	validation_0-rmse:1.04627
✔ Fold 1 RMSLE: 1.04627
🚀 Training Fold 2...
[0]	validation_0-rmse:1.09482
[100]	validation_0-rmse:1.05816
[200]	validation_0-rmse:1.04877
[300]	validation_0-rmse:1.04647
[400]	validation_0-rmse:1.04584
[500]	validation_0-rmse:1.04566
[600]	validation_0-rmse:1.04561
[700]	validation_0-rmse:1.04558
[771]	validation_0-rmse:1.04558
✔ Fold 2 RMSLE: 1.04557
🚀 Training Fold 3...
[0]	validation_0-rmse:1.09471
[100]	validation_0-rmse:1.05882
[200]	validation_0-rmse:1.04953
[300]	validation_0-rmse:1.04726
[400]	validation_0-rmse:1.04661
[500]	validation_0-rmse:1.04644
[600]	validation_0-rmse:1.04636
[700]	validation_0-rmse:1.04633
[800]	validation_0-rmse:1.04631
[900]	validation_0-rmse:1.04630
[950]	validation_0-rmse:1.04630
✔ Fold 3 RMSLE: 1.04630
🚀 Training Fold 4...
[0]	validation_0-rmse:1.09521
[100]	validation_0-rmse:1.05785
[200]	validation_0-rmse:1.04809
[300]	validation_0-rmse:1.04553
[400]	validation_0-rmse:1.04480
[500]	validation_0-rmse:1.04457
[600]	validation_0-rmse:1.04448
[700]	validation_0-rmse:1.04442
[800]	validation_0-rmse:1.04441
[900]	validation_0-rmse:1.04440
[1000]	validation_0-rmse:1.04438
[1100]	validation_0-rmse:1.04436
[1140]	validation_0-rmse:1.04438
✔ Fold 4 RMSLE: 1.04436
🚀 Training Fold 5...
[0]	validation_0-rmse:1.09641
[100]	validation_0-rmse:1.05924
[200]	validation_0-rmse:1.04941
[300]	validation_0-rmse:1.04689
[400]	validation_0-rmse:1.04616
[500]	validation_0-rmse:1.04592
[600]	validation_0-rmse:1.04583
[700]	validation_0-rmse:1.04577
[800]	validation_0-rmse:1.04574
[900]	validation_0-rmse:1.04572
[929]	validation_0-rmse:1.04572
✔ Fold 5 RMSLE: 1.04571
```

### 6. Model Evaluation
We evaluated models using:
- Root Mean Squared Log Error (RMSLE) → Measures the average logarithmic difference between actual and predicted premium amounts, reducing the impact of large outliers and ensuring better performance on skewed data.
- Feature Importance Analysis → Identifies top factors influencing premium pricing.

```python
# 📌 Compute and Print Overall RMSLE
overall_rmsle = np.mean(rmsle_per_fold)
print("\n📊 Cross-Validation RMSLE Scores per Fold:")
for i, score in enumerate(rmsle_per_fold):
    print(f"✔ Fold {i + 1} RMSLE: {score:.5f}")
print(f"\n🚀 Overall Cross-Validation RMSLE: {overall_rmsle:.5f}")

# 📌 Compute Final RMSLE Using All Out-of-Fold Predictions
final_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
print(f"\n✅ Final Model RMSLE: {final_rmsle:.5f}")
```
Output:
```
📊 Cross-Validation RMSLE Scores per Fold:
✔ Fold 1 RMSLE: 1.04627
✔ Fold 2 RMSLE: 1.04557
✔ Fold 3 RMSLE: 1.04630
✔ Fold 4 RMSLE: 1.04436
✔ Fold 5 RMSLE: 1.04571

🚀 Overall Cross-Validation RMSLE: 1.04564

✅ Final Model RMSLE: 1.04564
```
```python
# 📌 Compute Average Feature Importance
feature_importance_df["Average"] = feature_importance_df.mean(axis=1)
feature_importance_df = feature_importance_df.sort_values(by="Average", ascending=False)

# 📌 Plot Top 20 Important Features
plt.figure(figsize=(12, 6))
sns.barplot(
    x=feature_importance_df["Average"][:20], 
    y=feature_importance_df.index[:20], 
    palette="viridis"
)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 20 Important Features")
plt.show()
```
Output:

<img src="/assets/images/premium_prediction_feature_importance.png" alt="feature imporatant" width="600">

Key Actionable Insights:
- Previous Claims is the most influential factor in predicting premium amounts, indicating that individuals with past claims significantly impact the model's predictions.
- Customer Feedback, Annual Income & Credit Score, highlighting the role of customer sentiment and financial stability in  premium pricing.
- Year of policy start is among the top features, indicating a seasonal or yearly pattern in insurance premium pricing.
- Health Score plays a critical role, possibly due to its impact on risk assessment.
- Marital Status has moderate influence, likely because they somehow correlate with income stability and insurance needs.
- Since Previous Claims and Customer Feedback are the top predictors, collecting accurate and detailed historical claim data and customer feedback could enhance model performance.
- Since Annual Income, Credit Score, and Health Score play significant roles, insurers could offer targeted pricing based on these variables. This leads to a problem of segmenting customers based on financial & health data
- The significance of year suggests that premiums might fluctuate seasonally, making it beneficial to explore time-series adjustments.

### Conclusion

This project successfully built a data-driven insurance premium prediction model using EDA, feature engineering, and XGBoost. Our model mimics the insurer’s pricing approach, revealing key premium factors while improving transparency.

The code of this project is available [here](https://github.com/thanhtungvudata/insurance_premium_prediction). 

For further inquiries or collaboration, please contact me at [my email](mailto:tungvutelecom@gmail.com).
