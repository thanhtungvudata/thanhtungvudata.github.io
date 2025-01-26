---
title: "Smart Crop Selection: Using Machine Learning to Optimize Farming Decisions"
date: 2025-01-25
categories:
  - Data Science Projects
tags:
  - Data Science
  - Multiclass Classification
---
In this post, we will work on a project that leverages machine learning to assist farmers in making data-driven decisions about which crop to plant based on essential soil characteristics. To run the project, we follow the [essential steps](https://thanhtungvudata.github.io/data%20science%20insights/project-steps/) of a data science project as follows.

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
<img src="/assets/images/crop_data_cleaned_box_plot" alt="Box Plot" width="600">
<img src="/assets/images/crop_data_cleaned_box_plot" alt="Box Plot" width="600">

Key Actionable Insights:
- The correlation between features `P` and `K` becomes minor after removing outliers, it indicates that the observed multicollinearity was primarily influenced by the presence of extreme or anomalous data points. Outliers inflate correlation values, making it seem like two variables are highly related when they are not.
- With outliers removed, regression models should yield more reliable coefficients and predictions, reducing the risk of overfitting due to anomalous points.


### 4. Data Preprocessing
From the insights from the previous step, we will use the cleaned data (in which the outliers from both `P` and `K` are simultaneously removed) for data preprocessing steps. 

```python
# Feature and target selection
X = df_cleaned[['N', 'P', 'K', 'ph']]
y = df_cleaned['crop']
```