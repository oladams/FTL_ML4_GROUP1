
# Gene Expression Data Processing, Analysis, and Machine Learning

## Overview
This project focuses on analyzing gene expression data using machine learning techniques to classify samples into **Breast Cancer** or **Healthy Control** categories. 
The data preprocessing pipeline includes handling missing values, normalization, feature scaling, and addressing class imbalances. 
The machine learning models implemented are **Random Forest** and **Gradient Boosting**, evaluated using metrics such as accuracy, confusion matrix, ROC-AUC, and cross-validation scores.

---

## Data Preparation and Cleaning

### Dataset Loading and Inspection
The dataset, `Breast_Cancer.csv`, includes gene expression values, sample identifiers (`ID`), and a label indicating the health condition (`lable`). 
The first step involves inspecting the dataset:
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # For handling class imbalance
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Breast_Cancer.csv')  # Replace with the correct path to your dataset

# Inspecting the dataset
print(df.info())
print(df['lable'].value_counts())
```

### Handling Missing Values
Missing values were imputed using the **mean** strategy:
```python
# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns='ID')))
df_imputed.columns = df.drop(columns='ID').columns  # Restore original column names
df_imputed['ID'] = df['ID']  # Add back the 'ID' column
df_imputed['lable'] = df['lable']  # Add back the target column
```

### Splitting Features and Target
```python
# Split features and target
X = df_imputed.drop(columns=['ID', 'lable'])
y = df_imputed['lable']

# One-hot encoding (if necessary)
X = pd.get_dummies(X, drop_first=True)
```

### Handling Class Imbalance
Class imbalance in the dataset was addressed using **SMOTE**:
```python
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Feature Scaling
```python
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
```

---

## Model Training and Evaluation

### 1. Random Forest Classifier

#### Hyperparameter Tuning:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train_resampled)
best_rf_model = grid_search.best_estimator_
```

#### Model Performance:
```python
from sklearn.metrics import accuracy_score

# Model Performance
y_pred_rf = best_rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
```

#### Confusion Matrix:
```python
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, cmap='YlGnBu', xticklabels=['Healthy', 'Cancer'], yticklabels=['Healthy', 'Cancer'])
plt.title("Random Forest Confusion Matrix")
plt.show()
```

#### ROC-AUC Curve:
```python
from sklearn.metrics import roc_curve, auc

# ROC-AUC Curve
y_proba_rf = best_rf_model.predict_proba(X_test_scaled)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plotting ROC Curve for Random Forest
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.show()
```

### 2. Gradient Boosting Classifier

#### Training:
```python
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train_resampled)
```

#### Model Performance:
```python
# Model Performance
y_pred_gb = gb_model.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb * 100:.2f}%")
```

#### ROC-AUC Curve:
```python
# ROC-AUC Curve
y_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

# Plotting ROC Curve for Gradient Boosting
plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, color='darkgreen', lw=2, label=f'Gradient Boosting (AUC = {roc_auc_gb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Gradient Boosting ROC Curve')
plt.legend(loc='lower right')
plt.show()
```

---

## Key Insights

1. **Class Imbalance Handling:** Using **SMOTE** significantly improved model performance by balancing the dataset.
2. **Feature Importance:** **Random Forest** identified significant features influencing classification, and these insights can guide future research.
3. **Model Performance:** **Random Forest** outperformed **Gradient Boosting** with better ROC-AUC and cross-validation accuracy, indicating it was better suited for this dataset.

---

## Visualization Summary

- **Confusion Matrices:** Highlighted model accuracy and misclassifications.
- **Feature Importance Plot:** Provided insights into which features contributed most to the classification.
- **ROC Curves:** Demonstrated the classifier's ability to distinguish between classes and compared model performance.


## Screenshots

### BOXPlot before Normalization:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/BOXPlot_before_Normalization.png?raw=true)

### BOXPlot after RMA Normalization:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/BOXPlot_after_RMA_Normalization.png?raw=true)

### Histogram of Raw Expression Values:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/Histogram_of_Raw_Expression_Values.png?raw=true)

### Histogram of Normalized Expression Values:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Histogram_of_Normalized_Expression_Values.png?raw=true)

### Correlation Heatmap of Features:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Correlation_Heatmap_of_Features.png?raw=true)

### PCA Plot Clustering:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/PCA_Plot_Clustering.png?raw=true)

### Random Forest Confusion Matrix:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Random_Forest_Confusion_Matrix.png?raw=true)

### Random Forest Roc Curve (AUC = 0.99):
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Random_Forest_Roc_Curve.png?raw=true)

### Random Forest Feature Importance:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Random_Forest_Roc_Curve.png?raw=true)


### Gradient Boosting Confusion Matrix:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Gradient_Boosting_Confusion_Matrix.png?raw=true)

### Gradient Boosting Roc Curve (AUC = 0.99):
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Gradient_Boosting_Roc_Curve.png?raw=true)









