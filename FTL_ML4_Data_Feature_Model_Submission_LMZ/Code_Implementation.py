# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, classification_report, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE  # For handling class imbalance
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('Breast_Cancer.csv')  # Replace with the correct path to your dataset

# Inspect the dataset
print(df.info())
print(df['lable'].value_counts())  # Check the distribution of the target variable

# Handling missing values (imputation)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns='ID')))
df_imputed.columns = df.drop(columns='ID').columns  # Restore original column names
df_imputed['ID'] = df['ID']  # Add back the 'ID' column
df_imputed['lable'] = df['lable']  # Add back the target column

# Split features and target
X = df_imputed.drop(columns=['ID', 'lable'])
y = df_imputed['lable']

# One-hot encoding (if necessary)
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Visualization: Boxplot of raw data
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_imputed.drop(columns=['ID', 'lable']))
plt.title('Boxplot before Normalization (Breast Cancer)')
plt.xticks(rotation=90)
plt.show()

# Visualization: Boxplot of normalized data
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train_scaled)
plt.title('Boxplot after RMA Normalization (Breast Cancer)')
plt.xticks(rotation=90)
plt.show()

# Visualization: Histogram of raw data (before normalization)
plt.figure(figsize=(12, 6))
df_imputed.drop(columns=['ID', 'lable']).hist(bins=30, figsize=(15, 10))
plt.suptitle('Histogram of Raw Expression Values')
plt.show()

# Visualization: Histogram of normalized data
plt.figure(figsize=(12, 6))
pd.DataFrame(X_train_scaled).hist(bins=30, figsize=(15, 10))
plt.suptitle('Histogram of Normalized Expression Values')
plt.show()

# Visualization: Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df_imputed.drop(columns=['ID', 'lable']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# PCA for clustering visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_resampled, cmap='viridis')
plt.title('PCA Plot for Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cancer Class')
plt.show()

# Hyperparameter Tuning for Random Forest using GridSearchCV
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

# Train with the best Random Forest model
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Plot confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Healthy', 'Cancer'],
            yticklabels=['Healthy', 'Cancer'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Random Forest Confusion Matrix', fontsize=14)
plt.show()

# Classification Report for Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# ROC Curve for Random Forest
y_proba_rf = best_rf_model.predict_proba(X_test_scaled)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plotting ROC Curve for Random Forest
fig = px.area(
    x=fpr_rf, y=tpr_rf,
    title=f'Random Forest ROC Curve (AUC = {roc_auc_rf:.2f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
fig.show()

# Feature Importance for Random Forest
importances_rf = best_rf_model.feature_importances_
importance_df_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances_rf
}).sort_values(by='Importance', ascending=False)

# Bar Chart for Feature Importance (Random Forest)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_rf, palette='coolwarm')
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()

# Gradient Boosting Classifier for comparison
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train_resampled)

# Predictions with Gradient Boosting
y_pred_gb = gb_model.predict(X_test_scaled)

# Accuracy for Gradient Boosting
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb * 100:.2f}%")

# Confusion matrix for Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Plot confusion matrix for Gradient Boosting
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Healthy', 'Cancer'],
            yticklabels=['Healthy', 'Cancer'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Gradient Boosting Confusion Matrix', fontsize=14)
plt.show()

# Classification Report for Gradient Boosting
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

# ROC Curve for Gradient Boosting
y_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, y_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

# Plotting ROC Curve for Gradient Boosting
fig = px.area(
    x=fpr_gb, y=tpr_gb,
    title=f'Gradient Boosting ROC Curve (AUC = {roc_auc_gb:.2f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
fig.show()

# Cross-validation for Gradient Boosting
cv_scores_gb = cross_val_score(gb_model, X, y, cv=5)
print(f"\nGradient Boosting Cross-validation accuracy: {cv_scores_gb.mean() * 100:.2f}%")
