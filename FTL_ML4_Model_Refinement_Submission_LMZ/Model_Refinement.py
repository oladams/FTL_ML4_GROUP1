import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Breast_Cancer.csv')

# Handling missing values (imputation)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns='ID')))
df_imputed.columns = df.drop(columns='ID').columns

# Split features and target
X = df_imputed.drop(columns=['lable'])
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

# Hyperparameter Tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],  # Adjusting the number of trees
    'max_depth': [5, 10, 15],         # Increasing regularization
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2'],  # Reducing variance through feature selection
    'bootstrap': [True]              # Using bootstrap sampling
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train_resampled)
best_rf_model = grid_search.best_estimator_

# Evaluate the Random Forest Model
y_pred_rf = best_rf_model.predict(X_test_scaled)
train_accuracy_rf = best_rf_model.score(X_train_scaled, y_train_resampled)
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Training Accuracy: {train_accuracy_rf * 100:.2f}%")
print(f"Random Forest Testing Accuracy: {test_accuracy_rf * 100:.2f}%")
print(f"Accuracy Gap (Train vs Test): {abs(train_accuracy_rf - test_accuracy_rf) * 100:.2f}%")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Cross-validation
cv_scores_rf = cross_val_score(best_rf_model, X, y, cv=5)
print(f"\nRandom Forest Cross-validation Accuracy: {cv_scores_rf.mean() * 100:.2f}%")

# Learning Curve for Random Forest
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-Validation Score', color='orange')
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curve(best_rf_model, X_train_scaled, y_train_resampled, 'Random Forest Learning Curve')

# Feature Importance
importances_rf = best_rf_model.feature_importances_
importance_df_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances_rf
}).sort_values(by='Importance', ascending=False)

# Bar Chart for Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_rf, palette='coolwarm')
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()