# Machine Learning Project Documentation

## Model Refinement

### Overview
The model refinement phase is crucial in enhancing the performance of the machine learning model after the initial evaluation. This phase aims to optimize the model by adjusting key parameters, employing techniques like hyperparameter tuning, feature selection, and exploring different algorithms to achieve better accuracy, reduce overfitting, and ensure that the model generalizes well to unseen data.

### Model Evaluation
In the initial model evaluation, a **Random Forest** classifier was used, achieving **96%** accuracy on the test dataset. The model showed balanced performance, with precision, recall, and F1-score all near **1.0**. However, there were areas for potential improvement, such as:

- **Class Imbalance**: Although the model performed well, the class distribution between benign and malignant cases was imbalanced, leading to possible bias towards the majority class.
- **Overfitting Risk**: The model's high training accuracy compared to the test accuracy raised concerns about overfitting, though the cross-validation score was promising at **92.5%**.
- **Interpretability**: As a Random Forest model, the performance was strong, but it can be challenging to interpret directly, which could be an issue for applications requiring transparency.

Key metrics and visualizations used for the evaluation included:
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC Curve**

### Refinement Techniques
The following techniques were employed during the model refinement phase:

- **Hyperparameter Tuning**: Utilizing **GridSearchCV**, key hyperparameters of the Random Forest model were adjusted to improve generalization.
- **SMOTE for Handling Class Imbalance**: To mitigate the class imbalance, **SMOTE** was used to resample the training data, balancing the number of samples in each class.
- **Feature Scaling**: **StandardScaler** was applied to standardize the features, ensuring that the scale of the variables didn’t affect the model performance.
- **Cross-validation**: A **5-fold cross-validation** strategy was used to validate the model, ensuring robustness and reducing overfitting risks.

### Hyperparameter Tuning
Hyperparameter tuning was conducted to find the optimal values for several Random Forest parameters:
- **n_estimators**: The number of trees in the forest, tested for values **50, 100, 150**.
- **max_depth**: The maximum depth of each tree, tested for values **5, 10, 15**.
- **min_samples_split**: The minimum number of samples required to split an internal node, tested for values **5, 10, 15**.
- **min_samples_leaf**: The minimum number of samples required to be at a leaf node, tested for values **2, 4, 6**.
- **max_features**: The number of features to consider when looking for the best split, tested for **'sqrt' and 'log2'**.

**GridSearchCV** revealed that increasing the number of estimators and adjusting the maximum depth improved model stability and performance, yielding better accuracy and reducing overfitting.

### Cross-Validation
The cross-validation strategy employed during the model refinement phase was **5-fold cross-validation**. This method was chosen to provide a reliable estimate of the model's performance, ensuring that the model generalizes well across different subsets of data. The cross-validation score of **92.5%** was consistent with the performance on the test dataset, suggesting that the model was not overfitting and was robust.

### Feature Selection
While **Random Forest** inherently performs some form of feature selection, further refinement was done by analyzing the **feature importance**. This step helped identify key features that were most predictive of breast cancer. Unimportant features were excluded to reduce computational complexity without sacrificing performance. The model's feature importance visualization highlighted the most significant variables, allowing for better interpretation and focus on critical features.

---

## Test Submission

### Overview
The test submission phase involves preparing the trained model for evaluation on a separate test dataset, assessing its final performance, and preparing it for deployment. This phase ensures the model's generalization capability and readiness for real-world applications.

### Data Preparation for Testing
The test dataset was prepared by applying the same preprocessing steps used during training:
- **Missing Data Imputation**: Any missing values in the test data were handled using the same imputation strategy (mean imputation).
- **Feature Scaling**: The test dataset was scaled using the **StandardScaler** fitted on the training data to ensure consistency in the feature space.
- **One-Hot Encoding**: Categorical features in the test dataset were transformed using the same one-hot encoding approach as applied to the training set.

### Model Application
The trained **Random Forest** model was applied to the test dataset using the following code:

```python
# Predicting the labels for the test dataset
y_pred_test = best_rf_model.predict(X_test_scaled)

# Evaluating the model
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```

#### Conclusion
In the model refinement phase, the Random Forest model was optimized using techniques like hyperparameter tuning, class imbalance handling with SMOTE, feature scaling, and cross-validation. The model showed strong performance with 96% accuracy and demonstrated balanced precision, recall, and F1-scores. Cross-validation revealed robustness, and feature selection helped improve interpretability.

The test submission phase confirmed that the model generalized well to new data, maintaining its strong performance. The next step would involve deploying the model in a real-world healthcare system, where it could assist in early breast cancer detection.

### References
- Python Libraries: pandas, numpy, scikit-learn, imblearn, matplotlib, seaborn
- SMOTE: Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 2002.
- Random Forest Algorithm: Breiman, L. "Random Forests." Machine Learning, 2001.


## Screenshots

### Random Forest Leaning Curve:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Random_Forest_Leaning_Curve.png.png?raw=true)


### Random Forest Feature Importance:
![App Screenshot](https://github.com/oladams/FTL_ML4_GROUP1/blob/main/images/Random_forest_Feature_Importance.png?raw=true)
