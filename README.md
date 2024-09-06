# Shipment Occurrence Prediction Using XGBoost & LightGBM

## Overview

This project predicts whether a shipment occurred based on patient and appointment data. The dataset includes various demographic, appointment, and follow-up details for PrEP appointments. This project applies data preprocessing, SMOTE-Tomek for class balancing, feature engineering, and machine learning models like XGBoost and LightGBM to build a classifier. The project also emphasizes model evaluation using SHAP, ROC curves, and Partial Dependence Plots.

## Dataset

- The dataset contains **10,418 rows** and **34 columns**, including a mix of numerical, categorical, and date-related features.
- Missing values are addressed, and categorical variables are encoded for modeling.
- The target variable is whether a shipment occurred (`Shipment_Occurred`), created by checking if the 'First Shipped Date After Completed Appointment' column has a value.

## Preprocessing Steps

1. **Column Analysis**: Each column was analyzed to determine its type (e.g., numerical, categorical) and relevance to the problem.
2. **Handling Missing Values**:
   - Mode imputation for 'Zip Code' and 'City'.
   - Median imputation for 'Time spent at clinic(Min)'.
   - Missing values in 'Provider' set to "Unknown".
3. **Feature Engineering**:
   - Created a binary indicator for `latefill` based on the shipment waiting period.
   - Converted clinic names, e.g., replacing 'Woodland' with 'Louisville'.
   - Extracted date components like month and day of the week from appointment dates.
   - Encoded categorical variables using one-hot encoding.
   - Grouped cities, providers, and zip codes with fewer than 50 occurrences into "rare" categories.

## Class Balancing with SMOTE-Tomek

To address class imbalance, the dataset was resampled using **SMOTE-Tomek**. This technique generates synthetic examples for the minority class using SMOTE, followed by Tomek Links to clean overlapping points between classes.

## Model Training and Hyperparameter Tuning

### XGBoost Classifier

- **Hyperparameter Tuning**: Used `RandomizedSearchCV` to find the best combination of hyperparameters, such as `n_estimators`, `max_depth`, `learning_rate`, and more.
- **Best Parameters**: The best parameters for XGBoost were:
  - `n_estimators`: 200
  - `max_depth`: 5
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 1.0
- **Early Stopping**: Applied early stopping with an evaluation set to avoid overfitting.

### LightGBM Classifier

- Trained a **LightGBM model** after cleaning feature names and splitting the dataset into training and test sets.
- The **Leaf-wise growth strategy** in LightGBM makes it faster and more efficient than XGBoost in this context.

## Model Performance

### XGBoost Model Evaluation:

- **Accuracy**: 87.5%
- **Precision**: 0.88
- **Recall**: 0.86
- **Confusion Matrix**: Provided to assess True Positive and False Positive rates.
- **ROC Curve**: AUC score of 0.93, indicating high predictive performance.

### LightGBM Model Evaluation:

- **Accuracy**: 87.5%
- **AUC-ROC**: 93.15%, highlighting strong performance.
- **Classification Report**: The model balances precision and recall effectively, particularly in class 1 (shipment occurred).

## Feature Importance

- **XGBoost Feature Importance**: The most influential features were:
  - `Race_Black`
  - `Race_White`
  - `clinic_MCPC East`
- **SHAP Analysis**: Used SHAP values to explain model predictions and analyze the contribution of features like `Race_Black`, `city_ft_NASHVILLE`, and `Waiting time(Days)` to the model’s decisions.

## Visualizations

### SHAP Summary Plot
Provides global feature importance and shows how feature values impact model predictions. Top features include `Race_Black`, `city_ft_NASHVILLE`, and `Waiting time(Days)`.

### Partial Dependence Plots
Visualizes the relationship between specific features like `Age`, `Waiting time(Days)`, and `Time spent at clinic(Min)` and the predicted outcome.

### ROC Curve
Displays the trade-off between sensitivity and specificity across different threshold settings.

### Feature Importance Plot
Shows the importance of features in the XGBoost model, sorted in descending order of influence.

## Conclusion

Both XGBoost and LightGBM perform well on the shipment prediction task, with balanced accuracy, high AUC scores, and interpretable feature importance. The combination of SMOTE-Tomek and advanced feature engineering contributed to the models’ success in predicting shipment occurrences based on appointment and follow-up data.

## How to Run

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the preprocessing and training script:
   ```bash
   python train_model.py
   ```
3. Visualize the model's feature importance and SHAP analysis using Jupyter Notebook.

## Requirements

    Python 3.8+
    pandas
    numpy
    scikit-learn
    imbalanced-learn
    XGBoost
    LightGBM
    matplotlib
    SHAP

## Future Improvements

    Further optimization of hyperparameters.
    Explore additional techniques for handling missing values.
    Test on a more diverse dataset for generalizability.

