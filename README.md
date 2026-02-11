
# Arnold Face Classification

## Overview

This project uses Machine Learning to distinguish **Arnold Schwarzenegger** from non-Arnold individuals using facial image features.

## Objective

* Build pipelines for 3 classifiers
* Perform hyperparameter tuning using GridSearchCV
* Select best model using 5-fold cross-validation
* Evaluate on test data
* Achieve ≥ 80% accuracy

## Models Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree

## Workflow

1. Load dataset (`lfw_arnie_nonarnie.csv`)
2. Split into train/test (stratified)
3. Scale data using `StandardScaler`
4. Perform Grid Search with Cross-Validation
5. Select best model
6. Evaluate using:

   * Accuracy
   * Precision
   * Recall
   * F1 Score
7. Plot confusion matrix

## Technologies

* Python
* Scikit-learn
* Pandas
* Matplotlib

## Output

* Best model name
* Best parameters
* Cross-validation score
* Test metrics
* Confusion matrix visualization


