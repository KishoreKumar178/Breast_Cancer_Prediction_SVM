# Cancer Classification with Support Vector Machines (SVM)

## Overview

This repository contains Python code for performing cancer classification using Support Vector Machines (SVM). The code demonstrates how to build an SVM model, evaluate its performance, perform hyperparameter tuning using grid search, and utilize bagging with an ensemble of SVM models.

## Dataset

The code uses the "cancer.csv" dataset, which contains information about cancer samples. The dataset includes various features related to the samples, such as radius, texture, perimeter, area, smoothness, and more. The target variable is the diagnosis, which indicates whether the sample is malignant (M) or benign (B).

## Dependencies

The code relies on the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

These dependencies can be installed using pip or any other Python package manager. For example:

## Usage

1. Clone the repository:

git clone https://github.com/KishoreKumar178/Breast_Cancer_Prediction_SVM

2. Navigate to the project directory:


3. Ensure that the "cancer.csv" dataset is in the same directory as the Python script.

4. Open the Python script in an environment with the required dependencies installed (e.g., Jupyter Notebook, PyCharm, etc.).

5. Run the script to perform cancer classification with SVM:


The script includes the following steps:

- Importing necessary libraries
- Loading and preprocessing the dataset
- Splitting the data into training and test sets
- Standardizing the data
- Building an SVM model with a linear kernel
- Predicting with the model
- Evaluating the model's performance using various metrics (accuracy, precision, recall, F1-score, AUC-ROC)
- Plotting the AUC-ROC curve
- Performing hyperparameter tuning with grid search
- Evaluating the model's performance after hyperparameter tuning
- Plotting the AUC-ROC curve for the tuned model
- Utilizing bagging with an ensemble of SVM models
- Calculating the accuracy score of the bagging model on the test set

6. Customize the code as needed for your specific use case. For example, you can modify the hyperparameter grid in the grid search or change the SVM kernel type.


