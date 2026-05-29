# Bank Loan Approval Prediction

A supervised machine learning project in R for predicting personal loan approval and comparing Logistic Regression, Random Forest, and Decision Tree models.

## Project Overview

This project aims to predict whether a bank customer is likely to receive a personal loan approval based on demographic, financial, and banking-related features.

The analysis includes data cleaning, exploratory data analysis, model training, model evaluation, and performance comparison between different classification algorithms.

The main goal is to identify the model that performs best in predicting loan approval, especially considering that the target variable is imbalanced.

## Dataset

The dataset used in this project is available on Kaggle:

[Bank Loan Approval - LR, DT, RF and AUC](https://www.kaggle.com/datasets/vikramamin/bank-loan-approval-lr-dt-rf-and-auc/data)

The dataset contains information about bank customers, including variables such as age, income, education level, family size, average credit card spending, mortgage value, and banking product usage.

The target variable is:

- `Personal.Loan`
  - `0` = Loan not approved
  - `1` = Loan approved

## Objectives

The main objectives of this project are:
- Analyze the structure and distribution of the dataset
- Clean and prepare the data for machine learning
- Handle negative values in the `Experience` variable
- Remove non-informative variables such as `ID` and `ZIP.Code`
- Explore the imbalance of the target variable
- Train different supervised machine learning models
- Evaluate model performance using appropriate classification metrics
- Compare the models and identify the best-performing one

## Methods Used

The project follows these main steps:
1. Data loading
2. Data cleaning
3. Exploratory Data Analysis
4. Visualization of variable distributions
5. Correlation analysis
6. Train/test split
7. Cross-validation
8. Upsampling to handle class imbalance
9. Model training
10. Model evaluation
11. Final model comparison

## Machine Learning Models

Three supervised classification models were implemented:
- Logistic Regression
- Random Forest
- Decision Tree

## Evaluation Metrics

The models were evaluated using the following metrics:
- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- Confusion Matrix
- ROC Curve
- AUC Score


## Main Results

The Random Forest model achieved the best overall performance among the models tested.

It showed strong results in terms of:
- Accuracy
- Specificity
- Sensitivity
- Precision
- AUC score

Logistic Regression was useful for interpretation because it allowed the analysis of the effect of each variable on the probability of loan approval. However, it produced more false positives compared to the other models.

The Decision Tree model also performed well and was easier to interpret, but its overall performance was slightly lower than Random Forest.


## Technologies Used

- R
- RStudio
- caret
- ranger
- rpart
- rpart.plot
- pROC
- ggplot2
- dplyr
- readr

## Repository Structure

```text
bank-loan-approval-prediction/
│
├── README.md
├── bank_loan_approval_RStudio_clean.R
├── data/
│   └── bank.csv
├── report/
│   └── Report_Bank_Loan_Approval.pdf
└── outputs/
    └── model_results.csv
```

## How to Run the Project

1. Download the dataset from Kaggle.
2. Rename the dataset file as `bank.csv`, if necessary.
3. Place `bank.csv` in the same folder as the R script or inside a `data/` folder.
4. Open the R script in RStudio.
5. Run the script.


## Authors

- Cecilia Morici
- Francesca Selaj
- Morena Farinelli
