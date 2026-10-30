# 2026AJ05217-ml-assignment-2
Machine Learning subject Assignment 2 – Classification Models and Streamlit Deployment

# Customer Churn Prediction Using Machine Learning

## a. Problem Statement

Customer churn refers to the situation where customers discontinue their services or stop doing business with a company.

Retaining existing customers is important for businesses because acquiring new customers can be more expensive than retaining current ones. Identifying customers who are likely to leave allows companies to take preventive actions, improve customer satisfaction, and develop targeted retention strategies.

The objective of this project is to develop a machine learning-based customer churn prediction system that classifies customers into two categories:

**No Churn (0):** The customer is expected to continue using the service.
**Churn (1):** The customer is likely to discontinue the service.

Five classification algorithms are implemented and evaluated to compare their performance in predicting customer churn.

A Streamlit web application is also developed to allow users to explore model performance, compare classifiers, and upload compatible labelled CSV datasets for evaluation using the saved models.

## b. Dataset Description

The dataset used for this project is the **Telco Customer Churn Dataset** (`IT_customer_churn.csv`).

The dataset contains customer information related to demographics, subscribed services, account details, billing information, and customer tenure.

### Dataset Overview

| Description | Details |
|---|---|
| Dataset name | Telco Customer Churn |
| File name | IT_customer_churn.csv |
| Total records | 7,043 |
| Total input features | 19 |
| Target variable | Churn |
| Problem type | Binary Classification |
| Target classes | No and Yes |

### Features

The dataset contains the following types of information:

**Customer demographics:** Gender, Senior Citizen, Partner, and Dependents.
**Account information:** Tenure, Contract, Paperless Billing, and Payment Method.
**Subscribed services:** Phone Service, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, and Streaming Movies.
**Billing information:** Monthly Charges and Total Charges.

The target variable is `Churn`, which indicates whether a customer has discontinued the service.

### Data Preprocessing

The following preprocessing steps were performed:

1. Missing values in numerical features were handled using median imputation.
2. Missing values in categorical features were handled using most frequent value imputation.
3. Numerical features were standardized using StandardScaler.
4. Categorical features were converted into numerical representations using OneHotEncoder.
5. The target variable was converted into binary values:
   - No = 0
   - Yes = 1
6. The dataset was divided into training and testing sets using an 80:20 split.
7. Stratified sampling was used to maintain the class distribution in both sets.

The train-test split used random_state of 42 to ensure reproducibility.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/shannu711sr/2026AJ05217-ml-assignment-2

The repository contains the source code, dataset, saved machine learning models, application code, and dependency information required to run the project.

### Main Project Files

app.py - Streamlit application for model evaluation and comparison
train_models.py - Script used to preprocess data, train models, and save trained models
data/ - Contains the customer churn dataset
models/ - Contains the five saved classification models
requirements.txt - Lists the Python libraries required to run the project
.streamlit/config.toml - Streamlit application theme configuration
README.md - Project documentation

## d. Models Used and Evaluation Metrics

Five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (SVM)

The models were evaluated using the following metrics:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many predicted positive cases were actually positive.
- **Recall

