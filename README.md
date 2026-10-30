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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7984 | 0.8281 | 0.6347 | 0.5668 | 0.5989 | 0.4662 |
| KNN | 0.7637 | 0.7896 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Support Vector Machine (SVM) | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

Observations on the performance of each model on the chosen dataset. 

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved the best overall performance among the five models, with the highest Accuracy (80.55%), AUC (0.8419), F1-score (0.6040), and MCC (0.4790). It provides a good balance between identifying customers likely to churn and making correct churn predictions. Its Recall of 55.88% indicates that it identifies slightly more than half of the actual churn customers. Overall, it is the most balanced and reliable model for this dataset. |
| Decision Tree | Decision Tree achieved an Accuracy of 79.84% and an AUC of 0.8281, making it the second-best model in terms of overall performance. Its Precision (0.6347), Recall (0.5668), and F1-score (0.5989) are reasonably balanced. Its performance is close to Logistic Regression, but it falls slightly behind across the main evaluation metrics. |
| K-Nearest Neighbors (KNN) | KNN achieved an Accuracy of 76.37% and an AUC of 0.7896. Its Precision (0.5527) and Recall (0.5749) are relatively balanced, but its F1-score (0.5636) and MCC (0.4018) are the lowest among the five models. This indicates that KNN is less effective at distinguishing between customers who churn and those who do not churn compared with the other models. |
| Naive Bayes | Naive Bayes achieved the highest Recall (0.8369) among all five models, meaning it successfully identifies a large proportion of customers who actually churn. However, its Precision is only 0.4589, indicating that it also incorrectly classifies many non-churning customers as churners. It has the lowest Accuracy (69.48%), but it may be useful when identifying as many potential churn customers as possible is the primary objective. |
| Support Vector Machine (SVM) | SVM achieved an Accuracy of 79.13% and a Precision of 0.6418, showing that its churn predictions are reasonably accurate. However, its Recall of 0.4840 is the lowest among all five models, meaning it misses a considerable number of actual churn customers. Its overall performance is moderate, with an F1-score of 0.5518 and an MCC of 0.4264. |
| **Overall Winner for the Dataset** | **Logistic Regression is the best-performing model for the Bank Customer Churn dataset.** It achieved the highest Accuracy (80.55%), AUC (0.8419), F1-score (0.6040), and MCC (0.4790) among all five models. It offers the best overall balance of predictive performance compared with the other classifiers. Although Naive Bayes has a higher Recall, Logistic Regression provides a stronger overall result across the evaluation metrics and is therefore selected as the overall winner for this dataset. |