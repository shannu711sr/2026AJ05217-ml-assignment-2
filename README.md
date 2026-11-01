# 2026AJ05217-ml-assignment-2

**Machine Learning Subject Assignment 2 – Classification Models and Streamlit Deployment**

# Airline Passenger Satisfaction Prediction Using Machine Learning

## a. Problem Statement

Airline passenger satisfaction refers to the level of satisfaction customers experience with an airline's services, including in-flight services, seating comfort, online booking, staff service, and overall travel experience.

Understanding passenger satisfaction is important for airlines because satisfied customers are more likely to use the airline again and recommend its services to others. Identifying dissatisfied passengers allows airlines to understand areas that require improvement and take appropriate actions to enhance customer experience.

The objective of this project is to develop a machine learning-based passenger satisfaction prediction system that classifies airline passengers into two categories:

- **Satisfied (1):** The passenger is satisfied with the airline's services.
- **Neutral or Dissatisfied (0):** The passenger is either neutral or dissatisfied with the airline's services.

Five classification algorithms are implemented and evaluated to compare their performance in predicting passenger satisfaction.

A Streamlit web application is also developed to allow users to explore model performance, compare classifiers, and upload compatible labelled CSV datasets for evaluation using the saved models.

## b. Dataset Description

The dataset used for this project is the **Airline Passenger Satisfaction Dataset** from Kaggle.

**Dataset Source:** [Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

The dataset contains information about airline passengers, their travel details, flight experience, and ratings of various airline services.

### Dataset Overview

| Description | Details |
|---|---|
| Dataset name | Airline Passenger Satisfaction |
| Dataset used for model training | train.csv |
| Original training dataset records | 103,904 |
| Training records after splitting | 83,123 |
| Testing records after splitting | 20,781 |
| Total input features | 22 |
| Target variable | satisfaction |
| Problem type | Binary Classification |
| Target classes | Neutral or dissatisfied and Satisfied |

### Features

The dataset contains the following types of information:

**Passenger demographics:** Gender, Customer Type, and Age.

**Travel information:** Type of Travel, Class, and Flight Distance.

**Service ratings:** Inflight Wi-Fi Service, Departure/Arrival Time Convenience, Ease of Online Booking, Gate Location, Food and Drink, Online Boarding, Seat Comfort, Inflight Entertainment, On-board Service, Leg Room Service, Baggage Handling, Check-in Service, Inflight Service, and Cleanliness.

**Flight delay information:** Departure Delay in Minutes and Arrival Delay in Minutes.

The target variable is `satisfaction`, which indicates whether a passenger is satisfied with the airline's services.

The target variable was converted into binary values:

- Neutral or dissatisfied = 0
- Satisfied = 1

### Target Distribution

The original dataset contains the following distribution:

| Satisfaction Class | Number of Records | Percentage |
|---|---:|---:|
| Neutral or Dissatisfied (0) | 58,879 | 56.67% |
| Satisfied (1) | 45,025 | 43.33% |
| **Total** | **103,904** | **100%** |

### Data Preprocessing

The following preprocessing steps were performed before training the classification models:

1. The training dataset was loaded from the CSV file.
2. The `id` column was excluded because it does not contribute to passenger satisfaction prediction.
3. The target variable was converted into binary values:
   - Neutral or dissatisfied = 0
   - Satisfied = 1
4. Numerical and categorical features were identified separately.
5. Missing values in numerical features were handled using median imputation.
6. Missing values in categorical features were handled using most frequent value imputation.
7. Model-specific preprocessing techniques were applied based on the requirements of each classification algorithm.
8. Numerical features were standardized using `StandardScaler` for models that are sensitive to feature scales.
9. Polynomial feature engineering using `PolynomialFeatures(degree=2)` was applied to the numerical features for Logistic Regression and SVM. This generates squared terms and interaction terms, allowing these models to capture nonlinear relationships between input features.
10. Categorical features were converted into numerical representations using `OneHotEncoder(handle_unknown="ignore")`.
11. Decision Tree and Gaussian Naive Bayes were trained without numerical feature scaling or polynomial expansion.
12. K-Nearest Neighbors used median imputation and standard scaling for numerical features, without polynomial expansion.
13. The dataset was split into training and testing subsets using an 80:20 stratified split.
14. The five classification models were trained using the training subset and evaluated using the testing subset.

### Model-Specific Preprocessing Configurations

| ML Model | Numerical Feature Preprocessing | Categorical Feature Preprocessing |
|---|---|---|
| Logistic Regression | Median imputation, StandardScaler, degree-2 PolynomialFeatures and StandardScaler | Most frequent imputation and OneHotEncoder |
| Decision Tree | Median imputation only | Most frequent imputation and OneHotEncoder |
| K-Nearest Neighbors | Median imputation and StandardScaler | Most frequent imputation and OneHotEncoder |
| Gaussian Naive Bayes | Median imputation only | Most frequent imputation and OneHotEncoder |
| Support Vector Machine (SVM) | Median imputation, StandardScaler, degree-2 PolynomialFeatures and StandardScaler | Most frequent imputation and OneHotEncoder |

Different preprocessing configurations were selected because classification algorithms have different requirements. Logistic Regression, SVM and KNN are sensitive to feature scales, whereas Decision Tree and Gaussian Naive Bayes do not require standardization in the same way.

Polynomial feature engineering was introduced for Logistic Regression and SVM to help capture nonlinear relationships and interactions between numerical features.

### Dataset Split Details

| Dataset Subset | Number of Records | Percentage |
|---|---:|---:|
| Training dataset | 83,123 | 80% |
| Testing dataset | 20,781 | 20% |
| **Total** | **103,904** | **100%** |

The split was performed using stratification to maintain the original satisfaction-class distribution in both subsets. A fixed random state was used to make the split reproducible.

The training and testing subsets both maintained a satisfaction rate of approximately 43.33%.

The testing subset was kept separate from model training to evaluate the models on records that were not used during training.

## c. GitHub Repository Link

**GitHub Repository:** [2026AJ05217-ml-assignment-2](https://github.com/shannu711sr/2026AJ05217-ml-assignment-2)

The repository contains the source code, dataset-related files, saved machine learning models, application code, and dependency information required to run the project.

### Main Project Files

- `app.py` - Streamlit application for model evaluation and comparison.
- `train_models.py` - Script used to preprocess data, split the dataset, train models, evaluate performance, and save trained models.
- `data/` - Contains the dataset files.
- `models/` - Contains the five saved classification models and model comparison results.
- `requirements.txt` - Lists the Python libraries required to run the project.
- `.streamlit/config.toml` - Streamlit application theme configuration.
- `README.md` - Project documentation.

## d. Models Used and Evaluation Metrics

Five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (SVM)

Each model was trained using a model-specific preprocessing pipeline. This allows the preprocessing techniques to be selected according to the requirements of each classification algorithm.

The models were evaluated using the following metrics:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many predicted positive cases were actually positive.
- **Recall:** Measures how many actual positive cases were correctly identified.
- **F1-score:** Represents the harmonic mean of Precision and Recall.
- **MCC:** Matthews Correlation Coefficient measures the quality of binary classification predictions, taking all four confusion matrix categories into account.

### Model Evaluation Results

The following results were obtained using the 20% testing subset containing 20,781 records.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9360 | 0.9823 | 0.9324 | 0.9189 | 0.9256 | 0.8695 |
| Decision Tree | 0.9069 | 0.9688 | 0.9037 | 0.8788 | 0.8911 | 0.8101 |
| K-Nearest Neighbors (KNN) | 0.9296 | 0.9710 | 0.9423 | 0.8921 | 0.9165 | 0.8566 |
| Naive Bayes | 0.8502 | 0.9191 | 0.8295 | 0.8238 | 0.8266 | 0.6948 |
| Support Vector Machine (SVM) | 0.9374 | 0.9822 | 0.9351 | 0.9194 | 0.9272 | 0.8724 |

### Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved an Accuracy of 93.60% and the highest AUC of 0.9823 among all five models. Its Precision (0.9324), Recall (0.9189), and F1-score (0.9256) indicate strong and balanced classification performance. Its MCC of 0.8695 also indicates strong agreement between actual and predicted classifications. The introduction of degree-2 polynomial feature engineering helped improve its ability to capture relationships between numerical features.

**Decision Tree**

Decision Tree achieved an Accuracy of 90.69% and an AUC of 0.9688. Its Precision (0.9037), Recall (0.8788), and F1-score (0.8911) show that it performs reasonably well in identifying satisfied passengers. Its MCC of 0.8101 also indicates good classification performance. However, it performed below Logistic Regression, KNN, and SVM across the main evaluation metrics.

**K-Nearest Neighbors (KNN)**

KNN achieved an Accuracy of 92.96%, the highest Precision (0.9423) among all five models, and an F1-score of 0.9165. Its AUC of 0.9710 and MCC of 0.8566 also indicate strong classification performance. Its high Precision indicates that most passengers predicted as satisfied were actually satisfied. However, its Recall of 0.8921 is lower than that of Logistic Regression and SVM. Overall, KNN remains a strong classifier but does not achieve the highest overall performance in the latest experiment.

**Naive Bayes**

Gaussian Naive Bayes achieved an Accuracy of 85.02% and an AUC of 0.9191. Its Precision (0.8295), Recall (0.8238), and F1-score (0.8266) indicate comparatively lower performance than the other classifiers. Its MCC of 0.6948 is also the lowest among the five models. This suggests that Gaussian Naive Bayes is less effective at distinguishing between satisfied and neutral or dissatisfied passengers for this dataset.

**Support Vector Machine (SVM)**

SVM achieved the highest Accuracy (93.74%) among all five models, along with an AUC of 0.9822. Its Precision (0.9351), Recall (0.9194), and F1-score (0.9272) indicate strong and balanced classification performance. It also achieved the highest MCC (0.8724), indicating the strongest overall agreement between actual and predicted classifications. The use of degree-2 polynomial feature engineering allowed the model to capture additional relationships between numerical features. Overall, SVM provides the strongest performance across the main evaluation metrics.

### Overall Winner for the Dataset

**Support Vector Machine (SVM) is the best-performing model for the Airline Passenger Satisfaction dataset.**

It achieved the highest Accuracy (93.74%), F1-score (0.9272), and MCC (0.8724) among the five classifiers. It also achieved a strong AUC of 0.9822 and Recall of 0.9194.

Although Logistic Regression achieved a marginally higher AUC of 0.9823, SVM performed better in Accuracy, Precision, Recall, F1-score, and MCC. Therefore, SVM is selected as the overall winner based on its performance across the evaluation metrics.

### Overall Conclusion

This project implements five machine learning classification algorithms to predict airline passenger satisfaction.

The models are evaluated using multiple performance metrics to understand their strengths and weaknesses. Model-specific preprocessing and feature engineering techniques were applied to improve the suitability of the input data for each classification algorithm.

The results show that **Support Vector Machine (SVM) achieved the best overall performance**, with an Accuracy of 93.74%, an F1-score of 0.9272, and an MCC of 0.8724. It also achieved a strong AUC of 0.9822 and Recall of 0.9194.

Logistic Regression also demonstrated excellent performance, achieving an Accuracy of 93.60% and the highest AUC of 0.9823. K-Nearest Neighbors achieved the highest Precision of 0.9423, while Decision Tree delivered an Accuracy of 90.69%. Gaussian Naive Bayes recorded the lowest overall performance among the five classifiers.

The introduction of polynomial feature engineering for Logistic Regression and SVM improved their ability to capture relationships between numerical features, resulting in stronger classification performance.

Based on the latest evaluation results, SVM is selected as the overall winner for this dataset.

The Streamlit application provides an interactive interface where users can view the saved training evaluation metrics, select different classifiers, and evaluate uploaded labelled CSV datasets using the previously trained models.

The application uses saved models and does not perform model training or retraining during evaluation.