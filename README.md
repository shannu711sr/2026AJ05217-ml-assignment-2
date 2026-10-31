# 2026AJ05217-ml-assignment-2

Machine Learning Subject Assignment 2 – Classification Models and Streamlit Deployment

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

**Dataset Source:** https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

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

The following preprocessing steps were performed:

1. The training dataset was loaded from the CSV file.
2. The `id` column was excluded because it does not contribute to passenger satisfaction prediction.
3. The target variable was converted into binary values:
   - Neutral or dissatisfied = 0
   - Satisfied = 1
4. Numerical and categorical features were identified separately.
5. Missing values in numerical features were handled using median imputation.
6. Missing values in categorical features were handled using most frequent value imputation.
7. Numerical features were standardized using `StandardScaler`.
8. Categorical features were converted into numerical representations using `OneHotEncoder`.
9. The dataset was split into training and testing subsets using an 80:20 stratified split.
10. The five classification models were trained using the training subset and evaluated using the testing subset.

**Dataset split details:**

| Dataset Subset | Number of Records | Percentage |
|---|---:|---:|
| Training dataset | 83,123 | 80% |
| Testing dataset | 20,781 | 20% |
| **Total** | **103,904** | **100%** |

The split was performed using stratification to maintain the original satisfaction-class distribution in both subsets. A fixed random state was used to make the split reproducible.

The testing subset was kept separate from model training to evaluate the models on records that were not used during training.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/shannu711sr/2026AJ05217-ml-assignment-2

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
| Logistic Regression | 0.8764 | 0.9281 | 0.8690 | 0.8418 | 0.8551 | 0.7477 |
| Decision Tree | 0.9069 | 0.9688 | 0.9037 | 0.8788 | 0.8911 | 0.8101 |
| K-Nearest Neighbors (KNN) | 0.9296 | 0.9710 | 0.9423 | 0.8921 | 0.9165 | 0.8566 |
| Naive Bayes | 0.8499 | 0.9188 | 0.8295 | 0.8228 | 0.8261 | 0.6941 |
| Support Vector Machine (SVM) | 0.8764 | 0.9277 | 0.8700 | 0.8404 | 0.8549 | 0.7477 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an Accuracy of 87.64% and an AUC of 0.9281. Its Precision (0.8690), Recall (0.8418), and F1-score (0.8551) indicate reasonably balanced performance in identifying satisfied passengers. Its MCC of 0.7477 also indicates good classification performance. However, it performed below KNN and Decision Tree across the main evaluation metrics. |
| Decision Tree | Decision Tree achieved an Accuracy of 90.69% and an AUC of 0.9688, making it the second-best model in terms of overall performance. Its Precision (0.9037), Recall (0.8788), and F1-score (0.8911) show that it performs well in identifying satisfied passengers. Its MCC of 0.8101 also indicates strong agreement between actual and predicted classifications. |
| K-Nearest Neighbors (KNN) | KNN achieved the highest Accuracy (0.9296), Precision (0.9423), F1-score (0.9165), and MCC (0.8566) among all five models. It also achieved a strong AUC of 0.9710 and Recall of 0.8921. Its high Precision indicates that most passengers predicted as satisfied were actually satisfied. Its Recall also indicates that it successfully identifies a large proportion of actually satisfied passengers. Overall, KNN provides the strongest performance across the evaluation metrics and is the best-performing model for this dataset. |
| Naive Bayes | Naive Bayes achieved an Accuracy of 84.99% and an AUC of 0.9188. Its Precision (0.8295), Recall (0.8228), and F1-score (0.8261) indicate comparatively lower performance than the other classifiers. Its MCC of 0.6941 is also the lowest among the five models. This suggests that Gaussian Naive Bayes is less effective at distinguishing between satisfied and neutral or dissatisfied passengers for this dataset. |
| Support Vector Machine (SVM) | SVM achieved an Accuracy of 87.64% and an AUC of 0.9277. Its Precision (0.8700), Recall (0.8404), and F1-score (0.8549) show reasonably balanced classification performance. Its MCC of 0.7477 is approximately the same as Logistic Regression. Overall, SVM performed moderately well but did not achieve the performance of Decision Tree or KNN. |
| **Overall Winner for the Dataset** | **K-Nearest Neighbors (KNN) is the best-performing model for the Airline Passenger Satisfaction dataset.** It achieved the highest Accuracy (92.96%), Precision (0.9423), F1-score (0.9165), and MCC (0.8566). It also achieved a strong AUC of 0.9710 and Recall of 0.8921. Its strong performance across all six evaluation metrics indicates that it provides the best overall classification results for this dataset. Therefore, KNN is selected as the overall winner. |

### Overall Conclusion

This project implements five machine learning classification algorithms to predict airline passenger satisfaction.

The models are evaluated using multiple performance metrics to understand their strengths and weaknesses. The results show that **K-Nearest Neighbors (KNN) achieved the best overall performance**, with an Accuracy of 92.96% and an AUC of 0.9710. It also achieved the highest Precision, F1-score, and MCC among the five classifiers.

Decision Tree also demonstrated strong performance, achieving an Accuracy of 90.69% and an AUC of 0.9688. Logistic Regression and Support Vector Machine achieved similar results, while Naive Bayes recorded the lowest overall performance among the five classifiers.

The Streamlit application provides an interactive interface where users can view the saved training evaluation metrics, select different classifiers, and evaluate uploaded labelled CSV datasets using the previously trained models.

The application uses saved models and does not perform model training or retraining during evaluation.