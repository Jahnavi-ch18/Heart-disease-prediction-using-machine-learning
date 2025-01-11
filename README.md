# Heart-disease-prediction-using-machine-learning
This project aims to predict the presence of heart disease in patients using various machine learning algorithms.  The project explores several models, evaluates their performance, and provides a basis for understanding the risk factors associated with heart disease.

## Project Overview

Heart disease is a leading cause of death worldwide. Early detection and prediction are crucial for effective treatment and improved patient outcomes. This project leverages machine learning techniques to predict the likelihood of heart disease based on various clinical features.

The project follows these key steps:

1. **Data Collection:**  The project uses the "HeartDisease.csv" dataset, which contains patient data including age, gender, chest pain type, blood pressure, cholesterol levels, and other relevant features.
2. **Data Preparation and Exploration:** The data is loaded, cleaned, and explored using descriptive statistics and visualizations to understand the distributions and relationships between features.  This includes generating histograms, correlation matrices, and pair plots.
3. **Data Preprocessing:** Categorical features are one-hot encoded using `pd.get_dummies` to convert them into numerical representations suitable for machine learning models.
4. **Data Transformation:** Continuous features are standardized using `StandardScaler` to ensure that they have zero mean and unit variance, preventing features with larger values from dominating the models and improving model performance.
5. **Model Building:**  Several machine learning models are trained and evaluated, including:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Naive Bayes
    - Decision Tree
    - Random Forest
6. **Model Evaluation:**  The models are evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrices. The `print_score` function is defined to conveniently display these metrics for both training and testing sets.
7. **Single User Prediction:** The project demonstrates how to use the trained model (SVM in the `Logistic_X_SVM` notebook and Random Forest in the `Naive_bayes_X_Decision_Tree` notebook) to predict heart disease risk for a single patient given their features.

## Key Findings

* Exploratory Data Analysis reveals significant relationships between certain features and the presence of heart disease (e.g., chest pain type, resting EKG results, exercise-induced angina).
* The correlation analysis highlights features that are strongly correlated with the target variable, offering insights into potential risk factors.
* Model evaluation demonstrates the performance of different algorithms, with Random Forest and SVM showing promising results in terms of accuracy.

## Requirements

The project requires the following Python libraries:
pandas
numpy
matplotlib
seaborn
sklearn
You can install them using pip:
```bash
pip install -r requirements.txt
## Usage
Clone the repository: git clone <repository_url>

Install the required libraries: pip install -r requirements.txt

Open and run the Jupyter notebooks (Logistic_X_SVM.ipynb and Naive_bayes_X_Decision_Tree.ipynb) to explore the data analysis, model training, and evaluation process.

## Future Enhancements
* Experiment with other machine learning algorithms (e.g., gradient boosting, neural networks).

* Implement hyperparameter tuning to optimize model performance.

* Deploy the trained model as a web application for easier access.

* Explore feature engineering techniques to potentially improve predictive accuracy.

This enhanced README provides a more comprehensive description of the project, its methodology, findings, and potential future directions
