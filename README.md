# Machine Learning Assignment

## Overview
This project involves solving three machine learning tasks using a real-world dataset. The tasks include classification, regression, and feature extraction to demonstrate a wide range of machine learning techniques and their applications. The assignment is structured as follows:

1. **Question 1:** Classification Task  
2. **Question 2:** Regression Task  
3. **Question 3:** Feature Extraction and Classification/Regression  

---

## Question 1: Classification Task
**Objective:** Predict the `deposit` status of customers in the Bank Marketing dataset using classification models and evaluate their performance.

### Steps:
1. **Data Preprocessing:**  
   - Handled missing values and encoded categorical variables using Label Encoding.  
   - Standardized numerical features for better model performance.  

2. **Classification Models:**  
   - Trained Logistic Regression, Decision Tree, and Random Forest models with different train-test splits (70%, 80%, 90%).  
   - Evaluated models using accuracy and visualized the performance across train-test ratios.

3. **Ensemble Methods:**  
   - Applied Bagging, Boosting, and Stacking techniques to improve performance.  
   - Conducted 100 Monte Carlo runs for each ensemble method and visualized performance using boxplots.  

4. **Insights:**  
   - Ensemble methods outperformed base models by improving stability and accuracy.  
   - Selected the best-performing ensemble method and discussed its advantages and limitations.

---

## Question 2: Regression Task
**Objective:** Perform regression analysis using ensemble methods to predict a continuous target variable (`balance`) in the Bank Marketing dataset.

### Steps:
1. **Data Preprocessing:**  
   - Identified `balance` as the target variable and prepared features by encoding and standardizing the data.  

2. **Ensemble Methods for Regression:**  
   - Used Bagging Regressor, Gradient Boosting Regressor, and Stacking Regressor.  
   - Trained and evaluated the models, comparing their performance.  

3. **Insights:**  
   - Ensemble methods demonstrated better performance than individual regressors, reducing bias and variance.  
   - Visualized results to highlight the strengths of combining multiple models for regression tasks.

---

## Question 3: Feature Extraction and Classification/Regression
**Objective:** Apply PCA for dimensionality reduction and evaluate its impact on model performance.

### Steps:
1. **Dimensionality Reduction with PCA:**  
   - Used PCA to reduce the feature set while retaining at least 90% of the variance.  
   - Identified that **13 principal components** were sufficient to explain 90% of the data’s variance.  

2. **Model Evaluation with PCA-Reduced Features:**  
   - Trained Logistic Regression and Decision Tree classifiers on the reduced feature set.  
   - Compared the performance of models with and without PCA-reduced features.  

3. **Insights:**  
   - PCA effectively reduced dimensionality but led to a significant drop in classification accuracy.  
   - Highlighted the trade-off between reducing feature space and retaining task-relevant information.

---

## Files Included
- **`assignment.ipynb`**: Contains the Python code for all three questions.  
- **`bank.csv`**: The dataset used for this project.  
- **`report.pdf`**: Detailed report summarizing the methods, results, and insights for all questions.  

---

## Requirements
To run the code, ensure the following libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Use the following command to install the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
