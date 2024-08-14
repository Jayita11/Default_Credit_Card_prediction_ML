# Credit Card Default Prediction

This Jupyter Notebook contains the code and analysis for predicting credit card defaulters using various machine learning models. The dataset used in this notebook is split into training and test sets, and several preprocessing steps are performed before model training and evaluation.

## Table of Contents

1. [Libraries and Dataset](#libraries-and-dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)

## Libraries and Dataset

The notebook begins by importing the necessary libraries for data manipulation, visualization, and machine learning, including:

- `pandas` and `numpy` for data manipulation
- `matplotlib` and `seaborn` for data visualization
- `scikit-learn` for machine learning model training and evaluation
- `xgboost` for gradient boosting

The dataset used for this analysis is a credit card default dataset, which is loaded into Pandas DataFrames for both training and test sets.

## Data Preprocessing

Before training the models, the following preprocessing steps are carried out:

- **Feature Scaling:** The features are scaled using `MinMaxScaler` to ensure that all features contribute equally to the model.
- **Train-Test Split:** The dataset is split into training and testing subsets to validate the model performance on unseen data.

## Exploratory Data Analysis

EDA is performed to understand the distribution and relationships within the dataset, including:

- Visualizing the distribution of features
- Identifying correlations between features and the target variable
- Handling missing values and outliers

## Model Training

The notebook explores multiple machine learning models, including:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

Each model is trained using the preprocessed training data, and hyperparameter tuning is performed where necessary.

## Model Evaluation

The models are evaluated using various metrics, including:

- **F1 Score:** To measure the balance between precision and recall.
- **ROC-AUC:** To assess the model's ability to distinguish between classes.
- **Cross-Validation:** To evaluate the model's generalizability.

## Conclusion

The notebook concludes with a summary of the model performances, highlighting the best-performing model and potential areas for further improvement.

## How to Use

1. Clone the repository or download the notebook.
2. Ensure you have all the required libraries installed. You can install them using:
   ```bash
   pip install -r requirements.txt

