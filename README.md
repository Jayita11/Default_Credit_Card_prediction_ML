# Credit Card Default Prediction

This Jupyter Notebook contains the code and analysis for predicting credit card defaulters using various machine learning models. The dataset used in this notebook is split into training and test sets, and several preprocessing steps are performed before model training and evaluation.

## Table of Contents

1. [Libraries and Dataset](#libraries-and-dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Model Training & Selection](#model-training-&-Selection)
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

## Model Training & Selection

The primary model used for predicting credit card defaults is **Logistic Regression**, chosen for its simplicity and interpretability. After establishing a baseline performance with Logistic Regression, additional models were tested to explore potential improvements in predictive accuracy. These models include:

- **Support Vector Machine (SVC):** A robust classifier that seeks to maximize the margin between different classes.Improved class separation but was computationally expensive.
- **Random Forest Classifier:** An ensemble learning method that combines multiple decision trees to improve model performance and reduce overfitting.Provided better handling of class imbalance and robustness to overfitting, with significant performance gains.
- **XGBoost Classifier:** An advanced gradient boosting algorithm known for its efficiency and accuracy in classification tasks.Achieved the highest F1-scores and ROC-AUC metrics, making it the top performer.

Each of these models was trained on the same preprocessed training data, and their performance was compared against the baseline Logistic Regression model. The aim was to determine whether the use of more complex models could lead to better predictive accuracy and overall model performance.

### Best Model Selection: Attempt 2 with Under-Sampling and XGBoost

After evaluating various models and approaches, the best model was selected based on **Attempt 2**, which used **Under-Sampling with XGBoost**. 

**Using SMOTE** didn’t effectively increase recall in this case, likely due to the introduction of noise or overlapping examples between classes. Instead, **Under-Sampling** was chosen to better focus the model on the minority class. This method reduces the dominance of the majority class, enabling the model to identify more true positives in the minority class, thus increasing recall. This approach is particularly beneficial when the priority is to capture more instances of the minority class, even if it means potentially increasing false positives.

In summary, the **XGBoost** model with under-sampling was selected as the best model, offering a balanced trade-off between precision and recall, and effectively addressing the class imbalance issue.

## Model Evaluation

### Baseline Model: Logistic Regression

The **Logistic Regression** model achieved an AUC of **0.7474**, indicating moderate ability in distinguishing between defaulters and non-defaulters. However, it struggled with recall for the minority class.

### Best Model: XGBoost Classifier

The **XGBoost** model, optimized with **Optuna**, outperformed the baseline with an AUC of **0.7764**. This model showed improved recall and better overall discrimination between classes.

### ROC Curve Analysis

The ROC curves showed that the **XGBoost model** had a higher AUC compared to the **Logistic Regression model**, particularly improving recall, which aligns with the goal to capture more positive cases.

### Insights

- **Recall Improvement:** XGBoost demonstrated better recall, effectively identifying more defaulters.
- **Overall Performance:** The modest improvement in AUC and recall highlights the benefits of hyperparameter tuning and class balancing.

In summary, the **XGBoost model** emerged as the superior model, providing better predictive accuracy and class discrimination.

## Conclusion

The notebook concludes with a summary of the model performances, highlighting the best-performing model and potential areas for further improvement.

## How to Use

1. Clone the repository or download the notebook.
2. Ensure you have all the required libraries installed. You can install them using:
   ```bash
   pip install -r requirements.txt

