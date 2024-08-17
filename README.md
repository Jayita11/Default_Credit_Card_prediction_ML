# Credit Card Default Prediction

This Jupyter Notebook contains the code and analysis for predicting credit card defaulters using various machine learning models. The dataset used in this notebook is split into training and test sets, and several preprocessing steps are performed before model training and evaluation.

![7a758c47-7d4b-4500-a892-ca9e1c44c36d](https://github.com/user-attachments/assets/709e8f04-1b78-424c-a908-8674fcdf5a76)

## Table of Contents

1. [Libraries and Dataset](#libraries-and-dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Model Training & Selection](#model-training-&-Selection)
5. [Model Evaluation](#model-evaluation)
6. [Deployment](#deployment)
7. [💻 Run Instructions](#run-instructions)
8. [Conclusion](#conclusion)

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

### ROC Curve and Gini Coefficient Analysis

- **AUC:** The XGBoost model achieved an AUC of **0.7764**, suggesting good discrimination between events and non-events.
- **Gini Coefficient:** The model's Gini Coefficient was calculated as **0.5529**. This further confirms that the model is effective in its predictions. The Gini Coefficient ranges from -1 to 1, where a value closer to 1 signifies a perfect model, 0 indicates a model with no discriminative power, and -1 signifies a perfectly incorrect model.

### Insights

- **AUC of 0.77:** The model is good at distinguishing between events and non-events.
- **Gini Coefficient of 0.55:** This value supports the conclusion that the XGBoost model is effective in its predictions.

In summary, the **XGBoost model**, with its higher AUC and solid Gini Coefficient, emerged as the superior model, providing better predictive accuracy and class discrimination compared to the baseline Logistic Regression model.

-Base Model Classification Report

![Screenshot 2024-08-14 at 1 16 05 AM](https://github.com/user-attachments/assets/ea8e9a60-c19e-4831-b2f4-7869b62b7098)
-Base Model Classification Report

![Screenshot 2024-08-14 at 1 16 11 AM](https://github.com/user-attachments/assets/70753b8d-991b-417a-a7fa-2a715bb5aa94)
-Model Evaluation : ROC/AUC

![Screenshot 2024-08-14 at 1 16 20 AM](https://github.com/user-attachments/assets/a487ae45-9693-45e5-bfe3-5767a1b1534a)

## 🖥️ Run Instructions

If you wish to run this project locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone [https://github.com/Jayita11/Defaulter-Credit-Card-Prediction_ML]
    cd Defaulter-Credit-Card-Prediction_ML
    ```

2. **Set up a virtual environment**:

    ```bash
    python -m venv env
    ```

    Activate the environment:

    - On macOS/Linux:
      ```bash
      source env/bin/activate
      ```
    - On Windows:
      ```bash
      env\Scripts\activate
      ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

5. **Access the app**:

    Open your browser and go to [http://localhost:8501](http://localhost:8501) to use the Defaulter Credit Card Prediction_ML app.



## Deployment
Deploy the model using a Streamlit app (`app.py`). The app allows users to input various data about credit card holders to predict whether a customer will default or not. To use the app, follow the link provided below:

https://defaulter-credit-card-predictionml-vbua5rzo5qe4dcrhnhhrwb.streamlit.app/

![Animation_23](https://github.com/user-attachments/assets/3a54a724-35ea-4baa-9eb3-e7403cd8fe14)

## Conclusion

The notebook concludes with a summary of the model performances, highlighting the best-performing model and potential areas for further improvement.


