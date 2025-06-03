# Heart Disease Prediction: Exploratory Data Analysis & Classical Machine Learning

## Overview

This project dives into the UCI Heart Disease dataset to uncover insights and build predictive models for heart disease. The primary goal of this notebook is to perform a thorough Exploratory Data Analysis (EDA) to understand the data's characteristics and then to train and evaluate several classical machine learning models for predicting the presence and severity of heart disease.

## Dataset

The data used is the combined Heart Disease dataset from the UCI Machine Learning Repository, which consolidates data from four different locations: Cleveland, Hungary, Switzerland, and VA Long Beach. It contains 14 key attributes used for prediction.

## Workflow

This notebook follows a standard data science workflow:

1.  **Data Loading & Initial Exploration:** Importing necessary libraries and loading the dataset.
2.  **Exploratory Data Analysis (EDA):**
    *   Examining distributions of key features like age, gender, and dataset origin.
    *   Analyzing relationships between features (e.g., chest pain types across different datasets or genders).
    *   Investigating descriptive statistics (mean, median, mode) for relevant columns.
3.  **Data Preprocessing & Imputation:**
    *   Identifying and quantifying missing values.
    *   Implementing imputation strategies:
        *   `IterativeImputer` for numerical features like 'trestbps', 'ca', 'thalch', 'oldpeak', and 'chol'.
        *   Custom ML-based imputation functions (`impute_categorical_missing_data` and `impute_continuous_missing_data`) for categorical/object type columns using Random Forest.
4.  **Outlier Handling:**
    *   Visualizing outliers using box plots.
    *   Making informed decisions about handling potential outliers (e.g., removing rows with `trestbps == 0`).
5.  **Feature Encoding:** Using `LabelEncoder` for categorical features to prepare them for machine learning models.
6.  **Model Training & Evaluation:**
    *   Splitting the data into training and testing sets.
    *   Building a pipeline including `KNNImputer` (for any remaining NaNs post-split in numerical features, though ideally, X\_train/X\_test should be clean after initial imputation) and `MinMaxScaler`.
    *   Training and evaluating a suite of classical machine learning models:
        *   Logistic Regression
        *   K-Nearest Neighbors (KNN)
        *   Support Vector Machine (SVM)
        *   Naive Bayes (GaussianNB)
        *   Decision Tree
        *   Random Forest
        *   AdaBoost
        *   Gradient Boosting
        *   XGBoost
    *   Using cross-validation and test accuracy as performance metrics.
7.  **Results & Findings:** Summarizing the key insights and model performance.

## Key Findings & Outputs from EDA and Classical ML

1.  **Age Insights:**
    *   The youngest age recorded for heart disease in this dataset is 28 years.
    *   The peak age for heart disease occurrence for both males and females is around 54-55 years.
2.  **Gender Disparity:**
    *   Males constitute approximately 78.91% of the dataset, while females make up 21.09%.
    *   This indicates that males are about 274.23% more prevalent than females in this particular dataset.
3.  **Dataset Origin:**
    *   The Cleveland dataset contributes the highest number of patients, while Switzerland has the lowest.
    *   Gender distribution varies by dataset: Cleveland has the most females, VA Long Beach the fewest. Hungary has the most males, Switzerland the fewest.
4.  **Chest Pain (`cp`):**
    *   Asymptomatic angina is the most common type of chest pain reported.
    *   This is particularly prevalent in the Cleveland dataset.
5.  **Resting Blood Pressure (`trestbps`):**
    *   The notebook notes specific patterns (e.g., "Most of the females has trestbps is normal (143)"), though the original column is numerical and was later imputed. This finding might refer to patterns observed before detailed imputation or in conjunction with other categorical features not fully detailed here. *[Self-correction: The notebook output on this specific finding was a bit brief; this is a general observation based on the summary provided by the user in the original notebook's markdown.]*
6.  **Imputation Success:**
    *   Custom functions were developed and used to impute missing values for both categorical and numerical data, leveraging machine learning models like Random Forest for imputation.
    *   IterativeImputer was also used for numerical columns.
7.  **Outlier Management:**
    *   Rows with `trestbps` (resting blood pressure) equal to 0 were identified and removed as they likely represent data errors.
    *   A high number of `chol` (cholesterol) values at 0 were noted but not treated as outliers, suggesting this might be a specific encoding for missing/unrecorded data in some of the original datasets.
    *   Other potential outliers were considered in context (e.g., a `trestbps` of 200.0, while high, is physiologically possible).
8.  **Classical Model Performance:**
    *   Nine different classical machine learning models were evaluated.
    *   **GradientBoostingClassifier** was identified as the best-performing model among those tested, with a test accuracy of approximately **65.58%**.
    *   While this accuracy indicates room for improvement, it provided a benchmark for classical approaches on this dataset.

## Technologies Used

*   Python 3
*   Pandas & NumPy (Data Manipulation)
*   Matplotlib, Seaborn, Plotly (Data Visualization)
*   Scikit-learn (Preprocessing, Imputation, Modeling, Metrics)
*   XGBoost

## How to Use

1.  Ensure you have the required libraries installed (see import section in the notebook).
2.  Make sure the dataset file (`heart_disease_uci.csv`) is accessible at the path `/kaggle/input/heart-disease-data/`.
3.  Run the Jupyter notebook cells sequentially.

## Future Considerations

*   Advanced Feature Engineering and Selection.
*   Hyperparameter tuning for the top-performing classical models.
*   Exploring Deep Learning models (addressed in a separate notebook/folder).
