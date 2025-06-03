# Heart Disease Prediction Project

## Project Overview

This repository presents a comprehensive investigation into predicting heart disease using the UCI Heart Disease dataset. The project employs a dual-pronged approach, first establishing baseline performance with classical machine learning models after thorough Exploratory Data Analysis (EDA), and subsequently developing a more advanced Deep Learning (MLP) model incorporating techniques to handle class imbalance. The overarching goal is to build accurate and insightful predictive models for assessing heart disease risk based on patient attributes.

## Table of Contents

1.  [Project Objectives](#project-objectives)
2.  [Repository Structure](#repository-structure)
3.  [Methodology](#methodology)
    *   [Data Acquisition and Preparation](#data-acquisition-and-preparation)
    *   [Exploratory Data Analysis](#exploratory-data-analysis)
    *   [Data Preprocessing & Imputation](#data-preprocessing--imputation)
    *   [Classical Machine Learning Models](#classical-machine-learning-models)
    *   [Deep Learning Model with SMOTE](#deep-learning-model-with-smote)
4.  [Key Findings and Results](#key-findings-and-results)
5.  [Technologies Utilized](#technologies-utilized)
6.  [Setup and Usage](#setup-and-usage)
7.  [Future Work](#future-work)
8.  [Contributing](#contributing)
9.  [License](#license)

## Project Objectives

*   To conduct a detailed Exploratory Data Analysis (EDA) of the UCI Heart Disease dataset to identify key patterns, distributions, and relationships.
*   To implement robust data preprocessing techniques, including handling missing values through advanced imputation methods.
*   To train, evaluate, and compare a suite of classical machine learning models for heart disease prediction.
*   To develop a Deep Learning (Multi-Layer Perceptron) model, specifically addressing class imbalance using the SMOTE technique.
*   To provide a clear benchmark of model performance and identify areas for future improvement.

## Repository Structure

The project is organized as follows:

├── 01-Heart-Disease-EDA-Classical-ML/
│ ├── heart_disease_eda_classical_models.ipynb # EDA and classical ML model implementation
│ └── README.md # Detailed information for this section
├── 02-Heart-Disease-Deep-Learning-SMOTE/
│ ├── heart_disease_dl_smote_advanced.ipynb # Deep learning model with SMOTE
│ └── README.md # Detailed information for this section
├── .gitignore # Git ignore file
├── LICENSE.md # Project license
└── README.md # This main repository README


## Methodology

### Data Acquisition and Preparation
The project utilizes the combined Heart Disease dataset from the UCI Machine Learning Repository, amalgamating data from Cleveland, Hungary, Switzerland, and VA Long Beach. Initial preparation involves loading the data and addressing basic inconsistencies.

### Exploratory Data Analysis
A comprehensive EDA was performed to understand feature distributions, identify correlations, and gain insights into demographic and clinical factors associated with heart disease.

### Data Preprocessing & Imputation
*   Missing values were systematically addressed. Numerical features (`trestbps`, `ca`, `thalch`, `oldpeak`, `chol`) were imputed using `IterativeImputer`.
*   Categorical/boolean features (`thal`, `slope`, `fbs`, `exang`, `restecg`) were imputed using custom ML-based functions (in the classical ML notebook) or `SimpleImputer` (in the DL notebook).
*   Outliers were analyzed, and data errors (e.g., `trestbps == 0`) were rectified.

### Classical Machine Learning Models
(Covered in `01-Heart-Disease-EDA-Classical-ML/`)
*   **Feature Engineering:** Label encoding for categorical features.
*   **Modeling:** A pipeline including `KNNImputer` and `MinMaxScaler` was used to train and evaluate: Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
*   **Evaluation:** Performance was assessed using cross-validation accuracy and test set accuracy.

### Deep Learning Model with SMOTE
(Covered in `02-Heart-Disease-Deep-Learning-SMOTE/`)
*   **Preprocessing:** Numerical features were scaled using `StandardScaler`, and categorical features were one-hot encoded.
*   **Imbalance Handling:** SMOTE was applied to the training data to address significant class imbalance in the target variable.
*   **Model Architecture:** A Multi-Layer Perceptron (MLP) with Dense layers, Batch Normalization, Dropout, and L2 regularization was implemented.
*   **Training:** The model was trained using the Adam optimizer, `categorical_crossentropy` loss, and callbacks including `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.
*   **Evaluation:** Performance was measured by test loss, accuracy, AUC, a detailed classification report, and a confusion matrix.

## Key Findings and Results

*   **EDA Insights:** The dataset exhibits notable gender disparity (males ~79%) and age-related trends, with a peak heart disease occurrence around 54-55 years. Asymptomatic angina is the most common chest pain type.
*   **Classical Model Performance:** GradientBoostingClassifier emerged as the top-performing classical model with a test accuracy of approximately 65.58%.
*   **Deep Learning Performance:** The MLP model, after SMOTE and careful tuning, achieved a test accuracy of ~54.78% and an AUC of ~0.8430. While overall accuracy was lower than the best classical model, SMOTE helped in improving the recognition of minority classes, which was a key challenge.
*   **Model Artifacts:** The trained DL model (`best_heart_disease_dl_model.h5`) and its preprocessor (`dl_preprocessor.pkl`) are saved for reproducibility and future use.

*For detailed findings specific to each approach, please refer to the README files within the respective subdirectories.*

## Technologies Utilized

*   **Programming Language:** Python 3
*   **Core Libraries:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn, Plotly
*   **Machine Learning:** Scikit-learn (for preprocessing, imputation, classical models, metrics), XGBoost
*   **Deep Learning:** TensorFlow, Keras
*   **Imbalance Handling:** Imbalanced-learn (SMOTE)

## Setup and Usage

### Prerequisites
*   Python 3.8+
*   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [URL_OF_YOUR_REPOSITORY]
    cd heart-disease-prediction-repo
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. You can generate one from your environment using `pip freeze > requirements.txt` after installing all necessary packages locally.)*
    Alternatively, install packages individually:
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost tensorflow imbalanced-learn jupyter
    ```

4.  **Dataset:**
    The notebooks expect the dataset `heart_disease_uci.csv` to be located in a directory like `/kaggle/input/heart-disease-data/`. Please download the dataset from an appropriate source (e.g., UCI Machine Learning Repository) and adjust the file paths within the notebooks if necessary.

### Running the Notebooks
Navigate to the respective subdirectories (`01-Heart-Disease-EDA-Classical-ML` or `02-Heart-Disease-Deep-Learning-SMOTE`) and launch Jupyter Notebook:
```bash
jupyter notebook

Open and execute the cells in the .ipynb files.
Future Work
Advanced Feature Engineering: Explore interaction terms and domain-specific feature creation.
Hyperparameter Optimization: Conduct systematic tuning for both classical and deep learning models using tools like Optuna or KerasTuner.
Ensemble Modeling: Investigate stacking or blending of diverse models to potentially enhance predictive power.
Model Interpretability: Employ SHAP or LIME to gain deeper insights into model decision-making processes.
Deployment Strategy: Outline a potential pathway for deploying the most effective model as a predictive service.
Cross-Dataset Validation: Evaluate model generalization on other heart disease datasets.
Contributing
We welcome contributions to this project. If you'd like to contribute, please follow these steps:
Fork the repository.
Create a new branch (git checkout -b feature/your-feature-name).
Make your changes and commit them (git commit -m 'Add some feature').
Push to the branch (git push origin feature/your-feature-name).
Open a Pull Request.
Please ensure your code adheres to good practices and includes relevant documentation or tests where applicable.
License
This project is licensed under the MIT License - see the LICENSE.md file for details.
Copyright (c) [Year] [Your Name/Organization Name]
**Key features of this "Professional" README:**

*   **Clear Structure:** Uses a Table of Contents and well-defined sections.
*   **Concise Overview:** Quickly explains the project's purpose and main approaches.
*   **Methodology Detail:** Briefly outlines the key steps in each phase (EDA, classical ML, DL).
*   **Highlights Key Findings:** Summarizes the most important outcomes.
*   **Reproducibility Focus:** Provides clear setup and usage instructions, including dependency management (suggesting a `requirements.txt`).
*   **Forward-Looking:** Outlines potential future enhancements.
*   **Standard Sections:** Includes "Contributing" and "License" sections, common in open-source projects.

Remember to:

1.  Replace `[URL_OF_YOUR_REPOSITORY]` and `[Year] [Your Name/Organization Name]`.
2.  Create the `LICENSE.md` file (e.g., with the MIT license text).
3.  Generate and include a `requirements.txt` file in your repository's root for easy dependency installation by others. You can create it by running `pip freeze > requirements.txt` in your project's activated virtual environment after you've installed all the packages used by your notebooks.