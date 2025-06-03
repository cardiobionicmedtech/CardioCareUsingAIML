# Advanced Heart Disease Prediction: Deep Learning with SMOTE

## Overview

This notebook builds upon previous explorations by applying Deep Learning (DL) techniques to the UCI Heart Disease dataset. The primary focus is to develop a Multi-Layer Perceptron (MLP) model and address the significant class imbalance present in the target variable (`num`, representing heart disease severity).

## Dataset

The data used is the combined Heart Disease dataset from the UCI Machine Learning Repository. This notebook assumes initial data loading and some primary cleaning/imputation steps are similar to those in the classical ML approach.

## Workflow

1.  **Data Loading & Refined Imputation:**
    *   Loading the dataset.
    *   `IterativeImputer` (with `RandomForestRegressor`) for initial numerical column imputation.
    *   `SimpleImputer` (strategy='most_frequent') for remaining categorical and boolean columns to ensure a clean dataset for DL preprocessing and SMOTE.
    *   Removal of rows with `trestbps == 0`.
2.  **DL-Specific Preprocessing:**
    *   Separating features (X) and target (y).
    *   Identifying numerical, categorical, and boolean features.
    *   Converting boolean features to integers (0/1).
    *   Applying `StandardScaler` to numerical features.
    *   Applying `OneHotEncoder` (with `drop='first'`) to categorical features.
3.  **Train-Test Split:** Splitting the data into training and testing sets using stratification to maintain class proportions.
4.  **Handling Class Imbalance with SMOTE:**
    *   Applying the Synthetic Minority Over-sampling Technique (SMOTE) **only to the training data** to create a more balanced dataset for training the DL model.
    *   Dynamically adjusting `k_neighbors` for SMOTE based on minority class sizes.
5.  **Target Variable Encoding:** One-hot encoding the target variable (`y_train_smote` and `y_test`) for use with `categorical_crossentropy` loss.
6.  **Deep Learning Model (MLP):**
    *   **Architecture:** A Sequential Keras model with multiple Dense layers, Batch Normalization, Dropout, and L2 kernel regularization.
    *   **Activation Functions:** ReLU for hidden layers, Softmax for the output layer (multi-class classification).
    *   **Compilation:** Using the Adam optimizer, `categorical_crossentropy` loss, and tracking accuracy and AUC.
7.  **Model Training:**
    *   Training the MLP on the SMOTE-augmented training data.
    *   Employing callbacks:
        *   `EarlyStopping` (monitoring `val_auc`) to prevent overfitting and restore best weights.
        *   `ReduceLROnPlateau` to adjust learning rate if validation loss stagnates.
        *   `ModelCheckpoint` to save the best model based on `val_auc` (using `.h5` format to avoid potential environment-specific issues with the default `.keras` format).
8.  **Model Evaluation:**
    *   Loading the best saved model.
    *   Evaluating performance on the (unseen) test set using loss, accuracy, and AUC.
    *   Generating a detailed classification report and confusion matrix to assess per-class performance.
9.  **Visualization:** Plotting training and validation accuracy, loss, and AUC over epochs.
10. **Saving Artifacts:** Saving the fitted preprocessor pipeline (`dl_preprocessor.pkl`) for future use.

## Key Findings & Results from Deep Learning Approach

*   **Initial Challenge:** The target variable (`num`) showed significant class imbalance, with class 4 (critical heart disease) being severely underrepresented. This was a major challenge for the initial DL model.
*   **SMOTE Application:** SMOTE was applied to the training data to address this imbalance, resulting in a more balanced training set for the neural network.
*   **Model Performance (Post-SMOTE and Tuning):**
    *   **Test Accuracy:** Approximately **54.78%**
    *   **Test AUC:** Approximately **0.8430**
    *   **Classification Report Insights:**
        *   Class 0 (no disease): Precision 0.86, Recall 0.79, F1-score 0.82
        *   Class 1: Precision 0.55, Recall 0.41, F1-score 0.47
        *   Class 2: Precision 0.19, Recall 0.22, F1-score 0.20
        *   Class 3: Precision 0.26, Recall 0.41, F1-score 0.32
        *   Class 4: Precision 0.08, Recall 0.14, F1-score 0.10
    *   **Observations:** While the overall accuracy might not have surpassed all classical models, the application of SMOTE and the use of AUC as a monitoring metric aimed to improve the model's ability to recognize minority classes. The classification report shows that class 4, while still challenging, is no longer completely missed by the model, indicating some improvement in handling imbalance compared to the initial unweighted DL attempt. The AUC of ~0.84 is a reasonable score for this complex multi-class problem.
*   **Training Stability:** Callbacks like `EarlyStopping` and `ReduceLROnPlateau` were crucial in managing the training process and preventing severe overfitting.
*   **Saving Artifacts:** The best model and the preprocessor were saved, enabling future predictions on new data.

## Technologies Used

*   Python 3
*   Pandas & NumPy
*   Matplotlib, Seaborn
*   Scikit-learn (Preprocessing, Imputation, Metrics, SMOTE via `imblearn`)
*   TensorFlow & Keras (Deep Learning Model)

## How to Use

1.  Ensure all required libraries are installed.
2.  Place the dataset (`heart_disease_uci.csv`) in the `/kaggle/input/heart-disease-data/` directory (or update the path).
3.  Run the Jupyter notebook cells sequentially. The best model will be saved as `best_heart_disease_dl_model.h5` and the preprocessor as `dl_preprocessor.pkl`.

## Future Directions

*   Extensive hyperparameter tuning for the MLP (e.g., network architecture, dropout rates, L2 strength, optimizer parameters).
*   Exploring other oversampling/undersampling techniques or combinations.
*   Trying different DL architectures if MLP performance plateaus.
*   Feature selection specifically tailored for the DL model.
