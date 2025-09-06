# Diabetes Prediction ML Project

This project demonstrates an end-to-end **Machine Learning pipeline** to
predict diabetes outcomes using multiple algorithms. The dataset used is
the **Pima Indians Diabetes Dataset**.

------------------------------------------------------------------------

## 📌 Project Workflow

### 1. Load Libraries and Dataset

-   Imported essential libraries: **pandas, numpy, seaborn, matplotlib**
-   ML tools from **scikit-learn**:
    -   Preprocessing: `train_test_split`, `StandardScaler`,
        `LabelEncoder`
    -   Models: `LogisticRegression`, `KNeighborsClassifier`,
        `DecisionTreeClassifier`, `RandomForestClassifier`
    -   Evaluation: `accuracy_score`, `classification_report`,
        `confusion_matrix`, `roc_curve`, `roc_auc_score`

### 2. Exploratory Data Analysis (EDA)

-   Displayed dataset info, shape, null values, and column details.
-   Plotted outcome count using **Seaborn** countplot.
-   Generated **histograms** for each feature against the target
    `Outcome`.

### 3. Train/Test Split

-   80% data for training, 20% for testing.
-   Target variable: `Outcome`
-   Features: All other columns

### 4. Feature Scaling

-   Applied **StandardScaler** to normalize the features.

### 5. Logistic Regression

-   Trained logistic regression with `max_iter=200`
-   Evaluated predictions using:
    -   Confusion Matrix
    -   Accuracy Score
    -   Classification Report

### 6. K-Nearest Neighbors (KNN)

-   Tested odd `k` values from **1 to 25**
-   Selected the **best k** with lowest error rate
-   Evaluated with accuracy and classification report
-   Visualized error rates vs `k`

### 7. Decision Tree Classifier

-   Trained Decision Tree with `max_depth=6`
-   Evaluated predictions with metrics
-   Visualized decision tree with `plot_tree`

### 8. Random Forest Classifier

-   Trained with `n_estimators=100` and `random_state=42`
-   Evaluated predictions with metrics

### 9. Model Comparison

-   Compared models with a **bar chart of accuracy scores**
-   Plotted **ROC Curves** for all models with **AUC scores**

------------------------------------------------------------------------

## 📊 Models Used

1.  Logistic Regression
2.  K-Nearest Neighbors (KNN)
3.  Decision Tree Classifier
4.  Random Forest Classifier

------------------------------------------------------------------------

## 🔎 Visualizations

-   Countplot for Outcome distribution
-   Histograms for each feature vs Outcome
-   Error Rate vs K for KNN
-   Decision Tree visualization
-   Accuracy comparison bar chart
-   ROC Curve comparison

------------------------------------------------------------------------

## 🚀 How to Run

1.  Clone this repository

2.  Create a Python virtual environment:

    ``` bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Run the script:

    ``` bash
    python diabetes_prediction.py
    ```

------------------------------------------------------------------------

## 📂 Files

-   `diabetes_prediction.py` → Main ML pipeline script
-   `requirements.txt` → Python dependencies
-   `README.md` → Project Documentation

------------------------------------------------------------------------

## 📈 Sample Results

-   Accuracy scores of models are displayed in a bar chart
-   ROC curves with AUC values provide a visual model comparison

------------------------------------------------------------------------

## ✅ Conclusion

This project highlights how different ML algorithms perform on the
diabetes dataset. Random Forest and Logistic Regression generally
perform better, while KNN depends heavily on the choice of `k`. The
Decision Tree provides interpretability but may risk overfitting.

Colab Link: https://colab.research.google.com/drive/12l1glyyUYVSP5A-nhHIk5OHRsnIcZpZb?usp=sharing
Github Link: https://github.com/ToukirAhamedPigeon/diabetes_detection_model