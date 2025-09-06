## 1. Load Libraries and Dataset
# ---------------------------------------------------------------------------------------------------
# importing required libraries numpy, pandas, seaborn, matplotlib.pyplot for data load and analysis
# importing train_test_split for data split, LabelEncoder to encode data to numeric,
# StandardScaler to feature scaling and Algorithms (LogisticRegression, KNeighboursClassifier,
# DecisionTreeClassifier, PlotTree, RandomForestClassifier) for model
# importing accuracy_score, classification_report, confusion_matrix
# ---------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


# --------------------------------------------------------
# Custom function to print data in a nice way in result
# --------------------------------------------------------
def custom_print(label,values):
  print(label)
  print("")
  print(values)
  print("")
  print("")
  print("")

# --------------------------------------------------------
# Loading Diabetes Datasaet with provided csv file
# --------------------------------------------------------

diabetes_df = pd.read_csv('content/2025-09-01T04-03-42.625Z-2025-07-04T13-24-57.561Z-diabetes.csv')

custom_print("# Printing 1st 5 Rows from the dataframe", diabetes_df.head())

# --------------------------------------------------------

## 2. EDA of Dataset

custom_print("# Total Rows", len(diabetes_df))
custom_print("# Column Names", diabetes_df.columns)
custom_print("# Total Columns", len(diabetes_df.columns))
custom_print("# Checking Data detail info", diabetes_df.info())
custom_print("# I think every column is important, so I am not going to drop any column parmanently.","")
custom_print("# From Dataset detail Info, we have come to know that all columns has numeric value. So, no need to Convert categorical variables.","")
custom_print("# No Columns has any null value. So, this dataset do not need to handle missing values. See Null value count for each column.",diabetes_df.isnull().sum())

custom_print("# Outcome Column binary value count. Not balanced.", diabetes_df['Outcome'].value_counts())
sns.countplot(x='Outcome', data=diabetes_df)
plt.title('Outcome Count')
plt.show()

# --------------------------------------------------------
# Custom function to show histplots
# --------------------------------------------------------

def show_histplot(title, data, x, hue, bins):
  sns.histplot(data=data, x=x, hue=hue, multiple='stack', bins=bins)
  plt.title(title)
  plt.show()

for column in diabetes_df.columns:
  if column != 'Outcome':
    custom_print(f"Outcome ratio by {column}", diabetes_df.groupby(column)['Outcome'].value_counts(normalize=True).unstack())
    show_histplot(f"Histogram: Outcome by {column}", diabetes_df, column, "Outcome", 30)

# --------------------------------------------------------

## 3. Train/Test Split
# -----------------------------
# Prepare Features and Target
# -----------------------------
custom_print("# diabetes_df total columns", len(diabetes_df.columns))
x = diabetes_df.drop("Outcome", axis=1)  # dropping target column Outcome. Keeping all features except target in X
y = diabetes_df["Outcome"]               # Y= target column Outcome (Binary Diabates Outcome 0 or 1)


custom_print("# After Dropping Outcome column, X total columns", len(x.columns))
custom_print("# total rows of x before split", len(x))
custom_print("# X 1st 5 rows", x.head())
custom_print("# Y or Outcome", y)

# -------------------------------------------------
# Train/Test Split 80% Train Data, 20% Test Data
# -------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

custom_print("# Study after train/test split", "")
custom_print("# total rows of x_train after split", f"{len(x_train)}  {round((len(x_train)*100)/len(x),3)}%")
custom_print("# total rows of x_test after split", f"{len(x_test)}  {round((len(x_test)*100)/len(x),3)}%")
custom_print("# X_train 1st 5 rows", x_train)
custom_print("# Y_train or Outcome", y_train)
y_test_value_counts = y_test.value_counts()
custom_print("# Y_test Outcome Column binary value count.", y_test_value_counts)

# --------------------------------------------------------

## 4. Feature Scaling

# ----------------------------------
# Scaling train/test X columns value
# ----------------------------------

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert back to DataFrame for better readability (optional)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)


custom_print("# Scaled X_train 1st 5 rows", x_train_scaled.head())
custom_print("# Scaled X_test 1st 5 rows", x_test_scaled.head())

custom_print("# Scaled X_train Data Shape", x_train_scaled.shape)
custom_print("# Scaled X_test Data Shape", x_test_scaled.shape)

# --------------------------------------------------------

## 5. Logistic Regression

# --------------
# Training Model
# --------------
logistic_regression_model = LogisticRegression(max_iter=200)
logistic_regression_model.fit(x_train_scaled, y_train)

# -------------------------------
# Model Prediction with Test Data 
# -------------------------------
logistic_regression_model_predictions = logistic_regression_model.predict(x_test_scaled)
custom_print("# Logistic Regression Model Predictions", logistic_regression_model_predictions)
# ----------------------------------------------
# Test Accuracy Score and Classification Report
# ----------------------------------------------
logistic_regression_model_predictions_value_counts = pd.Series(logistic_regression_model_predictions).value_counts()
custom_print("# Logistic Regression Model Predictions Value Counts", logistic_regression_model_predictions_value_counts)
logistic_regression_model_confusion_matrix = confusion_matrix(y_test, logistic_regression_model_predictions)
custom_print("# Logistic Regression Model Confusion Matrix", logistic_regression_model_confusion_matrix)
logistic_regression_model_accuracy_score = accuracy_score(y_test, logistic_regression_model_predictions)
custom_print("# Logistic Regression Model Accuracy", logistic_regression_model_accuracy_score)
custom_print("# Logistic Regression Model Classification Report", classification_report(y_test, logistic_regression_model_predictions))

## 6. KNN
# ----------------------------------------------
# Test Accuracy Score and Classification Report
# ----------------------------------------------
errors = []
k_values = []  # store odd k values
for k in range(1, 25):
  if k % 2 != 0: # ignoring even values for k as sometimes even values create confusion
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_scaled, y_train)
    pred_k = knn.predict(x_test_scaled)
    errors.append(1 - accuracy_score(y_test, pred_k))
    k_values.append(k)   # record the odd k used

# ----------------------------------------------
# Measuring value of K where Error rate is Lowest
# ----------------------------------------------

custom_print("Errors", errors)
custom_print("Errors Length", len(errors))

plt.plot(k_values, errors, marker='o')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('Choosing K')
plt.show()

best_error = min(errors)
best_k = k_values[errors.index(best_error)]

custom_print(f"Best K = {best_k} with Error Rate = {best_error:.4f}","")

# --------------
# Training Model
# As Best K with Lowest Error = 21
# I will take k=n_neighbors=21
# --------------
knn_model = KNeighborsClassifier(n_neighbors=21)
knn_model.fit(x_train_scaled, y_train)

# -------------------------------
# Model Prediction with Test Data 
# -------------------------------
knn_model_predictions = knn_model.predict(x_test_scaled)
custom_print("# KNN Model Predictions", knn_model_predictions)
# ----------------------------------------------
# Test Accuracy Score and Classification Report
# ----------------------------------------------
knn_model_predictions_value_counts = pd.Series(knn_model_predictions).value_counts()
custom_print("# KNN Model Predictions Value Counts", knn_model_predictions_value_counts)
knn_model_confusion_matrix = confusion_matrix(y_test, knn_model_predictions)
custom_print("# KNN Model Confusion Matrix", knn_model_confusion_matrix)
knn_model_accuracy_score = accuracy_score(y_test, knn_model_predictions)
custom_print("# KNN Model Accuracy", knn_model_accuracy_score)
custom_print("# KNN Model Classification Report", classification_report(y_test, knn_model_predictions))
# --------------------------------------------------------

## 7. Decision Tree
# --------------
# Training Model
# --------------
decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(x_train_scaled, y_train)
# -------------------------------
# Model Prediction with Test Data 
# -------------------------------
decision_tree_model_predictions = decision_tree_model.predict(x_test_scaled)
custom_print("# Decision Tree Model Predictions", decision_tree_model_predictions)
# ----------------------------------------------
# Test Accuracy Score and Classification Report
# ----------------------------------------------
decision_tree_model_predictions_value_counts = pd.Series(decision_tree_model_predictions).value_counts()
custom_print("# Decision Tree Model Predictions Value Counts", decision_tree_model_predictions_value_counts)
decision_tree_model_confusion_matrix = confusion_matrix(y_test, decision_tree_model_predictions)
custom_print("# Decision Tree Model Confusion Matrix", decision_tree_model_confusion_matrix)
decision_tree_model_accuracy_score = accuracy_score(y_test, decision_tree_model_predictions)
custom_print("# Decision Tree Model Accuracy", decision_tree_model_accuracy_score)
custom_print("# Decision Tree Model Classification Report", classification_report(y_test, decision_tree_model_predictions))
# Visualize Tree
plt.figure(figsize=(100,70))
plot_tree(decision_tree_model, filled=True, feature_names=x.columns)
plt.show()
## 8. Random Forest
# --------------
# Training Model
# --------------
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(x_train_scaled, y_train)
# -------------------------------
# Model Prediction with Test Data 
# -------------------------------
random_forest_model_predictions = random_forest_model.predict(x_test_scaled)
custom_print("# Random Forest Model Predictions", random_forest_model_predictions)
# ----------------------------------------------
# Test Accuracy Score and Classification Report
# ----------------------------------------------
random_forest_model_predictions_value_counts = pd.Series(random_forest_model_predictions).value_counts()
custom_print("# Random Forest Model Predictions Value Counts", random_forest_model_predictions_value_counts)
random_forest_model_confusion_matrix = confusion_matrix(y_test, random_forest_model_predictions)
custom_print("# Random Forest Model Confusion Matrix", random_forest_model_confusion_matrix)
random_forest_model_accuracy_score = accuracy_score(y_test, random_forest_model_predictions)
custom_print("# Random Forest Model Accuracy", random_forest_model_accuracy_score)
custom_print("# Random Forest Model Classification Report", classification_report(y_test, random_forest_model_predictions))
# ----------------------------------
# Taking comparing value in a array
# ----------------------------------

models = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']
accuracies = [logistic_regression_model_accuracy_score, knn_model_accuracy_score,
              decision_tree_model_accuracy_score, random_forest_model_accuracy_score]

# --------------------------------------------
# Showing Bar Plot Graph for compare accuracy
# --------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# ---------------------------------------
# Comparison Models with ROC curve & AUC
# ---------------------------------------

prediction_results={
    'Logistic Regression Model': (logistic_regression_model, logistic_regression_model_predictions),
    'KNN Model': (knn_model, knn_model_predictions),
    'Decision Tree Model': (decision_tree_model, decision_tree_model_predictions),
    'Random Forest Model': (random_forest_model, random_forest_model_predictions)
}

# custom_print("# Prediction Results", prediction_results)

plt.figure(figsize=(10, 6))
for name, (model, _) in prediction_results.items():
    y_prob = model.predict_proba(x_test)[:, 1] #Predicting Probabilities 
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.show()