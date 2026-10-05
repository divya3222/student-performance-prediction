"""
Student Performance Prediction using Machine Learning
Dataset: student-por.csv
Project: Data Science Master Virtual Internship

The dataset contains 649 Portuguese-student records and 33 columns.
Target for regression: G3 (final grade, 0-20)
Target for classification: At_Risk = 1 when G3 < 10

Run in Google Colab/Jupyter:
    1. Upload student-por.csv
    2. Run this script/cells.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================
FILE_PATH = "student-por.csv"

df = pd.read_csv(FILE_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove exact duplicates if present
df = df.drop_duplicates().copy()

# ============================================================
# 2. BASIC DATA ANALYSIS
# ============================================================
print("\nDescriptive statistics:")
print(df.describe())

print("\nFinal grade distribution:")
print(df["G3"].value_counts().sort_index())

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Final grade distribution
plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=20, edgecolor="black")
plt.title("Distribution of Final Student Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Study time vs final grade
plt.figure(figsize=(7, 5))
groups = [df.loc[df["studytime"] == level, "G3"] for level in sorted(df["studytime"].unique())]
plt.boxplot(groups, labels=sorted(df["studytime"].unique()))
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()

# Absences vs final grade
plt.figure(figsize=(8, 5))
plt.scatter(df["absences"], df["G3"], alpha=0.7)
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()

# Previous grades vs final grade
plt.figure(figsize=(7, 5))
plt.scatter(df["G2"], df["G3"], alpha=0.7)
plt.title("Previous Grade (G2) vs Final Grade (G3)")
plt.xlabel("G2 - Second Period Grade")
plt.ylabel("G3 - Final Grade")
plt.tight_layout()
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(12, 9))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# 4. REGRESSION: PREDICT FINAL GRADE (G3)
# ============================================================
# G3 is the target. It must NOT be included as an input feature.
X = df.drop(columns=["G3"])
y = df["G3"]

numeric_features = X.select_dtypes(exclude="object").columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

# Preprocessing:
# - numerical: median imputation + standardization
# - categorical: most-frequent imputation + one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=6, random_state=42
    ),
    "K-Nearest Neighbors": KNeighborsRegressor(
        n_neighbors=7
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
}

regression_results = []
regression_pipelines = {}

print("\n" + "=" * 70)
print("REGRESSION MODEL RESULTS")
print("=" * 70)

for name, model in regression_models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    regression_results.append([name, mae, rmse, r2])
    regression_pipelines[name] = pipeline

    print(f"\n{name}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

regression_results_df = pd.DataFrame(
    regression_results,
    columns=["Model", "MAE", "RMSE", "R2"]
).sort_values("R2", ascending=False)

print("\nRegression comparison:")
print(regression_results_df)

# Best regression model
best_regression_name = regression_results_df.iloc[0]["Model"]
best_regression_model = regression_pipelines[best_regression_name]

print("\nBest regression model:", best_regression_name)

# Actual vs predicted graph
best_predictions = best_regression_model.predict(X_test)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_predictions, alpha=0.7)
plt.plot([0, 20], [0, 20], linestyle="--")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title(f"Actual vs Predicted Final Grade - {best_regression_name}")
plt.tight_layout()
plt.show()

# ============================================================
# 5. CLASSIFICATION: IDENTIFY STUDENTS AT ACADEMIC RISK
# ============================================================
# Define an academic-risk class:
# G3 < 10 -> At Risk
# G3 >= 10 -> Not At Risk
#
# This threshold can be changed if your institution uses another rule.

df["At_Risk"] = (df["G3"] < 10).astype(int)

X_class = df.drop(columns=["G3", "At_Risk"])
y_class = df["At_Risk"]

numeric_class = X_class.select_dtypes(exclude="object").columns.tolist()
categorical_class = X_class.select_dtypes(include="object").columns.tolist()

class_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_class
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_class
        )
    ]
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

classification_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, random_state=42
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
}

classification_results = []
classification_pipelines = {}

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL RESULTS")
print("=" * 70)

for name, model in classification_models.items():

    pipeline = Pipeline([
        ("preprocessor", class_preprocessor),
        ("model", model)
    ])

    pipeline.fit(Xc_train, yc_train)
    class_predictions = pipeline.predict(Xc_test)

    accuracy = accuracy_score(yc_test, class_predictions)
    precision = precision_score(
        yc_test, class_predictions, zero_division=0
    )
    recall = recall_score(
        yc_test, class_predictions, zero_division=0
    )
    f1 = f1_score(
        yc_test, class_predictions, zero_division=0
    )

    classification_results.append(
        [name, accuracy, precision, recall, f1]
    )
    classification_pipelines[name] = pipeline

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")

classification_results_df = pd.DataFrame(
    classification_results,
    columns=[
        "Model", "Accuracy", "Precision",
        "Recall", "F1"
    ]
).sort_values("F1", ascending=False)

print("\nClassification comparison:")
print(classification_results_df)

best_classification_name = classification_results_df.iloc[0]["Model"]
best_classification_model = classification_pipelines[
    best_classification_name
]

print("\nBest classification model:", best_classification_name)

# Confusion matrix for best classifier
best_class_predictions = best_classification_model.predict(Xc_test)

cm = confusion_matrix(yc_test, best_class_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not At Risk", "At Risk"],
    yticklabels=["Not At Risk", "At Risk"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_classification_name}")
plt.tight_layout()
plt.show()

print("\nClassification report:")
print(
    classification_report(
        yc_test,
        best_class_predictions,
        target_names=["Not At Risk", "At Risk"],
        zero_division=0
    )
)

# ============================================================
# 6. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
# This section explains which variables contribute most to the
# Random Forest regression model.

rf_pipeline = regression_pipelines["Random Forest"]

rf_model = rf_pipeline.named_steps["model"]
rf_preprocessor = rf_pipeline.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nTop 15 features:")
print(importance_df.head(15))

plt.figure(figsize=(9, 6))
top_features = importance_df.head(15).sort_values("Importance")
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features - Random Forest")
plt.tight_layout()
plt.show()

# ============================================================
# 7. EXAMPLE STUDENT PREDICTION
# ============================================================
# Select one record from the test set as an example.
sample_student = X_test.iloc[[0]]

predicted_grade = best_regression_model.predict(sample_student)[0]
predicted_risk = best_classification_model.predict(
    sample_student
)[0]

print("\n" + "=" * 70)
print("SAMPLE STUDENT PREDICTION")
print("=" * 70)

print("Predicted Final Grade (G3):", round(predicted_grade, 2))

if predicted_risk == 1:
    print("Academic Risk Status: AT RISK")
else:
    print("Academic Risk Status: NOT AT RISK")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
regression_results_df.to_csv(
    "regression_model_results.csv", index=False
)

classification_results_df.to_csv(
    "classification_model_results.csv", index=False
)

importance_df.to_csv(
    "feature_importance.csv", index=False
)

print("\nResult files saved:")
print("- regression_model_results.csv")
print("- classification_model_results.csv")
print("- feature_importance.csv")

print("\nProject execution completed successfully.")
