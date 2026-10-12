
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Load the dataset
print("Loading dataset...")
data = pd.read_csv("ai4i2020.csv")

print("Dataset loaded:", data.shape)

# 2. Remove columns that do not help predict failure
data = data.drop(
    columns=["UDI", "Product ID"],
    errors="ignore"
)

# 3. Select the target column
# Machine failure: 0 = No failure, 1 = Failure
target = "Machine failure"

if target not in data.columns:
    raise ValueError(
        "Machine failure column not found. "
        "Check the CSV column names."
    )

X = data.drop(columns=[target])
y = data[target].astype(int)

# Remove other failure-type columns from the input.
# They describe failure outcomes and could leak the answer.
failure_columns = [
    "TWF", "HDF", "PWF", "OSF", "RNF"
]
X = X.drop(
    columns=[c for c in failure_columns if c in X.columns],
    errors="ignore"
)

# 4. Identify text and numeric columns
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 5. Prepare the data-processing steps
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Compare multiple machine-learning algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1 Score (%)": round(f1 * 100, 2)
    })

    trained_models[name] = pipeline

# 8. Display model comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score (%)",
    ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 9. Select the model with the highest F1 score
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

best_predictions = best_model.predict(X_test)

print("\nBEST MODEL:", best_name)
print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, best_predictions))

# 10. Save the trained model
joblib.dump(best_model, "best_model.joblib")
results_df.to_csv("model_results.csv", index=False)

print("\nSaved model as best_model.joblib")
print("Saved comparison as model_results.csv")
print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
