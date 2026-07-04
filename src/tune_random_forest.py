import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

# -----------------------------------------
# Load Dataset
# -----------------------------------------

X_train = joblib.load("models/trained_models/X_train.pkl")
X_test = joblib.load("models/trained_models/X_test.pkl")

y_train = joblib.load("models/trained_models/y_train.pkl")
y_test = joblib.load("models/trained_models/y_test.pkl")

print("\nDataset Loaded Successfully!")

# -----------------------------------------
# Base Model
# -----------------------------------------

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# -----------------------------------------
# Hyperparameters to Search
# -----------------------------------------

param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [20, 40, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4]

}

# -----------------------------------------
# Random Search
# -----------------------------------------

print("\nSearching Best Parameters...")

search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_grid,

    n_iter=10,

    cv=3,

    scoring="accuracy",

    random_state=42,

    n_jobs=-1

)

search.fit(X_train, y_train)

print("\nSearch Completed!")

# -----------------------------------------
# Best Parameters
# -----------------------------------------

print("\nBest Parameters:")

print(search.best_params_)

# -----------------------------------------
# Best Model
# -----------------------------------------

best_model = search.best_estimator_

# -----------------------------------------
# Prediction
# -----------------------------------------

y_pred = best_model.predict(X_test)

# -----------------------------------------
# Evaluation
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\nModel Performance")

print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -----------------------------------------
# Save Model
# -----------------------------------------

joblib.dump(

    best_model,

    "models/trained_models/random_forest_tuned.pkl"

)

print("Tuned Model Saved Successfully!")