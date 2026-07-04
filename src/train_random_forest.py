import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)

# -----------------------------------------
# Load Data
# -----------------------------------------

X_train = joblib.load("models/trained_models/X_train.pkl")
X_test = joblib.load("models/trained_models/X_test.pkl")

y_train = joblib.load("models/trained_models/y_train.pkl")
y_test = joblib.load("models/trained_models/y_test.pkl")

print("\nDataset Loaded Successfully!")

# -----------------------------------------
# Train Model
# -----------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed!")

# -----------------------------------------
# Prediction
# -----------------------------------------

y_pred = model.predict(X_test)

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
# Confusion Matrix
# -----------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "images/confusion_matrices/random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion Matrix Saved!")

# -----------------------------------------
# Save Model
# -----------------------------------------

joblib.dump(
    model,
    "models/trained_models/random_forest.pkl"
)

print("Model Saved Successfully!")