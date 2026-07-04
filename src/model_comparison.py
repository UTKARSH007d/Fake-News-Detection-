import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# Model Results
# -----------------------------------------

results = {

    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "Support Vector Machine"
    ],

    "Accuracy": [
        0.9487,
        0.8539,
        0.9577,
        0.9541
    ],

    "Precision": [
        0.9430,
        0.8412,
        0.9465,
        0.9481
    ],

    "Recall": [
        0.9582,
        0.8826,
        0.9728,
        0.9635
    ],

    "F1 Score": [
        0.9505,
        0.8614,
        0.9595,
        0.9557
    ]

}

df = pd.DataFrame(results)

print(df)

# -----------------------------------------
# Save CSV
# -----------------------------------------

df.to_csv(
    "model_results/model_comparison.csv",
    index=False
)

print("\nComparison table saved!")

# -----------------------------------------
# Accuracy Plot
# -----------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "model_results/model_accuracy_comparison.png",
    dpi=300
)

plt.show()

print("Comparison chart saved!")