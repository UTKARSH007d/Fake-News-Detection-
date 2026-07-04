import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def main():

    print("=" * 60)
    print("TF-IDF FEATURE ENGINEERING")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(
    "data/processed/preprocessed_news.csv")

    # Remove missing rows
    df.dropna(inplace=True)

    # Remove empty documents
    df = df[df["content"].str.strip() != ""]
    print(df.isnull().sum())

    print("\nEmpty strings:")
    print((df["content"] == "").sum())
    X = df["content"]
    y = df["label"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    X_test_tfidf = tfidf.transform(X_test)

    print("\nTF-IDF Completed!")

    print("Training Shape :", X_train_tfidf.shape)
    print("Testing Shape  :", X_test_tfidf.shape)

    # Save Vectorizer
    joblib.dump(
        tfidf,
        "models/vectorizers/tfidf_vectorizer.pkl"
    )

    print("\nTF-IDF Vectorizer Saved!")

    # Save Processed Data
    joblib.dump(
        X_train_tfidf,
        "models/trained_models/X_train.pkl"
    )

    joblib.dump(
        X_test_tfidf,
        "models/trained_models/X_test.pkl"
    )

    joblib.dump(
        y_train,
        "models/trained_models/y_train.pkl"
    )

    joblib.dump(
        y_test,
        "models/trained_models/y_test.pkl"
    )

    print("Training/Test Data Saved Successfully!")


if __name__ == "__main__":
    main()