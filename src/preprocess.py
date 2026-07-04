import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


# =====================================================
# Initialize NLP Tools
# =====================================================

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

tokenizer = RegexpTokenizer(r"\w+")


# =====================================================
# Lowercase
# =====================================================

def to_lowercase(text):
    return str(text).lower()


# =====================================================
# Remove URLs
# =====================================================

def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


# =====================================================
# Remove HTML Tags
# =====================================================

def remove_html(text):
    return re.sub(r"<.*?>", "", text)


# =====================================================
# Remove Punctuation
# =====================================================

def remove_punctuation(text):
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )


# =====================================================
# Remove Numbers
# =====================================================

def remove_numbers(text):
    return re.sub(r"\d+", "", text)


# =====================================================
# Tokenization
# =====================================================

def tokenize_text(text):
    return tokenizer.tokenize(text)


# =====================================================
# Remove Stopwords
# =====================================================

def remove_stopwords(tokens):

    return [
        word
        for word in tokens
        if word not in stop_words
    ]


# =====================================================
# Lemmatization
# =====================================================

def lemmatize_text(tokens):

    return [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]


# =====================================================
# Complete Preprocessing Pipeline
# =====================================================

def preprocess_text(text):

    text = to_lowercase(text)

    text = remove_urls(text)

    text = remove_html(text)

    text = remove_punctuation(text)

    text = remove_numbers(text)

    tokens = tokenize_text(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize_text(tokens)

    return " ".join(tokens)


# =====================================================
# Main Function
# =====================================================

def main():

    print("=" * 60)
    print("NLP PREPROCESSING")
    print("=" * 60)

    # Load cleaned dataset
    df = pd.read_csv(
        "data/processed/cleaned_news.csv"
    )

    print("\nDataset Loaded Successfully!")
    print("Dataset Shape:", df.shape)

    print("\nApplying NLP preprocessing...")

    df["content"] = df["content"].apply(preprocess_text)

    print("Done!")

    print("\nFirst 5 Rows:\n")
    print(df.head())

    print("\nChecking Missing Values:")
    print(df.isnull().sum())

    print("\nSaving preprocessed dataset...")
    # Remove empty documents

    df["content"] = df["content"].str.strip()

    df = df[df["content"] != ""]

    df.reset_index(drop=True, inplace=True)
    df.to_csv(
        "data/processed/preprocessed_news.csv",
        index=False
    )

    print("Dataset saved successfully!")

    print("\nFinal Dataset Shape:")
    print(df.shape)


if __name__ == "__main__":
    main()