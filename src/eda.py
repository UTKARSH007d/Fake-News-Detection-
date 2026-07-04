import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Additional news-specific words to ignore
custom_stopwords = {
    "said", "would", "could", "also", "one", "two",
    "new", "first", "last", "mr", "mrs", "ms",
    "u", "us", "state", "states",
    "people", "government", "president",
    "republican", "democrat",
    "year", "years", "time",
    "say", "reuters",
    "even", "many", "may",
    "get", "make", "like",
    "clinton", "trump", "obama", "donald",
    "hillary"
}

stop_words.update(custom_stopwords)
# Load cleaned dataset
df = pd.read_csv("data/processed/preprocessed_news.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# -----------------------------
# Dataset Summary
# -----------------------------
print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes.to_string())

print("\nMissing Values:")
print(df.isnull().sum().to_string())

print("\nLabel Distribution:")
print(df["label"].value_counts().to_string())

# -----------------------------
# Class Distribution Plot
# -----------------------------
label_counts = df["label"].value_counts()

plt.figure(figsize=(8,5))
plt.bar(["Real", "Fake"], [label_counts[1], label_counts[0]])

plt.title("Fake vs Real News Distribution")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    "images/eda/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nClass distribution chart saved successfully!")
# =====================================================
# Word Count Analysis
# =====================================================

df["word_count"] = df["content"].apply(lambda x: len(x.split()))

print("\nWord Statistics")
print("-" * 40)

print(df["word_count"].describe())

plt.figure(figsize=(8,5))

plt.hist(df["word_count"], bins=50)

plt.title("Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    "images/eda/word_count_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# =====================================================
# Average Word Count by Class
# =====================================================

average_word_count = df.groupby("label")["word_count"].mean()

print("\nAverage Word Count by Class")
print("-" * 40)
average_word_count.index = ["Fake", "Real"]

print(average_word_count.to_string())

plt.figure(figsize=(8, 5))

plt.bar(
    average_word_count.index,
    average_word_count.values
)
plt.title("Average Word Count by News Type")
plt.xlabel("News Type")
plt.ylabel("Average Number of Words")

plt.tight_layout()

plt.savefig(
    "images/eda/average_word_count_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# =====================================================
# Top 20 Most Frequent Words
# =====================================================

# Separate Fake and Real news
fake_news = df[df["label"] == 0]["content"]
real_news = df[df["label"] == 1]["content"]


def get_top_words(text_series):

    words = " ".join(text_series).lower().split()

    words = [
        word
        for word in words
        if word.isalpha() and len(word) > 2
]

    words = [word for word in words if word not in stop_words]

    return Counter(words).most_common(20)


fake_top_words = get_top_words(fake_news)
real_top_words = get_top_words(real_news)
fake_words = [word for word, count in fake_top_words]
fake_counts = [count for word, count in fake_top_words]

plt.figure(figsize=(10,6))

plt.bar(fake_words, fake_counts)

plt.title("Top 20 Words in Fake News")

plt.xlabel("Words")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "images/eda/top20_fake_words.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
real_words = [word for word, count in real_top_words]
real_counts = [count for word, count in real_top_words]

plt.figure(figsize=(10,6))

plt.bar(real_words, real_counts)

plt.title("Top 20 Words in Real News")

plt.xlabel("Words")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "images/eda/top20_real_words.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ==========================================================
# WORD CLOUD - FAKE NEWS
# ==========================================================

from wordcloud import WordCloud

fake_text = " ".join(df[df["label"] == 1]["content"])

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(fake_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title("Fake News Word Cloud")

plt.savefig(
    "images/eda/fake_wordcloud.png",
    dpi=300
)

plt.show()

print("Fake WordCloud Saved!")
# ==========================================================
# WORD CLOUD - REAL NEWS
# ==========================================================

real_text = " ".join(real_news)

real_cloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stop_words
).generate(real_text)

plt.figure(figsize=(12,6))

plt.imshow(real_cloud, interpolation="bilinear")

plt.axis("off")

plt.title("Real News Word Cloud")

plt.savefig(
    "images/eda/real_wordcloud.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Real News WordCloud Saved!")
# ==========================================================
# CHARACTER COUNT
# ==========================================================

df["character_count"] = df["content"].apply(len)

plt.figure(figsize=(8,6))

plt.hist(df[df["label"]==0]["character_count"],
         bins=50,
         alpha=0.6,
         label="Fake")

plt.hist(df[df["label"]==1]["character_count"],
         bins=50,
         alpha=0.6,
         label="Real")

plt.title("Character Count Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.legend()

plt.savefig(
    "images/eda/character_count_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# WORD COUNT
# ==========================================================

df["word_count"] = df["content"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,6))

plt.hist(df[df["label"]==0]["word_count"],
         bins=50,
         alpha=0.6,
         label="Fake")

plt.hist(df[df["label"]==1]["word_count"],
         bins=50,
         alpha=0.6,
         label="Real")

plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.legend()

plt.savefig(
    "images/eda/word_count_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# AVERAGE WORD COUNT
# ==========================================================

avg_words = df.groupby("label")["word_count"].mean()

plt.figure(figsize=(6,5))

plt.bar(
    ["Fake","Real"],
    avg_words
)

plt.title("Average Word Count")

plt.ylabel("Words")

plt.savefig(
    "images/eda/average_word_count.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# AVERAGE CHARACTER COUNT
# ==========================================================

avg_characters = df.groupby("label")["character_count"].mean()

plt.figure(figsize=(6,5))

plt.bar(
    ["Fake","Real"],
    avg_characters
)

plt.title("Average Character Count")

plt.ylabel("Characters")

plt.savefig(
    "images/eda/average_character_count.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# WORD COUNT BOXPLOT
# ==========================================================

plt.figure(figsize=(8,6))

df.boxplot(
    column="word_count",
    by="label"
)

plt.title("Word Count Boxplot")

plt.suptitle("")

plt.xlabel("Label (0 = Fake, 1 = Real)")

plt.ylabel("Word Count")

plt.savefig(
    "images/eda/word_count_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# CHARACTER COUNT BOXPLOT
# ==========================================================

plt.figure(figsize=(8,6))

df.boxplot(
    column="character_count",
    by="label"
)

plt.title("Character Count Boxplot")

plt.suptitle("")

plt.xlabel("Label (0 = Fake, 1 = Real)")

plt.ylabel("Character Count")

plt.savefig(
    "images/eda/character_count_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

import seaborn as sns

plt.figure(figsize=(6,5))

sns.heatmap(
    df[["word_count","character_count","label"]].corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "images/eda/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# DATASET STATISTICS
# ==========================================================

print("\n==============================")
print("DATASET STATISTICS")
print("==============================")

print(f"Total Articles : {len(df)}")
print(f"Fake Articles  : {(df['label']==0).sum()}")
print(f"Real Articles  : {(df['label']==1).sum()}")

print(f"\nAverage Words       : {df['word_count'].mean():.2f}")
print(f"Average Characters  : {df['character_count'].mean():.2f}")

print(f"\nMaximum Words : {df['word_count'].max()}")
print(f"Minimum Words : {df['word_count'].min()}")

print(f"\nMaximum Characters : {df['character_count'].max()}")
print(f"Minimum Characters : {df['character_count'].min()}")

print("\nEDA Completed Successfully!")