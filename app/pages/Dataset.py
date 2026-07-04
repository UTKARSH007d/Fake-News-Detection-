import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Dataset Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Dashboard")
st.markdown("Explore the dataset used to train the Fake News Detection models.")

st.divider()

# ---------------- LOAD DATA ----------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "preprocessed_news_sample.csv"

df = pd.read_csv(DATA_PATH)

# ---------------- DATASET STATISTICS ----------------

st.subheader("📈 Dataset Statistics")

total_articles = len(df)
fake_articles = (df["label"] == 0).sum()
real_articles = (df["label"] == 1).sum()
missing_values = df.isnull().sum().sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Articles", f"{total_articles:,}")
c2.metric("Fake News", f"{fake_articles:,}")
c3.metric("Real News", f"{real_articles:,}")
c4.metric("Missing Values", missing_values)

st.divider()

# ---------------- DATASET INFORMATION ----------------

st.subheader("📋 Dataset Information")

left, right = st.columns(2)

with left:
    st.info(f"""
**Dataset Name**

WELFake Dataset

**Rows**

{df.shape[0]:,}

**Columns**

{df.shape[1]}
""")

with right:
    st.info(f"""
**Features**

• content

• label

**Classes**

Fake (0)

Real (1)
""")

st.divider()

# ---------------- CLASS DISTRIBUTION ----------------

st.subheader("📊 Class Distribution")

img_path = BASE_DIR / "images" / "eda" / "class_distribution.png"

left, right = st.columns([2,1])

with left:
    st.image(str(img_path), use_container_width=True)

with right:

    st.success(f"""
### Dataset Balance

📰 Fake News : **{fake_articles:,}**

✅ Real News : **{real_articles:,}**

The dataset is nearly balanced which helps
prevent model bias during training.
""")

st.divider()

# ---------------- DATA PREVIEW ----------------

st.subheader("👀 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
    height=350
)

st.divider()

# ---------------- DATA QUALITY ----------------

st.subheader("✅ Data Quality")

duplicate_rows = df.duplicated().sum()

q1, q2, q3 = st.columns(3)

q1.metric("Missing Values", missing_values)
q2.metric("Duplicate Rows", duplicate_rows)
q3.metric("Unique Labels", df["label"].nunique())

st.divider()

# ---------------- PREPROCESSING ----------------

st.subheader("⚙️ Preprocessing Pipeline")

cols = st.columns(7)

steps = [
    "Lowercase",
    "HTML Removal",
    "URL Removal",
    "Punctuation",
    "Stopwords",
    "Lemmatization",
    "TF-IDF"
]

icons = ["🔡","🧹","🌐","✂️","🚫","📖","🔢"]

for col, icon, step in zip(cols, icons, steps):
    with col:
        st.info(f"{icon}\n\n{step}")

st.divider()

# ---------------- KEY INSIGHTS ----------------

st.subheader("💡 Key Insights")

st.success("""
✔ Large dataset containing over **72,000** news articles.

✔ Nearly balanced Fake and Real news classes.

✔ Text cleaned using NLP preprocessing techniques.

✔ TF-IDF Vectorization with 5000 features.

✔ Used for both Classical Machine Learning and Transformer models.
""")