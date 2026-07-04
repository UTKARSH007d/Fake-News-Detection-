import streamlit as st
import joblib
import re
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Detector")

st.markdown("""
Paste any news article below and the trained **Random Forest**
model will classify it as **Fake** or **Real**.
""")

st.divider()

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "trained_models" / "random_forest.pkl"

VECTORIZER_PATH = BASE_DIR / "models" / "vectorizers" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(VECTORIZER_PATH)

# ----------------------------------------------------
# NLTK
# ----------------------------------------------------

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

# ----------------------------------------------------
# CLEANING FUNCTION
# ----------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ----------------------------------------------------
# INPUT
# ----------------------------------------------------

article = st.text_area(
    "Enter News Article",
    height=300,
    placeholder="Paste news article here..."
)
# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------

if st.button("🔍 Detect News", use_container_width=True):

    if article.strip() == "":

        st.warning("Please enter a news article.")

    else:

        cleaned_text = clean_text(article)

        vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        # IMPORTANT
        # Label Mapping:
        # 0 = Real
        # 1 = Fake

        real_probability = probabilities[0] * 100
        fake_probability = probabilities[1] * 100

        confidence = max(real_probability, fake_probability)

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(f"""
# 🚨 FAKE NEWS DETECTED

### Confidence : **{confidence:.2f}%**
""")

        else:

            st.success(f"""
# ✅ REAL NEWS

### Confidence : **{confidence:.2f}%**
""")

        st.divider()

        st.subheader("Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "✅ Real News",
                f"{real_probability:.2f}%"
            )

        with col2:

            st.metric(
                "🚨 Fake News",
                f"{fake_probability:.2f}%"
            )

        st.progress(confidence / 100)

        st.divider()

        with st.expander("🧹 View Preprocessed Text"):

            st.write(cleaned_text)

        st.divider()

        st.info(f"""
### Model Information

**Model Used:** Random Forest

**Vectorizer:** TF-IDF

**Training Accuracy:** 95.77%

**Dataset:** WELFake Dataset
""")

st.divider()

st.markdown("""
### 💡 Tips

- Paste complete news articles for best results.
- Very short headlines may reduce prediction accuracy.
- The dashboard uses the trained Random Forest model for fast real-time inference.
- Transformer models (DistilBERT, BERT, RoBERTa) achieved higher benchmark accuracy and are shown on the **Models** page.
""")

st.caption("Developed by Utkarsh Gupta | AI/ML Internship Project")