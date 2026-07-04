import streamlit as st

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("ℹ️ About Project")

st.markdown("""
Learn more about the AI-Powered Fake News Detection System,
its workflow, technologies used, and the developer behind the project.
""")

st.divider()

# ---------------- PROJECT ----------------

st.subheader("📌 Project Overview")

st.info("""
**TruthLens AI** is an AI-powered Fake News Detection System developed to classify
news articles into **Fake** and **Real** categories using Natural Language Processing,
Machine Learning, and Transformer models.

The project compares multiple ML algorithms with state-of-the-art Transformer
architectures to identify the most accurate solution.
""")

st.divider()

# ---------------- DATASET ----------------

st.subheader("📂 Dataset")

col1, col2 = st.columns(2)

with col1:

    st.success("""
**Dataset Name**

WELFake Dataset

• 72,134 News Articles

• Binary Classification

• Fake and Real News
""")

with col2:

    st.success("""
**Features**

• Content

• Label

• NLP Preprocessing

• TF-IDF Features
""")

st.divider()

# ---------------- PREPROCESSING ----------------

st.subheader("🧹 NLP Preprocessing")

st.info("""
✔ Lowercasing

✔ HTML Removal

✔ URL Removal

✔ Punctuation Removal

✔ Number Removal

✔ Stopword Removal

✔ Lemmatization

✔ TF-IDF Vectorization
""")

st.divider()

# ---------------- MODELS ----------------

st.subheader("🤖 Models Used")

ml, transformer = st.columns(2)

with ml:

    st.success("""
### Machine Learning

• Logistic Regression

• Naive Bayes

• Random Forest

• Support Vector Machine
""")

with transformer:

    st.success("""
### Transformer Models

• DistilBERT

• BERT

• RoBERTa
""")

st.divider()

# ---------------- TECHNOLOGIES ----------------

st.subheader("🛠 Technologies")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.success("Python")
    st.success("Pandas")

with c2:
    st.success("NumPy")
    st.success("NLTK")

with c3:
    st.success("Scikit-Learn")
    st.success("Streamlit")

with c4:
    st.success("Transformers")
    st.success("Matplotlib")

st.divider()

# ---------------- WORKFLOW ----------------

st.subheader("🔄 Project Workflow")

st.info("""
Dataset

⬇

Preprocessing

⬇

TF-IDF Feature Extraction

⬇

Machine Learning Models

⬇

Transformer Models

⬇

Model Evaluation

⬇

Streamlit Deployment
""")

st.divider()

# ---------------- RESULTS ----------------

st.subheader("🏆 Final Results")

st.success("""
**Best Classical Model**

Random Forest

Accuracy : **95.77%**

---

**Best Transformer Model**

RoBERTa

Accuracy : **99.65%**
""")

st.divider()

# ---------------- DEVELOPER ----------------

st.subheader("👨‍💻 Developer")

st.info("""
**Utkarsh Gupta**

B.Tech Computer Science Engineering (AI & ML)

Manipal University Jaipur

AI/ML Internship Project

Fake News Detection using Machine Learning and Transformers
""")

st.divider()

st.caption("© 2026 TruthLens AI | Developed by Utkarsh Gupta")