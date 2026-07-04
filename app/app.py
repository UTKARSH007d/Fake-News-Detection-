import streamlit as st

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="📰",
    layout="wide"
)

# ---------------- Title ----------------
st.title("📰 TruthLens AI")

st.markdown("""
### AI-Powered Fake News Detection Dashboard

Welcome to **TruthLens AI**, an intelligent Fake News Detection System built using
Machine Learning and Transformer-based NLP models.

This dashboard allows you to explore the dataset, visualize insights,
compare different models, and predict whether a news article is **Fake** or **Real**.
""")

st.divider()

# ---------------- Metrics ----------------
st.subheader("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", "72,134 Articles")

with col2:
    st.metric("Models", "7")

with col3:
    st.metric("Best Model", "RoBERTa")

with col4:
    st.metric("Accuracy", "99.65%")

st.divider()

# ---------------- Features ----------------
st.subheader("🚀 Dashboard Features")

feature1, feature2 = st.columns(2)

with feature1:
    st.success("✅ Explore Dataset")
    st.success("✅ Interactive EDA")
    st.success("✅ Compare ML Models")

with feature2:
    st.success("✅ Compare Transformer Models")
    st.success("✅ Detect Fake News")
    st.success("✅ View Model Performance")

st.divider()

# ---------------- Workflow ----------------
st.subheader("🔄 Machine Learning Pipeline")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
📥 **Data Collection**

• WELFake Dataset

• 72,134 News Articles
""")

with col2:
    st.info("""
🧹 **Preprocessing**

• Cleaning

• Stopword Removal

• Lemmatization

• TF-IDF
""")

with col3:
    st.info("""
🤖 **Training & Prediction**

• ML Models

• Transformers

• Live Detection
""")

# ---------------- Technology Stack ----------------
st.subheader("🛠 Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.success("Python")
    st.success("Pandas")

with tech2:
    st.success("NumPy")
    st.success("NLTK")

with tech3:
    st.success("Scikit-Learn")
    st.success("Streamlit")

with tech4:
    st.success("BERT")
    st.success("RoBERTa")
    st.success("DistilBERT")

st.divider()

# ---------------- Best Model ----------------
st.subheader("🏆 Best Performing Model")

st.success("""
### RoBERTa

Accuracy : **99.65%**

✔ Highest Accuracy

✔ Best Generalization

✔ Fine-tuned Transformer Model
""")

st.divider()

st.subheader("🚀 Quick Navigation")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Dataset Dashboard", use_container_width=True):
        st.switch_page("pages/Dataset.py")

with col2:
    if st.button("📈 Explore EDA", use_container_width=True):
        st.switch_page("pages/EDA.py")

with col3:
    if st.button("📰 Detect Fake News", use_container_width=True):
        st.switch_page("pages/Predictor.py")

st.divider()

st.caption("Developed by Utkarsh Gupta | AI/ML Internship Project")