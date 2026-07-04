import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Comparison")

st.markdown("""
Compare the performance of Classical Machine Learning models and
Transformer models trained on the WELFake Dataset.
""")

st.divider()

BASE_DIR = Path(__file__).resolve().parents[2]

CM_DIR = BASE_DIR / "images" / "confusion_matrices"

# -------------------------------
# ACCURACY TABLE
# -------------------------------

st.subheader("📊 Model Performance")

data = {
    "Model":[
        "Naive Bayes",
        "Logistic Regression",
        "Random Forest",
        "Support Vector Machine",
        "DistilBERT",
        "BERT",
        "RoBERTa"
    ],

    "Category":[
        "Machine Learning",
        "Machine Learning",
        "Machine Learning",
        "Machine Learning",
        "Transformer",
        "Transformer",
        "Transformer"
    ],

    "Accuracy (%)":[
        85.39,
        94.87,
        95.77,
        95.41,
        99.08,
        99.12,
        99.65
    ]
}

df=pd.DataFrame(data)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("🏆 Best Model")

st.success("""
### RoBERTa

Accuracy : **99.65%**

✅ Highest Accuracy

✅ Best Generalization

✅ Transformer Architecture

✅ Selected for Final Evaluation
""")

st.divider()

tab1,tab2=st.tabs([
    "Machine Learning Models",
    "Transformer Models"
])

# ====================================================

with tab1:

    st.subheader("Classical Machine Learning")

    col1,col2=st.columns(2)

    logistic=CM_DIR/"logistic_confusion_matrix.png"
    naive=CM_DIR/"naive_bayes_confusion_matrix.png"

    randomforest=CM_DIR/"random_forest_confusion_matrix.png"
    svm=CM_DIR/"svm_confusion_matrix.png"

    with col1:

        st.markdown("### Logistic Regression")

        if logistic.exists():
            st.image(logistic,use_container_width=True)

        else:
            st.warning("logistic_confusion_matrix.png not found")

        st.markdown("### Random Forest")

        if randomforest.exists():
            st.image(randomforest,use_container_width=True)

        else:
            st.warning("random_forest_confusion_matrix.png not found")
        with col2:

         st.markdown("### Naive Bayes")

        if naive.exists():
            st.image(naive, use_container_width=True)

        else:
            st.warning("naive_bayes_confusion_matrix.png not found")

        st.markdown("### Support Vector Machine")

        if svm.exists():
            st.image(svm, use_container_width=True)

        else:
            st.warning("svm_confusion_matrix.png not found")

# ====================================================

with tab2:

    st.subheader("Transformer Models")

    st.info("""
Transformer models (DistilBERT, BERT and RoBERTa) were fine-tuned using the
HuggingFace Transformers library.

Their performance significantly surpassed the traditional Machine Learning
models due to contextual language understanding.
""")

    result = pd.DataFrame(
        {
            "Model": [
                "DistilBERT",
                "BERT",
                "RoBERTa"
            ],

            "Accuracy (%)": [
                99.08,
                99.12,
                99.65
            ]
        }
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    st.success("""
### 🏆 Final Result

| Model | Accuracy |
|--------|----------|
| DistilBERT | **99.08%** |
| BERT | **99.12%** |
| **RoBERTa** | **99.65%** ✅ |

RoBERTa achieved the highest accuracy among all trained models and was selected
as the best-performing model for Fake News Detection.
""")

st.divider()

st.subheader("📌 Key Observations")

st.success("""
✅ Transformer models significantly outperform traditional Machine Learning models.

✅ Random Forest achieved the highest accuracy among classical ML algorithms.

✅ RoBERTa achieved the best overall performance with **99.65% Accuracy**.

✅ Fine-tuning pretrained language models greatly improves Fake News Detection performance.

✅ Confusion matrices indicate excellent classification capability with minimal false predictions.
""")

st.divider()

st.caption("TruthLens AI • AI Powered Fake News Detection Dashboard")      