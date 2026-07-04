import streamlit as st
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="EDA",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Exploratory Data Analysis")

st.markdown("""
Explore visual insights from the **WELFake Dataset** used to train the
Machine Learning and Transformer models.
""")

st.divider()

# ---------------- PATH ----------------

BASE_DIR = Path(__file__).resolve().parents[2]
EDA_DIR = BASE_DIR / "images" / "eda"

# ---------------- IMAGE PATHS ----------------

class_distribution = EDA_DIR / "class_distribution.png"

fake_words = EDA_DIR / "top20_fake_words.png"
real_words = EDA_DIR / "top20_real_words.png"

fake_cloud = EDA_DIR / "fake_wordcloud.png"
real_cloud = EDA_DIR / "real_wordcloud.png"

word_distribution = EDA_DIR / "word_count_distribution.png"
character_distribution = EDA_DIR / "character_count_distribution.png"

average_word = EDA_DIR / "average_word_count.png"
average_character = EDA_DIR / "average_character_count.png"

word_box = EDA_DIR / "word_count_boxplot.png"
character_box = EDA_DIR / "character_count_boxplot.png"

correlation = EDA_DIR / "correlation_heatmap.png"

# ---------------- TABS ----------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Class Distribution",
        "📝 Top Words",
        "☁️ Word Clouds",
        "📈 Text Statistics",
        "🔥 Correlation"
    ]
)

# ===========================================================
# TAB 1
# ===========================================================

with tab1:

    st.subheader("Class Distribution")

    if class_distribution.exists():

        st.image(
            str(class_distribution),
            use_container_width=True
        )

    else:

        st.warning("class_distribution.png not found")

    st.info("""
The dataset is nearly balanced between Fake and Real news,
helping reduce bias during model training.
""")

# ===========================================================
# TAB 2
# ===========================================================

with tab2:

    st.subheader("Top 20 Words")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📰 Fake News")

        if fake_words.exists():

            st.image(
                str(fake_words),
                use_container_width=True
            )

        else:

            st.warning("top20_fake_words.png not found")

    with col2:

        st.markdown("### ✅ Real News")

        if real_words.exists():

            st.image(
                str(real_words),
                use_container_width=True
            )

        else:

            st.warning("top20_real_words.png not found")

    st.success("""
These charts display the 20 most frequent words appearing
after preprocessing.
""")
    # ===========================================================
# TAB 3
# ===========================================================

with tab3:

    st.subheader("Word Clouds")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📰 Fake News")

        if fake_cloud.exists():

            st.image(
                str(fake_cloud),
                use_container_width=True
            )

        else:

            st.warning("fake_wordcloud.png not found")

    with col2:

        st.markdown("### ✅ Real News")

        if real_cloud.exists():

            st.image(
                str(real_cloud),
                use_container_width=True
            )

        else:

            st.warning("real_wordcloud.png not found")

    st.success("""
Word clouds highlight the most dominant words after
text preprocessing.
""")

# ===========================================================
# TAB 4
# ===========================================================

with tab4:

    st.subheader("Text Statistics")

    st.markdown("### Distribution of Word Count")

    if word_distribution.exists():

        st.image(
            str(word_distribution),
            use_container_width=True
        )

    else:

        st.warning("word_count_distribution.png not found")

    st.markdown("---")

    st.markdown("### Distribution of Character Count")

    if character_distribution.exists():

        st.image(
            str(character_distribution),
            use_container_width=True
        )

    else:

        st.warning("character_count_distribution.png not found")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Average Word Count")

        if average_word.exists():

            st.image(
                str(average_word),
                use_container_width=True
            )

        else:

            st.warning("average_word_count.png not found")

    with col2:

        st.markdown("### Average Character Count")

        if average_character.exists():

            st.image(
                str(average_character),
                use_container_width=True
            )

        else:

            st.warning("average_character_count.png not found")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Word Count Boxplot")

        if word_box.exists():

            st.image(
                str(word_box),
                use_container_width=True
            )

        else:

            st.warning("average_word_count_boxplot.png not found")

    with col2:

        st.markdown("### Character Count Boxplot")

        if character_box.exists():

            st.image(
                str(character_box),
                use_container_width=True
            )

        else:

            st.warning("character_count_boxplot.png not found")

# ===========================================================
# TAB 5
# ===========================================================

with tab5:

    st.subheader("Correlation Heatmap")

    if correlation.exists():

        st.image(
            str(correlation),
            use_container_width=True
        )

    else:

        st.warning("correlation_heatmap.png not found")

    st.success("""
The correlation heatmap illustrates relationships between
different numerical features extracted from the dataset.
""")

st.divider()

st.info("📌 All visualizations shown above were generated during the Exploratory Data Analysis (EDA) phase and are displayed here for interactive exploration.")