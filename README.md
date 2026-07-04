# 📰 TruthLens AI – Fake News Detection System

An AI-powered Fake News Detection System built using **Natural Language Processing (NLP)** and **Machine Learning**. The application classifies news articles as **Fake** or **Real** and provides an interactive dashboard for dataset exploration, visualizations, model comparison, and live prediction.

---

## 🚀 Live Demo

🌐 Streamlit App: https://fakenews00.streamlit.app/

---

## 📂 GitHub Repository

🔗 https://github.com/UTKARSH007d/Fake-News-Detection-

---

## 📖 Project Overview

TruthLens AI is designed to identify fake news articles using Machine Learning and NLP techniques.

The application allows users to:

- 📊 Explore the dataset
- 📈 View Exploratory Data Analysis (EDA)
- 🤖 Compare Machine Learning models
- 🧠 Compare Transformer models
- 📰 Predict whether a news article is Fake or Real
- ℹ️ Learn about the project

---

# 📊 Dataset

**Dataset:** WELFake Dataset

- Total Articles: 72,124
- Features:
  - Content
  - Label
- Classes:
  - Fake News
  - Real News

---

# ⚙️ Preprocessing

The text preprocessing pipeline includes:

- Lowercase Conversion
- HTML Removal
- URL Removal
- Number Removal
- Punctuation Removal
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization

---

# 🤖 Machine Learning Models

The project compares multiple Machine Learning algorithms:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Tuned Random Forest

---

# 🧠 Transformer Models

The following transformer models were also evaluated:

- BERT
- DistilBERT
- RoBERTa

RoBERTa achieved the best benchmark performance, while Random Forest was selected for deployment due to its faster inference and lower computational requirements.

---

# 📊 Dashboard Pages

### 🏠 Home

Project overview, workflow, technology stack, and navigation.

### 📂 Dataset

Dataset statistics and information.

### 📈 EDA

- Class Distribution
- Word Count Analysis
- Character Count Analysis
- Word Clouds
- Top Fake Words
- Top Real Words
- Correlation Heatmap
- Boxplots

### 🤖 Models

Performance comparison of Machine Learning and Transformer models.

### 📰 Predictor

Predict whether a news article is Fake or Real.

### ℹ️ About

Project details, workflow, technologies, and developer information.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Transformers
- PyTorch
- Streamlit
- Git
- GitHub

---

# 📁 Project Structure

```
Fake-News-Detection-
│
├── app/
│   ├── app.py
│   ├── pages/
│   └── utils/
│
├── images/
├── model_results/
├── models/
├── src/
├── requirements.txt
└── README.md
```

---

# ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/UTKARSH007d/Fake-News-Detection-.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

# 🎯 Future Improvements

- Explainable AI (XAI)
- Real-time News Verification
- Multilingual Fake News Detection
- Social Media Integration
- REST API Development
- Mobile Application

---

# 👨‍💻 Developer

**Utkarsh Gupta**

B.Tech Artificial Intelligence & Machine Learning

Manipal University Jaipur

---

## ⭐ If you found this project useful, consider giving it a star!