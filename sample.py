import pandas as pd

df = pd.read_csv("data/processed/preprocessed_news.csv")

df.sample(5000, random_state=42).to_csv(
    "data/processed/preprocessed_news_sample.csv",
    index=False
)
