import pandas as pd
from transformers import pipeline

# 1. Load the LLM Pipeline
# This might take a minute the first time as it downloads the model weights
print("Loading FinBERT LLM...")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def get_sentiment_scores(csv_path):
    df = pd.read_csv(csv_path)
    texts = (df['headline'] + " " + df['summary'].fillna("")).tolist()

    print(f"Analyzing {len(texts)} articles. This may take a while...")

    # Process in batches of 10 to avoid crashing your memory
    batch_size = 10
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        results = sentiment_analyzer(batch)

        for res in results:
            score = res['score'] if res['label'] == 'positive' else (
                -res['score'] if res['label'] == 'negative' else 0)
            all_scores.append(score)

        if i % 100 == 0:
            print(f"Progress: {i}/{len(texts)}...")

    df['sentiment_score'] = all_scores
    df.to_csv(csv_path.replace(".csv", "_scored.csv"), index=False)


if __name__ == "__main__":
    get_sentiment_scores("data/AAPL_news.csv")
