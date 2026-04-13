import pandas as pd


def prepare_final_data(price_path, news_path):
    # Load data
    prices = pd.read_csv(price_path)
    news = pd.read_csv(news_path)

    # Convert timestamps to datetime objects
    prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    news['created_at'] = pd.to_datetime(news['created_at'])

    # Round news time to the nearest hour to match price bars
    news['timestamp'] = news['created_at'].dt.floor('h')

    # Calculate average sentiment per hour (in case there are multiple headlines)
    hourly_sentiment = news.groupby(
        'timestamp')['sentiment_score'].mean().reset_index()

    # Merge prices and sentiment
    # 'left' join ensures we keep all price rows even if there was no news
    df = pd.merge(prices, hourly_sentiment, on='timestamp', how='left')

    # Fill hours with no news with 0 (neutral sentiment)
    df['sentiment_score'] = df['sentiment_score'].fillna(0)

    # Create our Target: Did the price go UP in the next hour?
    # Shift(-1) looks at the NEXT row's close price
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop the last row because we don't know the "next" price for it
    df = df.dropna()

    df.to_csv("data/training_master.csv", index=False)
    print(f"Master dataset created with {len(df)} rows.")
    return df


if __name__ == "__main__":
    prepare_final_data("data/AAPL_historical.csv", "data/AAPL_news_scored.csv")
