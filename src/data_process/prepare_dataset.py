import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    # 1. Rename columns for readability
    # column_mapping = {
    #     0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 
    #     4: 'close', 5: 'volume', 6: 'sentiment_score', 7: 'target'
    # }
    # df = df.rename(columns=column_mapping)
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. FEATURE ENGINEERING (Relative Metrics)
    # Price Momentum: Log returns normalize price changes across different levels
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility: The 'size' of the candle relative to price
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Volume Intensity: Current volume vs its 5-period moving average
    df['volume_intensity'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Sentiment Velocity: Is the news trend improving or declining?
    df['sentiment_change'] = df['sentiment_score'].diff()
    
    # 3. CLEANUP
    # Dropping NaNs created by shift() and rolling()
    df = df.dropna().copy()
    
    # 4. SCALING (The "Accuracy Fix")
    # We scale these so the model doesn't prioritize high prices over low sentiment scores
    feature_cols = [
        'returns', 'range_pct', 'volume_intensity', 
        'sentiment_score', 'sentiment_change'
    ]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df[feature_cols], df['target']

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
    
    X, y  = engineer_features(df)
    processed_df = pd.concat([X, y], axis=1)

    # Drop the last row because we don't know the "next" price for it
    # df = df.dropna()

    processed_df.to_csv("data/training_master.csv", index=False)
    print(f"Master dataset created with {len(df)} rows.")
    return df


if __name__ == "__main__":
    prepare_final_data("data/AAPL_historical.csv", "data/AAPL_news_scored.csv")
