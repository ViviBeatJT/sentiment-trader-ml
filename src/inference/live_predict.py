import asyncio
from alpaca.data.live import StockDataStream, NewsDataStream
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Setup FinBERT (Inference Mode)
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.eval() # Set to evaluation mode

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Returns 0: Neutral, 1: Positive, 2: Negative (depends on model mapping)
    return torch.argmax(prediction).item()

# 2. Define the Real-Time Handlers
async def news_handler(data):
    sentiment = get_sentiment(data.headline)
    sentiment_map = {0: "Neutral", 1: "Positive", 2: "Negative"}
    print(f"NEW HEADLINE: {data.headline}")
    print(f"SENTIMENT SCORE: {sentiment_map[sentiment]}")
    
    # Store this for the price handler to use
    global current_sentiment
    current_sentiment = sentiment

async def trade_handler(data):
    # This fires on every single price tick (very fast!)
    print(f"LIVE PRICE {data.symbol}: ${data.price}")
    
    # Logic: If news is positive AND price starts moving up...
    # (Simplified example)
    if 'current_sentiment' in globals() and current_sentiment == 1:
        print(">>> SIGNAL DETECTED: Bullish Sentiment + Price Activity. Executing...")
        # Your trade execution code here

# 3. Start the Streams
async def main():
    # Replace with your keys
    api_key = "YOUR_API_KEY"
    secret_key = "YOUR_SECRET_KEY"

    # Initialize Streams
    news_stream = NewsDataStream(api_key, secret_key)
    price_stream = StockDataStream(api_key, secret_key)

    # Subscribe
    news_stream.subscribe_news(news_handler, "AAPL")
    price_stream.subscribe_trades(trade_handler, "AAPL")

    # Run both concurrently
    await asyncio.gather(
        news_stream._run_forever(),
        price_stream._run_forever()
    )

if __name__ == "__main__":
    asyncio.run(main())