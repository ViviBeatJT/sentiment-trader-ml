import os
from dotenv import load_dotenv
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
from datetime import datetime
import pandas as pd

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize News Client
news_client = NewsClient(API_KEY, SECRET_KEY)


def fetch_news_data(symbol, start_date, end_date):
    print(f"--- Fetching ALL news for {symbol} ---")
    all_articles = []  # Store the actual news dicts here
    next_page_token = None
    data = []
    
    while True:
        request_params = NewsRequest(
            symbols=symbol,
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d'),
            limit=50,
            page_token=next_page_token
        )

        response = news_client.get_news(request_params)

        # Based on your print(all_news[0]):
        # The response is a NewsSet which, when converted to a dict/tuple,
        # has the data in the second element [1] under the 'news' key.

        # This line extracts the list of 50 articles from the current page
        current_page_articles = response.data['news']
        all_articles.extend(current_page_articles)

        next_page_token = response.next_page_token
        print(f"Fetched {len(all_articles)} articles so far...")

        if not next_page_token or len(all_articles) >= 2000:
            break

        # Now 'n' is a standard dictionary from your print output
        # for n in all_articles:
        #     if isinstance(n, dict):
        #         item = {
        #             'created_at': n.get('created_at'),
        #             'headline': n.get('headline'),
        #             'summary': n.get('summary')
        #         }
        #     else:
        #         item = {
        #             'created_at': n.created_at,
        #             'headline': n.headline,
        #             'summary': n.summary
        #         }
        #     data.append(item)
    for n in all_articles:
        if isinstance(n, dict):
            item = {
                'created_at': n.get('created_at'),
                'headline': n.get('headline'),
                'summary': n.get('summary')
            }
        else:
            item = {
                'created_at': n.created_at,
                'headline': n.headline,
                'summary': n.summary
            }
        data.append(item)

    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')

    df.to_csv(f"data/{symbol}_news.csv", index=False)
    print(f"Saved {len(df)} articles to data/{symbol}_news.csv")
    return df


if __name__ == "__main__":
    fetch_news_data("AAPL", "2024-01-01", "2026-04-01")
