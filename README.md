# Sentiment-Aware Quant Trader


<!-- Common commands -->
source venv/bin/activate


# TODO
1. fix the fetch_news failure, right now only fetches data from 2026-03-31
done
2. fix the process_sentiment failure, right now only generates score for 100 rows, maybe have been fixed. run the code first: python src/data_process/process_sentiment.py
done.
3. train_model.py not tested, not run yet.
done.

2026.05.08 TODO
1. improve model accuracy, current accuracy is low, maybe increase the epoch or something 
2. The prediction should base on live price and current news. the predict code haven't been run yet. 