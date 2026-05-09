# fetch historical price data
python src/data_process/fetch_data.py
# fetch historical news
python src/data_process/fetch_news.py
# preprocess news sentiment using LLM
python src/data_process/process_sentiment.py
# combine the price data and news data
python src/data_process/prepare_dataset.py
# train model
python src/train_model.py
# inference
python src/inference/predict.py

