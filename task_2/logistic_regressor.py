import preproc
import tensorflow
import pandas as pd
import numpy as np
from one_hot_encoder import Encoder
from logistic_model import fit, predict, evaluate
import random

#read the csv
tweets = pd.read_csv("preprocessed_tweets.csv")
tweets.drop_duplicates(subset='text', inplace=True)

#create a word bank from it
encoder = Encoder(tweets)
encoder.find_max_tweet()
print("Max length found. Length: ", encoder.max_tweet_length)
encoder.create_word_dictionary()
print("Word dictionary created.")
tweet_vectors = []

for x in tweets.text:
    vector = encoder.convert_tweet(x)
    tweet_vectors.append(vector)



labels = []

for i in tweets.airline_sentiment:
    if i == "positive":
        labels.append(0)
    if i == "negative":
        labels.append(1)
    else:
        labels.append(2)


tweet_vectors = np.asarray(tweet_vectors).astype(np.float32)

airline_sentiment_labels = np.asarray(labels).astype(np.float32)[:14216]

print("Vectors created :)")

test_data = random.randint(0,11000)
reg = fit(tweet_vectors, airline_sentiment_labels, encoder.max_tweet_length, 3)
predictions = predict(tweet_vectors[test_data:test_data+2000], reg)

evaluation = evaluate(predictions, airline_sentiment_labels[test_data:test_data+2000])
print(evaluation)
