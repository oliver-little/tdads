import preproc
import tensorflow
import pandas as pd
import numpy as np
from one_hot_encoder import Encoder
from logistic_model import fit, predict, evaluate
import time
import random

#read the csv
tweets = pd.read_csv("preprocessed_tweets.csv")

tweets.drop_duplicates(subset='text', inplace=True)

tweets = tweets.sample(frac=1)


encoder = Encoder(tweets)
encoder.find_max_tweet()
print("Max length found. Length: ", encoder.max_tweet_length)

encoder.create_word_dictionary()
print("Word dictionary created.")
"""
#need vectors for each tweet
encoder.generate_vectors()
"""
labels = []

for i in tweets.airline_sentiment:
    if i == "positive":
        labels.append(0)
    elif i == "negative":
        labels.append(1)
    else:
        labels.append(2)

with open("vectors_file.txt", 'r') as file:
    vects = file.readlines()

tweet_vectors = []

for lines in vects:
    t = lines.strip('][').split(']')
    pp = t[0].split(',')
    final = [int(x) for x in pp]
    tweet_vectors.append(final)

#conver the vectors into numpy arrays
tweet_vectors = np.asarray(tweet_vectors).astype(np.float32)
airline_sentiment_labels = np.asarray(labels).astype(np.float32)

print("Vectors created :)")

temp_time = time.time()
reg = fit(tweet_vectors, airline_sentiment_labels, 12960, 3)
fit_time = time.time() - temp_time
index = 0
weights = reg[1]


temp_time = time.time()
test_data = random.randint(0, 13000)
predictions = predict(tweet_vectors[test_data:test_data+1000], reg[0])
predictions_time = time.time() - temp_time

evaluation = evaluate(predictions, airline_sentiment_labels[test_data:test_data+1000])

print("\n===================\nAccuracy:", evaluation[0])
#print("Positive: ", evaluation[1][0][0]/(evaluation[1][0][0]+evaluation[1][0][1]))
#print("Negative: ", evaluation[1][1][0]/(evaluation[1][1][0]+evaluation[1][1][1]))
#print("Neutral: ", evaluation[1][2][0]/(evaluation[1][2][0]+evaluation[1][2][1]))
print("===============================")
print("fit time: ", fit_time)
print("prediction time: ", predictions_time)
