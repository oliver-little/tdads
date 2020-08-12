import tensorflow
import pandas as pd
import numpy as np
from one_hot_encoder import Encoder
from logistic_model import fit, predict, evaluate
import time
import random
import csv
import os

#read the csv
tweets = pd.read_csv("preprocessed_tweets.csv")
tweets.drop_duplicates(subset='text', inplace=True)

#randomise it
tweets = tweets.sample(frac=1)

encoder = Encoder(tweets)
#find the max length tweet
encoder.find_max_tweet()
print("Max length found. Length: ", encoder.max_tweet_length)

#create a word dictionary
encoder.create_word_dictionary()
print("Word dictionary created.")

#only create a vector file if its not already there
if not os.path.isfile("./vectors_file.txt"):
    #convert all the tweets into vectors
    encoder.generate_vectors()

#convert the sentiment labels into numbers
labels = []
for i in tweets.airline_sentiment:
    if i == "positive":
        labels.append(0)
    elif i == "negative":
        labels.append(1)
    else:
        labels.append(2)

#read the vectors file
with open("vectors_file.txt", 'r') as file:
    vects = file.readlines()

#convert the tweet_vectors into a usable format
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

#Fit the model and get the time taken
temp_time = time.time()
reg = fit(tweet_vectors, airline_sentiment_labels, 12960, 3)
fit_time = time.time() - temp_time


#Get the models weights and save them to a csv
weights = reg[1]
file = open("results.csv", 'w', newline='')
writer = csv.writer(file)
writer.writerow(["Word","Sentiment","Sentiment Confidence"])
for i, we in enumerate(weights[0]):
    if  we.argmax() == 0:
        sentiment = "positive"
    elif we.argmax() == 1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    writer.writerow([encoder.word_dictionary[i], sentiment, max(we)])
file.close()

#use the model for predictions and time it
temp_time = time.time()
test_data = random.randint(0, 13000)
predictions = predict(tweet_vectors[test_data:test_data+1000], reg[0])
predictions_time = time.time() - temp_time

evaluation = evaluate(predictions, airline_sentiment_labels[test_data:test_data+1000])
print("\n===================\nAccuracy:", evaluation)
print("===============================")
print("fit time: ", fit_time)
print("prediction time: ", predictions_time)
