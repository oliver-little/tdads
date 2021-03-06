import pandas as pd
import numpy as np
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from preproc import *

OUT_DIR = "../code_output/"

NUM_TOPICS = 9

def pd_read(filename = "tweets.csv", lower = True):
    """ Read tweets from filename

    Parameters:
        filename (str)
        lower (bool): optional lowercase

    Returns:
        pandas.DataFrame()
    """
    tweets = pd.read_csv(filename)
    tweets.drop_duplicates(subset='text', inplace=True)
    if lower:
        tweets.text = tweets.text.str.lower()
    return tweets

if __name__ == "__main__":
    tweets = pd_read("preprocessed_negative_tweets.csv")
    texts = tweets["text"].values

    vectorizer = TfidfVectorizer(stop_words="english", use_idf=True, smooth_idf=True)
    svd = TruncatedSVD(n_components=NUM_TOPICS, n_iter=15)

    pipeline = Pipeline([("tfidf", vectorizer), ("svd", svd)])
    topicPredictions = pipeline.fit_transform(texts)
    topicNums = [np.argmax(text) for text in topicPredictions]

    if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)

    with open(OUT_DIR + "topic_tweets.csv", "w", encoding="utf8", newline="") as outputFile:
        writer = csv.writer(outputFile)
        for textNum in range(len(texts)):
            writer.writerow([topicNums[textNum], texts[textNum]])

    
