import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from preproc import *

def preprocess(tweets):
    tweets = remove_airline_tags(tweets)
    tweets.text = remove_links(tweets.text)
    tweets.text = lt_gt_conversion(ampersand_conversion(arrow_conversion(tweets.text)))
    tweets.text = with_without_conversion(tweets.text)
    tweets.text = hashtag_to_words(tweets.text)
    tweets = translate_all_emoji(tweets)
    tweets.text = remove_contractions(tweets.text)
    tweets.text = remove_punctuation(tweets.text)
    tweets.text = lemmatize_texts(tweets.text)
    X = tweets["text"].values
    Y = tweets["airline_sentiment"].astype("category").cat
    return X, Y.codes, Y.categories

def fit(x_train, x_test):
    text_clf = Pipeline([
        ("tfidf", TfidfVectorizer(use_idf=True, smooth_idf=True)),
        ("clf", BernoulliNB())])
    
    text_clf.fit(x_train, y_train)
    return [text_clf]

def predict(y_train, fit_return_list):
    text_clf = fit_return_list[0]
    predicted = text_clf.predict(x_test)
    return predicted

if __name__ == "__main__":
    X, Y, Y_categories = preprocess(pd_read("tweets.csv"))
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    fit_return_list = fit(x_train, x_test)
    predicted = predict(y_train, fit_return_list)
    
    print("Accuracy: " + str(np.mean(predicted == y_test)))







