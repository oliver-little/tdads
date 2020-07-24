# Implementation of topic detection in tweets using Latent Dirichlet Allocation
import pandas as pd
import numpy as np
import csv
import pyLDAvis
import pyLDAvis.gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from pprint import pprint
from preproc import *

NUM_TOPICS = 8

FILTERED_WORDS = ["airline", "otherairline", "user", "-pron-"]

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

## PREPROCESSING

# Get tweets
tweets = pd_read("preprocessed_negative_tweets.csv")

# Get only negative ones (for this task)
tweets = tweets[tweets.airline_sentiment == "negative"]

tweets = remove_airline_tags(tweets)
print("airlines removed")
tweets.text = remove_links(tweets.text)
print("links removed")
tweets.text = lt_gt_conversion(ampersand_conversion(arrow_conversion(tweets.text)))
print("encoded symbols converted")
tweets.text = with_without_conversion(tweets.text)
print("with and without converted")
tweets.text = hashtag_to_words(tweets.text)
print("hashtags converted")
tweets = translate_all_emoji(tweets)
print("emoji converted")
tweets.text = remove_contractions(tweets.text)
print("contractions removed")
tweets.text = remove_punctuation(tweets.text)
print("punctuation removed")
tweets.text = lemmatize_texts(tweets.text)
print("words lemmatized")
tweets.text = remove_stopwords(tweets.text)
print("stopwords removed")
tweets.text = tweets.text.str.lower()

texts = tweets["text"].values

# Tokenize and remove short words or filtered words
tokenized_texts = []
for text in texts:
    split_text = text.split()
    split_text = [word for word in split_text if len(word) > 2 and word not in FILTERED_WORDS]
    tokenized_texts.append(split_text)

# Create a dictionary for each word, and a bag of words
text_dictionary = Dictionary(tokenized_texts)

# Remove words that appear in over 50%, or less than 0.5%, and keep the top 66% of the vocabulary
text_dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=len(text_dictionary)//2)
text_bow = [text_dictionary.doc2bow(text) for text in tokenized_texts]

# Create the LDA Model
lda = LdaModel(text_bow, num_topics=NUM_TOPICS, id2word=text_dictionary, offset=2, passes=10, alpha="auto", eta="auto")
pprint(lda.print_topics())

vis = pyLDAvis.gensim.prepare(lda, text_bow, text_dictionary)
pyLDAvis.save_html(vis, "topic_analysis.html")


    
