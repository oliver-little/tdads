import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import pandas as pd
import string
import re

import emo_unicode

pd.set_option('display.max_colwidth', None)


stop_words = stopwords.words("english")

airline_usertags = {
        "@virginamerica": "Virgin America",
        "@united": "United",
        "@southwestair": "Southwest",
        "@jetblue": "Delta",
        "@usairways": "US Airways",
        "@americanair": "American",
}


def remove_airline_tags(tweets):
    """ replace airline & user tags with AIRLINE, OTHER AIRLINE, or USER

    Parameters:
        tweets (pandas.DataFrame)

    Returns:
        texts (pandas.Series)

    Usage:
        tweets.text = remove_airline_tags(tweets)
    """
    ret_texts = []
    for index, tweet in tweets.iterrows():
        ret_text = []
        for word in tweet.text.split():
            if word in airline_usertags and airline_usertags[word] == tweet.airline:
                ret_text.append("@AIRLINE")
            elif word in airline_usertags:
                ret_text.append("@OTHER_AIRLINE")
            elif word.startswith("@"):
                ret_text.append("@USER")
            else:
                ret_text.append(word)
        ret_texts.append(" ".join(ret_text))
    return pd.Series(ret_texts)

def remove_links(tweet_texts):
    """ Removes URLs in all provided tweets

    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = remove_links(tweets.text)
    """
    ret_texts = []
    for tweet in tweet_texts:
        ret_texts.append(re.sub(r"http\S+", "", tweet))
    return ret_texts
            

def translate_emoji(tweet):
    """ Translate emoji to :text: in single tweet

    Parameters:
        tweet (str)

    Returns:
        tweet (str)
    """
    ret_text = []
    for char in tweet:
        if char in emo_unicode.UNICODE_EMO:
            ret_text.append(emo_unicode.UNICODE_EMO[char])
        else:
            ret_text.append(char)
    return "".join(ret_text)


def translate_all_emoji(tweets):
    """ Translates emoji in all provided tweets.

    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = translate_all_emoji(tweets.text)
    """
    tweets.texts = tweets.texts.apply(lambda t: translate_emoji(str(t)))
    return tweets


def remove_stopwords(tweet_texts, stopwords = stop_words):
    """ Remove stopwords from tweets

    Parameters:
        tweet_texts (pandas.Series)
        stopwords (list): list of stopwords. optional; will use nltk stopwords by default.

    Returns:
        texts (pandas.Series)

    Usage:
        tweets.text = remove_stopwords(tweets.text)
    """
    ret_texts = []
    for tweet in tweet_texts:
        ret_text = []
        for word in str(tweet).split():
            if word not in stopwords:
                ret_text.append(word)
        ret_texts.append(" ".join(ret_text))
    return ret_texts


def stem_texts(tweet_texts):
    """ Stem texts

    Parameters:
        tweet_texts (pandas.Series)

    Returns:
        texts (pandas.Series)

    Usage:
        tweets.text = stem_texts(tweets.text)
    """
    ps = PorterStemmer()

    ret_texts = []
    for tweet in tweet_texts:
        ret_text = []
        for word in str(tweet).split():
            ret_text.append(ps.stem(word))
        ret_texts.append(" ".join(ret_text))
    return ret_texts


def lemmatize_texts(tweet_texts):
    """ Lemmatize texts

    Parameters:
        tweet_texts (pandas.Series)

    Returns:
        texts (pandas.Series)

    Usage:
        tweets.text = lemmatize_texts(tweets.text)
    """
    lemmatizer = WordNetLemmatizer()
    ret_texts = []
    for tweet in tweet_texts:
        nlp = spacy.load("en", disable=['parser', 'ner'])
        doc = nlp(tweet)
        ret_texts.append(" ".join([token.lemma_ for token in doc]))
    return ret_texts

def remove_punctuation(tweet_texts):
    """ Strips all punctuation in all provided tweets.

    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = remove_punctuation(tweets.text)
    """
    ret_texts = []
    table = str.maketrans("", "", string.punctuation)
    for tweet in tweet_texts:
        ret_texts.append(tweet.translate(table))
    return ret_texts

def pd_read(filename, lower = True):
    """ Read tweets from filename

    Parameters:
        filename (str)
        lower (bool): optional lowercase

    Returns:
        pandas.DataFrame()
    """
    tweets = pd.read_csv("tweets.csv")
    tweets.drop_duplicates(subset='text', inplace=True)
    if lower:
        tweets.text = tweets.text.str.lower()
    return tweets


if __name__ == '__main__':
    # nltk.download("stopwords")
    # nltk.download("wordnet")
    # nltk.download("punkt")
    texts = ["Nice RT @VirginAmerica: Vibe with the moodlight from takeoff to touchdown. #MoodlitMonday #ScienceBehindTheExperience http://t.co/Y7O0uNxTQP"]

    #tweets = pd_read("tweets.csv")
    print("opened")
    # tweets.text = stem_texts(tweets.text)
    #print(tweets.text)
    print(lemmatize_texts(texts))
    print(lemmatize_texts(remove_punctuation(remove_links(texts))))
