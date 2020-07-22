import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from copy import deepcopy
import spacy
import pandas as pd
import string
import re

import emo_unicode

pd.set_option('display.max_colwidth', None)

lemmatizer = WordNetLemmatizer()

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
    tweets = deepcopy(tweets)
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
        tweets.loc[index, "text"] = " ".join(ret_text)
    return tweets

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
        ret_texts.append(re.sub(r"http\S+", "", str(tweet)))
    return ret_texts

def hashtag_to_words(tweet_texts):
    """ Converts hashtags with TitleCase to normal words (e.g: #ThisIsAHashtag to This is a Hashtag)
    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = remove_links(tweets.text)
    """
    ret_texts = []
    for tweet in tweet_texts:
        ret_text = []
        for word in str(tweet).split():
            if word.startswith("#"):
                word = word[1:]
                word = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()
                ret_text.extend(word)
            else:
                ret_text.append(word)                
        ret_texts.append(" ".join(ret_text))
    return ret_texts

def with_without_conversion(tweet_texts):
    """ Converts "w"/ to "with" and "w/out" to "without"
    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = with_without_conversion(tweets.text)
    """
    ret_texts = []
    for tweet in tweet_texts:
        tweet.replace("w/out", "without")
        tweet.replace("w/", "with")
        ret_texts.append(tweet)
    return ret_texts

def arrow_conversion(tweet_texts):
    """ Converts &gt; (>) or -&gt; (->) into "to"
    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = arrow_conversion(tweets.text)
    """
    ret_texts = []
    for tweet in tweet_texts:
        tweet.replace("-&gt;", " to ")
        tweet.replace("&gt;", " to ")
        ret_texts.append(tweet)
    return ret_texts

def ampersand_conversion(tweet_texts):
    """ Converts & (represented as (&amp;) into "and"
    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = ampersand_conversion(tweets.text)
    """
    ret_texts = []
    for tweet in tweet_texts:
        tweet.replace("&amp;", " and ")
        ret_texts.append(tweet)
    return ret_texts

def lt_gt_conversion(tweet_texts):
    """ Converts < (represented as &lt;) and > (represented as &gt;) into "less than" and "greater than"
    Parameters:
        texts (pandas.Series)

    Returns
        texts (pandas.Series)

    Usage:
        tweets.text = lt_gt_conversion(tweets.text)
    """
    ret_texts = []
    for tweet in tweet_texts:
        tweet.replace("&lt;", " less than ")
        tweet.replace("&gt;", " greater than ")
        ret_texts.append(tweet)
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
    tweets.text = tweets.text.apply(lambda t: translate_emoji(str(t)))
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
    ret_texts = []
    nlp = spacy.load("en", disable=['parser', 'ner'])
    for tweet in nlp.pipe(tweet_texts):
        ret_texts.append(" ".join([token.lemma_ for token in tweet]))
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

    tweets = pd_read("tweets.csv")
    print("opened")
    # tweets.text = stem_texts(tweets.text)
    tweets.text = lemmatize_texts(tweets.text)
    print(tweets.text)
