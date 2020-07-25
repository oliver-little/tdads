# Implementation of topic detection in tweets using Latent Dirichlet Allocation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from preproc import *

# Grid search parameters
TOPIC_RANGE = range(3, 11)
ALPHA_RANGE = np.arange(0.01, 1, 0.25)
BETA_RANGE = np.arange(0.01, 1, 0.25)

# Hyperparameters (from grid search)
NUM_TOPICS = 8
ALPHA = 0.5
BETA = 0.5

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

def preprocess(tweets):
    # Get only negative ones (for this task)
    tweets = tweets[tweets.airline_sentiment == "negative"]

    tweets = remove_airline_tags(tweets)
    tweets.text = remove_links(tweets.text)
    tweets.text = lt_gt_conversion(ampersand_conversion(arrow_conversion(tweets.text)))
    tweets.text = with_without_conversion(tweets.text)
    tweets.text = hashtag_to_words(tweets.text)
    tweets = translate_all_emoji(tweets)
    tweets.text = remove_contractions(tweets.text)
    tweets.text = remove_punctuation(tweets.text)
    tweets.text = lemmatize_texts(tweets.text)
    tweets.text = remove_stopwords(tweets.text)
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
    text_corpus = [text_dictionary.doc2bow(text) for text in tokenized_texts]
    return (text_dictionary, text_corpus)

# Assumes that unprocessed data is provided
# Data is a pandas dataframe in the form of the tweets.csv file
def fit(tweets_train):
    dictionary, corpus = preprocess(tweets_train)

    # Replace with gensim.models.ldamodel.LdaModel if this causes issues
    lda = LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10, alpha=ALPHA, eta=BETA)
    return [lda]

def predict(tweets_test, fit_return_list):
    lda = fit_return_list[0]
    dictionary, corpus = preprocess(tweets_test)
    predictions = []
    # LdaModel prediction returns an array of tuples in the form (TOPIC_NUM, PROBABILITY)
    for probabilities in lda[corpus]:
        predictions.append(max(probabilities, key=lambda x: x[1])[0])
    return predictions

def getModelCoherence(dictionary, corpus, topic, a, b):

    # Remaking the provided texts fixes an error
    reconstructedTexts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
    
    # Replace with gensim.models.ldamodel.LdaModel if this causes issues
    lda = LdaMulticore(corpus, num_topics=topic, id2word=dictionary, passes=10, alpha=a, eta=b)
    coherence_model = CoherenceModel(model=lda, texts=reconstructedTexts, dictionary=dictionary, coherence="c_v")
    coherence = coherence_model.get_coherence()
    return coherence
        
def gridSearch(tweets, verbose=0):
    dictionary, corpus = preprocess(tweets)

    results = {"Topics" : [], "Alpha" : [], "Beta" : [], "Coherence" : []}

    NUM_PARAMETERS = str(len(TOPIC_RANGE) * len(ALPHA_RANGE) * len(BETA_RANGE))
    if verbose == 1:
        print("Total number of parameters to test: " + NUM_PARAMETERS)

    count = 1
    for topic in TOPIC_RANGE:
        for alpha in ALPHA_RANGE:
            for beta in BETA_RANGE:
                if verbose == 1:
                    print(str(count) + "/" + NUM_PARAMETERS + ": alpha=" + str(alpha) + " beta=" + str(beta) + " num_topics=" + str(topic))
                coherence = getModelCoherence(dictionary, corpus, topic, alpha, beta)
                results["Topics"].append(topic)
                results["Alpha"].append(alpha)
                results["Beta"].append(beta)
                results["Coherence"].append(coherence)
                count += 1
    if verbose == 1:
        print("Finished grid search. Saving results to lda_gridsearch_results.csv")

    try:
        pd.DataFrame(results).to_csv("lda_gridsearch_results.csv", index=False)
    except PermissionError:
        input("Could not save file, likely because file is open in another program. Press enter to try again.")
        pd.DataFrame(results).to_csv("lda_gridsearch_results.csv", index=False)
    
    
    

if __name__ == "__main__":
    tweets = pd_read("preprocessed_negative_tweets.csv")
    print("Fitting")
    return_list = fit(tweets)
    print("Testing on unseen tweets.")
    predictions = predict(tweets, return_list)
    print("Saving to tweets_with_topics.csv")
    data = {"Predictions" : predictions, "Text" : tweets["text"].values}
    df = pd.DataFrame(data=data)
    df.to_csv("tweets_with_topics.csv", index=False)
    



    
