# -*- coding: utf-8 -*-
"""
Program to create and manipulate a dictionary of the words from all the given tweets.

Initially provides the option to read in the tweets.csv file, and create the dictionary
from it, excluding the given stopwords, as well as any non-alphabetic/ascii characters,
which is then converted to a pandas dataframe and saved to a file DictData.csv.
In addition, some key features of the dataset relating to the sentiment evaluations
are displayed (but not saved, as it stands).

If the condition is false, the DictData is simply loaded from the file into a 
pandas dataframe, from where it can be analysed to find, for example, words which are 
particularly associated with positive or negative sentiment.

Created on Sat Jul 18 13:00:21 2020
@author: David Marples

"""

import pandas as pd
import numpy as np

def readRawData(tweetdata):
    """Read tweets, extract words from the text field, and construct a dictionary.
    
    Looks at each tweet, and scores whether it is positive,negative, or neutral (implicit)
    It then looks through each word in the text field.
    Any non-alphabetic characters are stripped, and words of length <2 are ignored
    If it's in the dictionary, then the total and relevant sentiment scores are updated.
    Otherwise it is added to the dictionary, with the "value" being a tuple of
    [total appearances,no. in positive tweets, n0. in negative tweets].
    Note that if a word appears several times in a tweet, all will be counted.
    
    The function returns the dictionary, together with the total number of tweets,
    and the number of positive and negative tweets (according to the original file).
    """
    
    """This is basically the NLTK stopword list, with a few removed (see below)"""
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
             "your", "yours", "yourself", "yourselves", "he", "him", "his", 
             "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
             "they", "them", "their", "theirs", "themselves", "what", "which", 
             "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
             "was", "were", "be", "been", "being", "have", "has", "had", "having", 
             "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
             "or", "because", "as", "until", "while", "of", "at", "by", "for", 
             "with", "about", "against", "between", "into", "through", "during", 
             "before", "after", "above", "below", "to", "from", "up", "down", 
             "in", "out", "on", "off", "over", "under", "again", "further", 
             "then", "once", "here", "there", "when", "where", "why", "how", 
             "all", "any", "both", "each", "few", "more", "most", "other", 
             "some", "such", "own", "same", "so", "than", "too", "very", "s", 
             "t", "can", "will", "just", "don", "should", "now"]
    
    """ deleted stop words: "no", "nor", "not", "only",    """
    
    all_words = {}
    positive = tweetdata[tweetdata.airline_sentiment == 'positive'].count()
    negative = tweetdata[tweetdata.airline_sentiment == 'negative'].count()
    tweetdata.text = tweetdata.text.str.lower()
    total = tweetdata.shape[0]
    for x in range(total):
        words = tweetdata.text[x].split()
        for word in words:
           if not word.isalpha():
                #Strip any non-alphabetical characters
                for character in word:
                    temp=""
                    if 'A' <=character <= 'Z' or 'a' <= character <= 'z':
                        temp += character
                word = temp
           if word in stopwords:
                word = ""
           if word not in all_words and len(word) >1:
                all_words[word] = [0,0,0]
           all_words[word][0] += 1
           if tweetdata.airline_sentiment[x] == "positive":
                all_words.get(word)[1] += 1
           if tweetdata.airline_sentiment[x] == "negative":
                all_words.get(word)[2] += 1          
    
    return all_words, total, negative, positive

#Here's the main body, which deals with the data
tweetdata = pd.read_csv("tweets.csv")
    
"""Initial test, to determine whether to create dictionary, or simply load it"""

if 1 == 0:
    all_words, total, negative, positive = readRawData(tweetdata)

    print("Total was ",total,", with ",positive," ranked positive and ",negative," ranked negative")
    
    """Here we look at the relative abundance of tweets rated positive or negative.
    The take-home message is that people prefer to whinge!"""
    sentratio = negative / positive
    print("The ratio of negative to positive was ",sentratio)
    """Show the fraction of all tweets that are postiive or negative
    The outcome is that about 63% are negative and about 16% are positive (rest neutral)"""
    negrate = negative / total
    posrate = positive / total
    print("the fraction of negative to total was ",negrate,", and postives was ",posrate)

    print("Total number of words collected was",len(all_words)) #8312 in its current form

    """Convert dictionary to a flat list, suitable to be imported into a pandas dataframe"""
    flat_list=[[entry[0],entry[1][0],entry[1][1],entry[1][2]] for entry in all_words.items()]

    dict = pd.DataFrame(flat_list)
    dict.columns = ['word','uses','positives','negatives']
    """Add some extra columns, created from existing data.
       The fractions give the fraction of tweets in which the word appeared
       which were positive or negative.
       The relative positivity or negativity relate that fraction to the overall
       fraction: in other words, is this word preferentially associated with 
       this particular sentiment.
       Because of the underlying positive (0.16) and negative (0.63) rates, the
       relative values need to be treated very differently: maybe something like 
       2.5 for positivity being similar to ~1.2 for negativity?
       When I think of a sensible approach to normalise this, I will.
    """
    dict['pos_fraction'] = dict['positives'] / dict['uses']
    dict['neg_fraction'] = dict['negatives'] / dict['uses']
    dict['rel_positivity'] = dict['pos_fraction'] / posrate
    dict['rel_negativity'] = dict['neg_fraction'] / negrate
    dict.to_csv("DictData.csv",index=False)


else:
    """Save the time and just load the existing version from file"""
    dict = pd.read_csv("DictData.csv")

"""I've relabelled the columns so they fit in my output window """
dict.columns = ['word','uses','pos','neg','pos_frac','neg_frac','rel_pos','rel_neg']


"""Make lists of all the words that are used 'commonly' (here 3 times in 15000 tweets...)
   AND have a relatively high number of associated positive or negative scores"""
common_pos = dict[(dict['uses']>3) & (dict['rel_pos']>1.2)]\
                .sort_values(by=['rel_pos','uses'],ascending=False)
common_neg = dict[(dict['uses']>3) & (dict['rel_neg']>1.2)]\
                .sort_values(by=['rel_neg','uses'],ascending=False)

#Combine the two lists, to give a set of key words 
keys = pd.concat([common_pos,common_neg],ignore_index=True)

#actual_sentiment = []
actual_sentiment = tweetdata.airline_sentiment
my_sentiment = []
tweet_vectors = np.zeros((len(tweetdata),len(keys)),dtype=np.int16)
for x in range(len(tweetdata)):
    score = 0
    #actual_sentiment.append(tweetdata.airline_sentiment[x])
    words = tweetdata.text[x].split()
       
    for word in words:
        word_line = keys[keys['word'] == word]
        if len(word_line)>0:
            index = word_line.index[0]
            tweet_vectors[x,index] += 1
            ps = word_line.rel_pos[index]
            ns = word_line.rel_neg[index]
            if ps > 1.2 and ns < 0.9:
                score += ps
            elif ps < 0.9 and ns > 1.2:
                score -= ns
                
    if score > 0.8:
        my_sentiment.append('positive')
    elif score <-1:
        my_sentiment.append('negative')
    else:
        my_sentiment.append('neutral')
right = 0
for x in range(len(my_sentiment)):
    if my_sentiment[x] == actual_sentiment[x]:
        right += 1
print("I got",right*100/len(my_sentiment),"% 'correct'.")

"""We'll save the vector data and labels, so we can use them from a NN.
   Vectors already in the form of a numpy array, so easy to write (and read)
   Labels exported as a text file, and processed in basic_NN"""

np.save("TweetVectors",tweet_vectors)
f = open("true_sentiment.txt","w")
for item in actual_sentiment:
    f.write(item)
    f.write(chr(10))
f.close()

