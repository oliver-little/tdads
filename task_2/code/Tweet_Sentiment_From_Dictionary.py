# -*- coding: utf-8 -*-
"""
Program to analyse tweets using the dictionary created from tweets.csv.

Initially the specified tweet file is read in and cleaned, to remove the given 
stopwords, as well as any non-alphabetic/ascii characters, and converted to a 
pandas dataframe.

The dictionary is loaded, and a subset of key words is identified, depending on 
how many times they occurred, and how strong their positive or negative
associations are.

The test tweets are then searched for the key words, and a weighted sum of their 
sentiment is used to classify the tweet.

The classification determined is compared with the official values, and a 
success rate determined.

Finally the word matches, stored as vectors, and the offical sentiment labels 
are stored, so they can be passed to the simple neural net classifier.

Created on Sat Jul 18 13:00:21 2020
@author: David Marples

"""

import pandas as pd
import numpy as np

def readAndCleanData(filename):
    """Read tweets, and clean the text field.
    
    Looks at each tweet, and each word in the text field.
    Any non-alphabetic characters are stripped, and words of length <2,
    and stopwords, are ignored
     
    The function returns the tweet data in a pandas dataframe.
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
    
    tweetdata = pd.read_csv(filename)
    tweetdata.text = tweetdata.text.str.lower()
    total = tweetdata.shape[0]
    for x in range(total):
        cleaned_string = ""
        words = tweetdata.text[x].split()
        for word in words:
           if not word.isalpha():
                #Strip any non-alphabetical characters
                temp=""
                for character in word:
                    if 'A' <=character <= 'Z' or 'a' <= character <= 'z':
                        temp += character
                word = temp
           if word in stopwords:
                word = ""
           cleaned_string += word + ' '
        tweetdata.text[x] = cleaned_string 
    return tweetdata

#Here's the main body, which deals with the data
filename = "jet2new_Combined.csv"
tweetdata = readAndCleanData(filename)
    
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
vecfilename = filename[:-4]+"Vectors"
sentimentfilename = filename[:-4]+"_true_sentiment.txt"
print(filename)
print(vecfilename)
print(sentimentfilename)
np.save(vecfilename,tweet_vectors)
f = open(sentimentfilename,"w")
for item in actual_sentiment:
    f.write(item)
    f.write(chr(10))
f.close()

