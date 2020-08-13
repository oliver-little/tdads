import numpy as np
import pandas as pd
import pickle
import os
from preproc import *

OUT_DIR = "../code_output/"

pd.options.mode.chained_assignment = None 

AIRPORTS = pd.read_csv("airport-codes.csv")
CITIES = pd.read_csv("world-cities.csv")

# This only considers 3 letter Airport codes and cities with up to two words in the name
def getLocationsInTweet(tweet_text):
    splitText = tweet_text.split()

    locations = []

    for x in range(len(splitText)):
        word = splitText[x]
        thisLocation = {}
        if word in ["from", "leaving"] and x < len(splitText) - 1:
            nextWord = splitText[x+1]
            if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                thisLocation["from"] = nextWord + " " + splitText[x+2]
            elif isLocation(nextWord):
                thisLocation["from"] = nextWord
        elif word in ["to", "2"]:
            if x > 0:
                prevWord = splitText[x-1]
                if x > 1 and isLocation(splitText[x-2] + " " + prevWord):
                    thisLocation["from"] = splitText[x-2] + " " + prevWord
                elif isLocation(prevWord):
                    thisLocation["from"] = prevWord
            if x < len(splitText) - 1:
                nextWord = splitText[x+1]
                if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                    thisLocation["to"] = nextWord + " " + splitText[x+2]
                elif isLocation(nextWord):
                    thisLocation["to"] = nextWord
        elif word in ["in", "into", "at"] and x < len(splitText) - 1:
            nextWord = splitText[x+1]
            if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                thisLocation["to"] = nextWord + " " + splitText[x+2]
            elif isLocation(nextWord):
                thisLocation["to"] = nextWord
        elif word in ["via", "through"] and x < len(splitText) - 1:
            nextWord = splitText[x+1]
            if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                thisLocation["mid"] = nextWord + " " + splitText[x+2]
            elif isLocation(nextWord):
                thisLocation["mid"] = nextWord
        if thisLocation != {}:
            locations.append(thisLocation)
    return locations

def isLocation(locationString):
    return (len(locationString) == 3 and AIRPORTS["iata_code"].str.contains(locationString.upper()).any()) or CITIES["name"].str.lower().eq(locationString.lower()).any()
            
if __name__ == "__main__":
    tweets = pd_read("tweets.csv")
    tweets.text = remove_punctuation(lt_gt_conversion(with_without_conversion(arrow_conversion(remove_links(tweets.text)))))
    locationTweets = []

    tweets["locations_from_text"] = np.nan
    for index, tweet in tweets.iterrows():
        locations = getLocationsInTweet(tweet["text"])
        if len(locations) > 0:
            col = tweets["locations_from_text"]
            col.loc[index] = locations
            tweets["locations_from_text"] = col

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    tweets = tweets.dropna(subset=["locations_from_text"])
    tweets.to_csv(OUT_DIR + "tweets_with_locations.csv", index_label=False)
    print("Saving tweets with locations to: " + OUT_DIR + "tweets_with_locations.csv")

    sentiment_dict = dict(tuple(tweets.groupby("airline_sentiment")))
    sentimentCounts = {}

    for sentiment, sentiment_tweets in sentiment_dict.items():
        sentimentCounts[sentiment] = {}
        for locations in sentiment_tweets["locations_from_text"]:
            type(locations)
            thisTweet = set()
            for locationDict in locations:
                for key, location in locationDict.items():
                    if location not in thisTweet:
                        if location not in sentimentCounts[sentiment]:
                            sentimentCounts[sentiment][location] = 1
                        else:
                            sentimentCounts[sentiment][location] += 1
                        thisTweet.add(location)

    commonKeys = sentimentCounts["positive"].keys() & sentimentCounts["negative"].keys() & sentimentCounts["neutral"].keys()

    totalsList = []

    for key in commonKeys:
        total = sentimentCounts["negative"][key] + sentimentCounts["positive"][key] + sentimentCounts["neutral"][key]
        if total > 20:
            totalsList.append([key, sentimentCounts["positive"][key], sentimentCounts["neutral"][key], sentimentCounts["negative"][key], total])

    df = pd.DataFrame(data=totalsList, columns=["key", "positive", "neutral", "negative", "total"])
    df.to_csv(OUT_DIR + "locations_totals.csv", index=False, index_label=False)
    print("Saving location totals to: " + OUT_DIR + "tweet_with_locations.csv")
    

