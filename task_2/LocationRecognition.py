import numpy as np
import pandas as pd
from preproc import *

AIRPORTS = pd.read_csv("airport-codes.csv")
CITIES = pd.read_csv("world-cities.csv")

# This only considers 3 letter Airport codes and cities with up to two words in the name
# TODO: countries
def getLocationsInTweet(tweet_text):
    splitText = tweet_text.split()

    fromLocation = set()
    midLocation = set()
    toLocation = set()

    for x in range(len(splitText)):
        word = splitText[x]
        if word in ["from", "leaving"] and x < len(splitText) - 1:
            nextWord = splitText[x+1]
            if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                fromLocation.add(nextWord + " " + splitText[x+2])
            elif isLocation(nextWord):
                fromLocation.add(nextWord)
        elif word in ["to", "2"]:
            if x > 0:
                prevWord = splitText[x-1]
                if x > 1 and isLocation(splitText[x-2] + " " + prevWord):
                    fromLocation.add(splitText[x-2] + " " + prevWord)
                elif isLocation(prevWord):
                    fromLocation.add(prevWord)
            if x < len(splitText) - 1:
                nextWord = splitText[x+1]
                if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                    toLocation.add(nextWord + " " + splitText[x+2])
                elif isLocation(nextWord):
                    toLocation.add(nextWord)
        elif word in ["in", "into", "at"] and x < len(splitText) - 1:
            nextWord = splitText[x+1]
            if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                toLocation.add(nextWord + " " + splitText[x+2])
            elif isLocation(nextWord):
                toLocation.add(nextWord)
        elif word in ["via", "through"] and x < len(splitText) - 1:
            nextWord = splitText[x+1]
            if x < len(splitText) - 2 and isLocation(nextWord + " " + splitText[x+2]):
                midLocation.add(nextWord + " " + splitText[x+2])
            elif isLocation(nextWord):
                midLocation.add(nextWord)
    return (fromLocation, midLocation, toLocation)

def isLocation(locationString):
    return (len(locationString) == 3 and AIRPORTS["iata_code"].str.contains(locationString.upper()).any()) or CITIES["name"].str.lower().eq(locationString.lower()).any()
            
if __name__ == "__main__":
    tweets = pd_read("tweets.csv")
    tweets.text = remove_punctuation(lt_gt_conversion(with_without_conversion(arrow_conversion(remove_links(tweets.text)))))
    locationTweets = []

    for tweet in tweets["text"]:
        fromLocation, midLocation, toLocation = getLocationsInTweet(tweet)
        if len(fromLocation) > 0 or len(midLocation) > 0 or len(toLocation) > 0:
            locationTweets.append([tweet, fromLocation, midLocation, toLocation])
        
                                       
    

