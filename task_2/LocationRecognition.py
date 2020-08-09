import numpy as np
import pandas as pd
import pickle
from preproc import *

pd.options.mode.chained_assignment = None 

AIRPORTS = pd.read_csv("airport-codes.csv")
CITIES = pd.read_csv("world-cities.csv")

# This only considers 3 letter Airport codes and cities with up to two words in the name
# TODO: countries
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

    tweets = tweets.dropna(subset=["locations_from_text"])
    tweets.to_csv("tweets_with_locations.csv", index_label=False)       
    

