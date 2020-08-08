import pandas as pd
import numpy as np
import ast

tweets = pd.read_csv("tweets_with_locations.csv")
negativeTweets = tweets[tweets["airline_sentiment"] == "negative"]
print(negativeTweets)
negativeLocationCounts = {}

for index, tweet in negativeTweets.iteritems():
    for stringLocations in tweets["locations_from_text"]:
        locations = ast.literal_eval(stringLocations)
        for locationDict in locations:
            for key, value in locationDict.items():
                if value not in negativeLocationCounts:
                    negativeLocationCounts[value] = 0
                else:
                    negativeLocationCounts[value] += 1

print(negativeLocationCounts)
