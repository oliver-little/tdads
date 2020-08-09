import pandas as pd
import numpy as np
import ast

tweets = pd.read_csv("tweets_with_locations.csv")
sentiment_dict = dict(tuple(tweets.groupby("airline_sentiment")))
sentimentCounts = {}

for sentiment, sentiment_tweets in sentiment_dict.items():
    sentimentCounts[sentiment] = {}
    for locationString in sentiment_tweets["locations_from_text"]:
        # Hacky solution because of the way the locations were saved (as a list of dicts, but in the csv as a string)
        # Therefore, eval them to get the dict out (using this module makes it safer) then iterate over the list, and iterate over the dict
        locations = ast.literal_eval(locationString)
        # Set used to remove the effect of the same location being recognised more than once.
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

negativePercentages = {}

for key in commonKeys:
    total = sentimentCounts["negative"][key] + sentimentCounts["positive"][key] + sentimentCounts["neutral"][key]
    if total > 20:
        negativePercentages[key] = sentimentCounts["negative"][key] / total
        print(key + " " + str(negativePercentages[key]))


