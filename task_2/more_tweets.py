import pandas as pd
import GetOldTweets3 as got

tweetcriteria = got.manager.TweetCriteria().setQuerySearch("to:jet2tweets")\
                                           .setSince("2015-03-01")\
                                           .setUntil("2015-04-01")\

tweets = got.manager.TweetManager.getTweets(tweetcriteria)
tweets_list = []
for tweet in tweets:
    tweet_list = {}
    tweet_list["text"] = tweet.text
    tweet_list["tweet_id"] = tweet.id
    tweet_list["name"] = tweet.username
    tweet_list["tweet_created"] = tweet.date
    tweet_list["retweet_count"] = tweet.retweets
    tweet_list["favourites_count"] = tweet.favorites
    tweet_list["tweet_coordinates"] = tweet.geo
    tweets_list.append(tweet_list)

tweets_df = pd.DataFrame(tweets_list)
print(tweets_df)
# tweets_df.columns = ["tweet_id", "permalink", "name", "text", "tweet_created", "retweet_count", "favourites_count"]
tweets_df.to_csv("extra_data/jet2new.csv", index=False)
