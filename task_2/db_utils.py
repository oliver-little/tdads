import sys
import json
import datetime
import sqlite3
import pandas as pd
import numpy as np

DB_NAME = "tweets.db"

def _populate_db(csv_name = "tweets.csv"):
    """ Populate database from provided CSV """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    csv_in = pd.read_csv(csv_name)
    csv_in.drop_duplicates("tweet_id", inplace=True)
    for _, row in csv_in.iterrows():

        lat, lon = None, None
        if str(row.tweet_coord) == "":
            lat, lon = json.loads(str(row.tweet_coord))

        datetime_obj = datetime.datetime.strptime(str(row.tweet_created), "%Y-%m-%d %H:%M:%S %z")

        dt_seconds =  (datetime_obj - datetime.datetime(1970, 1, 1, tzinfo=datetime_obj.tzinfo)).total_seconds()

        tup_to_push = (
            str(row.tweet_id),
            row.airline_sentiment,
            row.airline_sentiment_confidence,
            row.negativereason,
            row.negativereason_confidence,
            row.airline,
            row.name,
            row.retweet_count,
            row.text,
            lat,
            lon,
            row.tweet_created,
            row.tweet_location,
            dt_seconds,
        )
        c.execute("INSERT INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tup_to_push)
    conn.commit()
    conn.close()


def _create_db():
    """ Create the initial database structure. THIS WILL OVERWRITE THE DB """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DROP TABLE tweets")
    c.execute("""CREATE TABLE tweets(
                    tweet_id TEXT PRIMARY KEY,
                    airline_sentiment TEXT,
                    airline_sentiment_confidence REAL,
                    negativereason TEXT,
                    negativereason_confidence REAL,
                    airline TEXT,
                    name TEXT,
                    retweet_count INTEGER,
                    text TEXT,
                    lat REAL,
                    lon REAL,
                    tweet_created INTEGER,
                    tweet_location TEXT,
                    user_timezone TEXT
                )""")
    conn.commit()
    conn.close()


def format_outputs(lst):
    """ Format outputs into pd DataFrame """
    ret = pd.DataFrame(lst)
    ret.columns = ["id", "airline_sentiment", "airline_sentiment_confidence", "negativereason", "negativereason_confidence", "airline", "name", "retweet_count", "text", "lat", "lon", "tweet_created", "tweet_location", "user_timezone"]
    return ret


def sql_cmd(cmd):
    """ Directly execute a SQL command on the db. YOU DO THIS AT YOUR OWN RISK """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(cmd)
    conn.commit()
    conn.close()


def tweets_containing(text, as_pd=True):
    """ get all tweets containing the provided text

    Parameters:
        text (str): text to search for
        as_pd (bool): convert to pandas dataframe? optional, default true

    Returns:
        either
            pandas.DataFrame["id", "airline_sentiment", "airline_sentiment_confidence", "negativereason", "negativereason_confidence", "airline", "name", "retweet_count", "text", "lat", "lon", "tweet_created", "tweet_location", "user_timezone"]
        or
            list
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(f"SELECT * FROM tweets WHERE instr(text, '{text}') > 0")
    ret = c.fetchall()
    if as_pd:
        ret = format_outputs(ret)
    conn.close()
    return ret


if __name__ == '__main__':
    print("RUNNING THIS PROGRAM AS MAIN WILL OVERWRITE THE DATABASE. ARE YOU SURE? (Y/N)\n>  ", end="")
    choice = input()
    if choice.lower() != 'y':
       sys.exit(-1)
    _create_db()
    _populate_db()
