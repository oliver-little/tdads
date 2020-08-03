# -*- coding: utf-8 -*-
"""
Program to analyse correlations within given tweet data.

Identifies the subset of tweets created in each hour of the day (24 hr clock)
using the 'tweet_created' field, and determines the total number, and number of
positive and negative tweets. It expresses the latter two as a percentage of the
total for that hour, and prints and plots the data.

It similarly bins and plots the data by airline.

Finally it converts the 'user_timezone' data for 'zones' with at least 20 tweets
to actual (GMT/UTC) timezones, and uses that to bin the data. 

Requires pandas, numpy and matplotlib.

Created on Thursday Jul 30th 2020
@author: David Marples

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def zone_Sort(e):
    #Function to allow sorting of the timezone list by number of tweets
    return e[1]

def DataPlot(*args):
    """Routine to plot a graph, given data in a numpy array.
       Expects either 4 or 5 parameters:
    (optional) X coordinates (otherwise y index is used)
    Y coordinates
    X axis label
    Y axis label
    Title for the graph """
    fig, ax = plt.subplots()
    if len(args) == 5:
        ax.plot(args[0],args[1])
    else:
        ax.plot(args[0])
    ax.set(xlabel=args[-3],ylabel=args[-2],title=args[-1])
    plt.show()


#Here's the main body, which deals with the data
tdat = pd.read_csv("tweets_cleaned.csv")


tdat['hour'] = tdat.tweet_created.str[-5:-3]

time_sentiment = np.zeros((24,5),dtype=np.int16)

for x in range(24):  
    #Make a 'time' string match what is in the dataframe
    if x > 9:
        time = str(x)
    else:
        time = '0' + str(x)  
    #select the elements of the dataframe that match this hour    
    hr = tdat[(tdat.hour == time)]
    time_sentiment[x,0] = len(hr)
    time_sentiment[x,1] = len(hr[(hr.airline_sentiment == 'positive')])
    time_sentiment[x,2] = len(hr[(hr.airline_sentiment == 'negative')])
    time_sentiment[x,3] = 100 * time_sentiment[x,1] / time_sentiment[x,0]
    time_sentiment[x,4] = 100 * time_sentiment[x,2] / time_sentiment[x,0]    
    print('During hour',x,'there were',time_sentiment[x,0],'tweets, of which',
          time_sentiment[x][3],'% were positive and',time_sentiment[x][4],'% negative')


DataPlot(time_sentiment[:,3:5],
         'time (hr)',
         '% of tweets at this time',
         'Positive (blue) and Negative (gold)Tweets by Time')


airlines = ['Virgin America','United','Southwest','Delta','US Airways','American']

al_sentiment = np.zeros((6,5),dtype=np.int16)
for x in range(len(airlines)):
    al = tdat[(tdat.airline == airlines[x])]
    al_sentiment[x,0] = len(al)
    al_sentiment[x,1] = len(al[(al.airline_sentiment == 'positive')])
    al_sentiment[x,2] = len(al[(al.airline_sentiment == 'negative')])
    al_sentiment[x,3] = 100 * al_sentiment[x,1] / al_sentiment[x,0]
    al_sentiment[x,4] = 100 * al_sentiment[x,2] / al_sentiment[x,0]
    
    print('At ',airlines[x],', there were',al_sentiment[x][0],'tweets, of which',
          al_sentiment[x][3],'% were positive and',al_sentiment[x][4],'% negative')

DataPlot(al_sentiment[:,3:5],
         'airline index',
         '% of tweets for this airline',
         'Positive (blue) and Negative (gold) Tweets by Airline')


"""Now create a list of all the 'time zones' listed in the user_timezone column,
   and work out how many of each there are, creating a list of tuples (zone,number). 
   Finally we sort the list from commonest to least common, so we can 
   add the (hard-coded) actual offsets from GMT/UTC in a dictionary. 
   This list could easily be extended. At the moment it covers all 'zones' with
   at least 20 uses, and the total is around 2/3 of all the tweets."""
timezones = tdat.user_timezone.unique()
tz_list = []
for tz in timezones:
    #create a list of how common each of those time zones is
    tz_list.append((tz,len(tdat[(tdat.user_timezone == tz)])))
tz_list.sort(reverse = True,key=zone_Sort)

#Hard-coded list for the entries with at least 20 entries:
times = [-5,-6,-8,-5,-4,-7,-7,0,-9,10,-10,1,-5,-5,-5,1,4,-3]
#Create a dictionary of time zones for the n most common 'zones'
tz_dict = {}
for x in range(len(timezones)):
    if x < len(times):
        tz_dict[tz_list[x][0]] = str(times[x])
    else:
        tz_dict[tz_list[x][0]] =' '

tdat['tz'] = '' #Create a new blank column, to write in the time zones we know
for x in range(len(tdat)):
    tdat.tz[x] = tz_dict[tdat.user_timezone[x]] 

tz_sentiment = np.zeros((25,6),dtype=np.int16)
for x in range(25):
    tz = tdat[(tdat.tz == str(x-12))]
    if len(tz) > 0:
        tz_sentiment[x,0] = len(tz)
        tz_sentiment[x,1] = len(tz[(tz.airline_sentiment == 'positive')])
        tz_sentiment[x,2] = len(tz[(tz.airline_sentiment == 'negative')])
        tz_sentiment[x,3] = 100 * tz_sentiment[x,1] / tz_sentiment[x,0]
        tz_sentiment[x,4] = 100 * tz_sentiment[x,2] / tz_sentiment[x,0]
    tz_sentiment[x,5] = x - 12
    
    print('For time zone',tz_sentiment[x,5],', there were',tz_sentiment[x][0],
          'tweets, of which ',tz_sentiment[x][3],'% were positive and',
          tz_sentiment[x][4],'% negative')

DataPlot(tz_sentiment[:,5],
         tz_sentiment[:,3:5],
         'Time zone (GMT = 0)',
         '% of tweets for this time zone',
         'Positive (blue) and Negative (gold) Tweets by Time zone')

