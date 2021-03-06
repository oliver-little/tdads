import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_pos_neg(column):

  tweets = pd.read_csv("tweets.csv")
  categories = []
  pos_neg_counts = []

  #get all the categories possible for the given column
  for index, row in tweets.iterrows():
    if row[column] not in categories:
      categories.append(row[column])
      #for each new value there needs to
      pos_neg_counts.append([0,0])


  #count the positive and negatives for the categories
  for _, row in tweets.iterrows():
    index = categories.index(row[column])

    if row['airline_sentiment'] == 'positive':
      pos_neg_counts[index][0] += 1

    if row['airline_sentiment'] == 'negative':
      pos_neg_counts[index][1] += 1


  return (categories, pos_neg_counts)


def graph_maker(x_label, bar_names, pos_values, neg_values):

  x_pos = [i for i, _ in enumerate(bar_names)]
  plt.bar(x_pos, pos_values, width=0.2, color='green')

  x_pos = [i for i,_ in enumerate(bar_names)]
  plt.bar(x_pos, neg_values, width=0.2, color='red')

  plt.xlabel("Timezones")
  plt.ylabel("Counts")
  plt.xticks(x_pos, bar_names, rotation=0)

  colors = {'Negative':'red', 'Positive':'green'}
  labels = list(colors.keys())
  handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
  plt.legend(handles, labels)

  plt.show()


def get_proportions(pos_values, neg_values, min_tweets):

  pos_percents = []
  neg_percents = []

  for pos, neg in zip(pos_values, neg_values):
    total = int(pos) + int(neg)
    if total > min_tweets:
      pos_percents.append(int(pos/total*100))
      neg_percents.append(int(neg/total*100))

  final_list = [[x,y] for x,y in zip(pos_percents, neg_percents)]

  return final_list



"""
Categories: just the names of the bars
pos_neg_counts: just the counts returned by function
new_len: How many bars do you want
selection: Three types either:
          - Best: Take the n most positive results
          - Worst: Take the n most negative results
          - Combined: Take the n most tweeted results
"""

def shorten_values(categories, values, new_len, selection):

  temp_list = values
  temp_names = categories
  final_list_names = []
  final_list_values = []


  if selection == "best":
    while len(final_list_names) != new_len:

      # find the current max of the list and add it to the final list
      current_max = 0
      index = 0
      for x in temp_list:
        if x[0] > current_max:
          current_max = x[0]
          index = temp_list.index(x)

      highest = temp_list.pop(index)
      highest_name = temp_names.pop(index)
      if str(highest_name) != "nan":
        final_list_names.append(highest_name)
        final_list_values.append(highest)


  if selection == "worst":
    while len(final_list_names) != new_len:

      # find the current max of the list and add it to the final list
      current_max = 0
      index = 0
      for x in temp_list:
        if x[1] > current_max:
          current_max = x[1]
          index = temp_list.index(x)

      highest = temp_list.pop(index)
      highest_name = temp_names.pop(index)

      if str(highest_name) != 'nan':
        final_list_names.append(highest_name)
        final_list_values.append(highest)

  if selection == "combined":
    while len(final_list_names) != new_len:

      # find the current max of the list and add it to the final list
      current_max = 0
      index = 0
      for x in temp_list:
        if x[0]+x[1] > current_max:
          current_max = x[0]+x[1]
          index = temp_list.index(x)

      highest = temp_list.pop(index)
      highest_name = temp_names.pop(index)
      if str(highest_name) != "nan":
        final_list_names.append(highest_name)
        final_list_values.append(highest)

  return final_list_names, final_list_values

"""
=========
VARIABLES
=========
"""
#negative_reason, airline, name, retweet_count, tweet_created, tweet_location, user_timezone
column = str(input('Enter column to graph: '))
type_of_short = str(input('Enter way to sort (best, worst, combined): '))
amount_of_bars = int(input('How many bars on the chart: '))
min_tweets = int(input('Enter the minimum number of tweets to be included: '))


"""
============
COUNT GRAPHS
============
"""
counts = get_pos_neg(column)
shortened_counts = shorten_values(counts[0], counts[1], amount_of_bars, type_of_short)
names = shortened_counts[0]
values = shortened_counts[1]

# get the number of positives and negatives for the bar chart
pos_val = [x[0] for x in values]
neg_val = [x[1] for x in values]

graph_maker(column, names, pos_val, neg_val)
"""
===================
PROPORTIONAL GRAPHS
===================
"""
counts = get_pos_neg(column)
values = counts[1]

pos_val = [x[0] for x in values]
neg_val = [x[1] for x in values]
proportions = get_proportions(pos_val, neg_val, min_tweets)
shortened_counts = shorten_values(counts[0], proportions, amount_of_bars, type_of_short)

names = shortened_counts[0]
values = shortened_counts[1]
pos_val = [x[0]+x[1] for x in values]
neg_val = [x[1] for x in values]

graph_maker(column, names, pos_val, neg_val)
