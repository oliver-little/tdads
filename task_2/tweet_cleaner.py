# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:34:46 2020

@author: phsdd
"""

f = open("tweets.csv","rb")
filedat = f.read()
data=""
for x in range(len(filedat)):
    if filedat[x]<128:
        data += (chr(filedat[x]))
        
f.close()
f = open("Tweets_cleaned.csv","w")
f.write(data)
f.close()
