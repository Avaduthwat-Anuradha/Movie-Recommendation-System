import pandas as pd
import numpy as np
import tweepy
import traceback

auth = tweepy.OAuthHandler("aKBt8eJagd4PumKz8LGmZw", "asFAO5b3Amo8Turjl2RxiUVXyviK6PYe1X6sVVBA")
auth.set_access_token("1914024835-dgZBlP6Tn2zHbmOVOPHIjSiTabp9bVAzRSsKaDX", "zCgN7F4csr6f3eU5uhX6NZR12O5o6mHWgBALY9U4")
api = tweepy.API(auth)

dup = []

dataset = pd.read_csv("movies.csv",sep="::")
dataset = dataset.values
tweets = ''
for i in range(len(dataset)):
    user = dataset[i,0]
    tid = dataset[i,1]
    try:
        #print(tid)
        tweetData = ''
        for tweet in api.search(q=tid, lang="en", rpp=3):
            text = str(tweet.text)
            text = text.strip("\n")
            text = text.strip()
            text = text.replace('"', "")
            if text not in dup:
                dup.append(text)
                tweets+=str(user)+",\""+str(text)+"\""+"\n"
                print(str(user)+" \""+str(tweets)+"\"")
            
    except:
        traceback.print_exc()
        #pass    
    
f = open("tweets.csv", "w",encoding='utf-8')
f.write(tweets)
f.close()
