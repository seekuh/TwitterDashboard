import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime
from flask import Flask, jsonify, request, abort
from flask_restplus import Api, Resource, reqparse
from tweepy import OAuthHandler, API, Stream, Cursor
import base64
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import encrypt
import analytics

st.set_page_config(
layout="centered"
)

# Connection Twitter API
creds = encrypt.decrypt_file('twitter.key', 'twitter_credentials.json')
creds = json.loads(creds.decode('utf-8'))

auth = OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
api_twitter = API(auth)

side_bg = "seekuh.jpg"
side_bg_ext = "jpg"

def main():
  st.sidebar.title('Twitter Analysis')
  page = st.sidebar.selectbox('Choose a page', ["Homepage", "Twitter Dashboard", "Explore Analytics Functions"])

  if page == 'Homepage':
    st.image(side_bg)
  
  elif page == 'Explore Analytics Functions':
    st.title('Analytics Functions')

    user_input = st.sidebar.text_area("Your Tweet:")
    st.header('Your Tweet: ')
    st.markdown(user_input) #var for tweet to analyse

    options = st.sidebar.multiselect('What would you like to have analysed?',
    ['Entities', 'Sentiment', 'Disaster Classification'])


    #taking the input of streamlit to process the analysis
    if ("Entities" in options):
      ents = analytics.get_entities(user_input)
      st.header("detected entities:")
      st.markdown(ents)
      ent_str=""
      for i in range(len(ents)):
        if ents[i][1] not in ['PERCENT', 'CARDINAL']: #Wordcloud hat ein Problem mit Zahlen
          ent_str = " ".join((ent_str, str(ents[i][0])))
      print(ent_str)
      if ent_str:
        wordcloud = WordCloud(width=400, height=200, background_color="white", colormap="seismic").generate(ent_str) 
        st.image(wordcloud.to_array())

    if "Sentiment" in options:
      sent = analytics.get_sentiment(user_input)
      st.header("detected sentiment:")
      st.markdown(sent)

    if "Disaster Classification" in options:
      disas = analytics.get_disaster_prediction(user_input)
      st.markdown("Disaster or not:")
      st.markdown(disas)


  elif page == 'Twitter Dashboard':
    st.title('Twitter Dashboard')

    st.sidebar.title('Twitter Search')

    user_input2 = st.sidebar.text_input("Your Searchword")
    'Your Search Word: ', user_input2 #var for hashtag

    user_input2 += ' -filter: retweet'
    today = datetime.date.today()
    lastdays = today - datetime.timedelta(days=7)
    start_date = st.date_input('Start date', today)
    end_date = st.date_input('End date', lastdays)

    tweets = Cursor(api_twitter.search, 
            q=user_input2, 
            lang="en").items(10)
    tweets_list = {tweet.id_str: {'text': tweet.text, 
         'creation date': tweet.created_at.strftime('%Y-%m-%d'), 
         'id': tweet.id_str, 
         'favourite_counts': tweet.favorite_count, 
         'retweet_counts': tweet.retweet_count,
         'hashtags': [hashtag['text'] for hashtag in tweet.entities['hashtags']],
         'user_name': tweet.user.screen_name,
         'user_id': tweet.user.id_str, 
         'user_location': tweet.user.location, 
         'user_follower_count': tweet.user.followers_count, 
         'coordinates': tweet.coordinates, 
         'location': tweet.place} for tweet in tweets}


if __name__ == '__main__':
  main()