import streamlit as st
import numpy as np
import pandas as pd
import json
import datetime
import time
from flask import Flask, jsonify, request, abort
from flask_restplus import Api, Resource, reqparse
from tweepy import OAuthHandler, API, Stream, Cursor
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import cm

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

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def main():
  st.sidebar.title('Twitter Analysis')
  page = st.sidebar.selectbox('Choose a page', ["Homepage", "Twitter Dashboard", "Explore Analytics Functions"])

  if page == 'Homepage':
    set_png_as_page_bg('background.png')

    st.text("")
    st.text("")
    st.text("")

    st.title('Twitter Analyse Dashboard')
    st.title('Big Data Consulting Project SS21')

    st.text("")
    st.text("")
    st.text("")

    expander = st.beta_expander("Who are we?")
    expander.write("A student trying to bring order into the chaotic world of Twitter!")

    expander2 = st.beta_expander("What are we doing?")
    expander2.write("Trying to get good grades by combining NLP methods, analysis on unstructured data and statistics to get insights of social dynamics. Complex interconnections in simple visualizations BLA.")

    expander3 = st.beta_expander("Why?")
    expander3.write("Because I have to")

  
  elif page == 'Explore Analytics Functions':
    set_png_as_page_bg('background2.png')

    st.text("")
    st.text("")
    st.text("")

    st.title('Analytics Functions')
    
    st.text("")
    st.text("")
    st.text("")

    user_input = st.sidebar.text_area("Your Tweet:")
    st.subheader('Your Tweet: ')
    st.markdown(user_input)

    options = st.sidebar.multiselect('What would you like to have analysed?',
    ['Entities', 'Sentiment', 'Disaster Classification'])

    st.text("")
    st.text("")
    st.text("")
    #taking the input of streamlit to process the analysis
    col1, col2, col3 = st.beta_columns(3)

    if ("Entities" in options):
      ents = analytics.get_entities(user_input)
      t = "<font color='red'>Entity Detection</font>"
      col1.markdown(t, unsafe_allow_html=True)
      for i in range(len(ents)):
        entities = ents[i][0]
        labels = ents[i][1]
        col1.text("entity " + str(i) + ": "+ entities)
        col1.text("label: " + labels)


    if "Sentiment" in options:
      sent = analytics.get_sentiment(user_input)
      t = "<font color='red'>Sentiment Analysis</font>"
      col2.markdown(t, unsafe_allow_html=True)
      col2.text("detected sentiment:")
      col2.text(sent)

    if "Disaster Classification" in options:
      df_user_input = pd.DataFrame([[user_input]], columns=["text"])
      disas = analytics.get_disaster_prediction(df_user_input)
      if disas == [[0]]:
        disas = "No"
      else: 
        disas = "Yes"
      t = "<font color='red'>Disaster Classification</font>"
      col3.markdown(t, unsafe_allow_html=True)
      col3.text("tweet describes a disaster:")
      col3.text(disas)



  elif page == 'Twitter Dashboard':
    set_png_as_page_bg('background2.png')
    
    st.text("")
    st.text("")
    st.text("")

    st.title('Twitter Dashboard')

    st.sidebar.title('Twitter Search')

    user_input2 = st.sidebar.text_input("Your Searchword")
    'Your Searchword: ', user_input2 #var for hashtag
    user_input2 += ' -filter:retweets'    
    today = datetime.date.today()
    lastdays = today - datetime.timedelta(days=7)
    st.sidebar.date_input('Start Date', lastdays, min_value = lastdays, max_value = today)
    st.sidebar.date_input('End Date', today, min_value = lastdays, max_value = today)
    tweets = Cursor(api_twitter.search, 
            q=user_input2, 
            since=lastdays, 
            until=today, 
            lang="en").items(50)

    tweets_list = [[tweet.text, 
         tweet.created_at, 
         tweet.id_str,             
         tweet.favorite_count, 
         tweet.retweet_count,
         [hashtag['text'] for hashtag in tweet.entities['hashtags']],
         tweet.user.screen_name,
         tweet.user.id_str, 
         tweet.user.location, 
         tweet.user.followers_count, 
         tweet.coordinates, 
         tweet.place] for tweet in tweets]

    cols = ['text', 'created_at', 'id', 'likes', 'retweets', 'hashtags', 'user_screen_name', 'user_id', 
        'user_location', 'user_followers', 'coordinates', 'place']
    
    df = pd.DataFrame(tweets_list, columns = cols)
    df['sentiment'] = list(map(analytics.get_sentiment, df['text'].values.tolist()))
    df['entities'] = list(map(analytics.get_entities, df['text'].values.tolist()))
    df['disaster'] = analytics.get_disaster_prediction(df)
    #df['entities'] = df['text'].apply(lambda x: analytics.get_entities(x))

    df_disaster = df[df['disaster'] == 1]
    
    st.text("")
    st.text("")
    st.text("")

    st.subheader('Matched Tweets')
    st.dataframe(df)

    st.text("")
    st.text("")
    st.text("")

    st.subheader('Tweets describing a disaster')
    st.dataframe(df_disaster)

    st.text("")
    st.text("")
    st.text("")

    st.subheader('Locations in dataset')

    selected_df = st.sidebar.radio('Select either classified tweet or all tweets', ("all tweets", "classified tweets"))

    col1, col2, col3 = st.beta_columns(3)

    if selected_df == "classified tweets":
      df=df_disaster

    #worldcloud for entity location
    ents = df['entities'].values.tolist()
    col1.subheader("Entities:")
    st.text("")
    st.text("")
    st.text("")
    ent_str=""
    for i in range(len(ents)):
      for j in range(len(ents[i])):
        if ents[i][j][1] == 'GPE': 
          ent_str = " ".join((ent_str, str(ents[i][j][0])))
    if ent_str:
      wordcloud = WordCloud(width=200, height=300, background_color="white", colormap="twilight_shifted").generate(ent_str) 
      col1.image(wordcloud.to_array())

    #wordcloud for entity person
    ent_str2=""
    col2.subheader("Persons:")
    st.text("")
    st.text("")
    st.text("")
    for i in range(len(ents)):
      for j in range(len(ents[i])):
        if ents[i][j][1] == 'PERSON': 
          ent_str2 = " ".join((ent_str2, str(ents[i][j][0])))
    if ent_str2:
      wordcloud = WordCloud(width=200, height=300, background_color="white", colormap="twilight_shifted").generate(ent_str2) 
      col2.image(wordcloud.to_array())

    #wordcloud for entity organisations
    ent_str3=""
    st.text("")
    st.text("")
    st.text("")
    col3.subheader("Organisations:")
    for i in range(len(ents)):
      for j in range(len(ents[i])):
        if ents[i][j][1] == 'ORG': 
          ent_str3 = " ".join((ent_str3, str(ents[i][j][0])))
    if ent_str3:
      wordcloud = WordCloud(width=200, height=300, background_color="white", colormap="twilight_shifted").generate(ent_str3) 
      col3.image(wordcloud.to_array())

    st.subheader('Percentage of tweets describing a disaster')
    hist_values = df.disaster.value_counts(normalize=True)
    st.bar_chart(hist_values)
    cs=cm.Set1(np.arange(40)/40.)

    st.subheader('Sentiments in dataset')
    sents = df['sentiment'].values.tolist()
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    sentiment_level = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    sizes = [sents.count(level)/len(sents) for level in sentiment_level]

    slices = [1,2,3] * 4 + [20, 25, 30] * 2
    cmap = plt.cm.twilight
    colors = cmap(np.linspace(0., 1., len(slices)))

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, startangle=90, colors=colors)
    ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)

    
if __name__ == '__main__':
  main()