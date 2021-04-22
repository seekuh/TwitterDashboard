import json
from datetime import datetime
from flask import Flask, jsonify, request, abort
from flask_restplus import Api, Resource, reqparse
from tweepy import OAuthHandler, API, Stream, Cursor

import encrypt
import analytics

creds = encrypt.decrypt_file('src/twitter.key', 'src/twitter_credentials.json')

creds = json.loads(creds.decode('utf-8'))

auth = OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
api_twitter = API(auth)

params = reqparse.RequestParser()
params.add_argument('text', required = True, help='Text from tweet')
params.add_argument('entities', choices=('yes', 'no'), required = False, help='Find entities in Tweet')
params.add_argument('sentiment', choices=('yes', 'no'), required = False, help='Detect sentiment of tweet')
params.add_argument('disaster', choices=('yes', 'no'), required = False, help='Does the tweet describe a disaster?')

app = Flask(__name__)
api = Api(app=app, version='1.0', title='Twitter Analysis', description='Service to analyze tweets.')

@api.doc(params={})
@api.route('/twitteranalysis')
class TwitterAnalysis(Resource):
 
  @api.expect(params)
  def post(self):
    args = params.parse_args()
    if not args['text']:
      abort(400, 'There is no text given.')

    if args['text'] == '':
      abort(400, 'Text parameter may be empty.')

    results = {'text': args['text']}

    if args['entities'] == 'yes':
      ents = analytics.get_entities(args['text'])
      results.update({'entities': ents})

    if args['sentiment'] == 'yes':
        sent = analytics.get_sentiment(args['text'])
        results.update({'sentiment': sent})

    if args['disaster'] == 'yes':
        disas = analytics.get_disaster_prediction(args['text'])
        results.update({'disaster': disas})

    return(results)

params_search = reqparse.RequestParser()
params_search.add_argument('text', required = True, help='Search term to look for')

@api.doc(params_search={})
@api.route('/twittersearch')
class TwitterSearch(Resource):
 
  @api.expect(params_search)
  def post(self):
    args = params_search.parse_args()
    if not args['text']:
      abort(400, 'There is no text given.')

    if args['text'] == '':
      abort(400, 'Text parameter may be empty.')
    search_words = args['text']+' -filter:retweets'
    # from_date = datetime.today() - timedelta(days = 7)
    # to_date = datetime.today().strftime('%Y-%m-%d')
    tweets = Cursor(api_twitter.search, q=search_words, lang="en").items(10)
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

    return(tweets_list)

if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0', port='5000')


