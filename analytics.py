import re
import pickle
import torch
import nltk
import sklearn

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
import unicodedata


model = load_model('model.h5')
sentiment_model = pipeline('sentiment-analysis',
                   tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
                   model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"))
ner_model = spacy.load('en_core_web_lg')
desaster_model = pickle.load(open("desaster_multinomial_NB.sav", 'rb'))
tfidf = pickle.load(open('feature.pkl', 'rb'))
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# reference:~ https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/nlp%20proven%20approach/contractions.py

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def clean_text(raw_text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", raw_text).split())

def get_entities(raw_text):
    #text = clean_text(raw_text)
    doc = ner_model(raw_text)
    ents = []
    for ent in doc.ents:
        e = (ent.text, ent.label_)
        ents.append(e)
    return ents

def get_sentiment(raw_text):
    text = clean_text(raw_text)
    sentiments = sentiment_model(text)
    encoder={'1 star': 'very negative', '2 stars': 'negative', '3 stars':'neutral', '4 stars':'positive', '5 stars': 'very positive'}
    
    return encoder[sentiments[0]['label']]

def get_disaster_prediction_old(raw_text):
    ps = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', raw_text)
    text = text.lower()
    text = text.split()
    text = [ps.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    features = tfidf.transform([text])
    pred = desaster_model.predict(features)
    if pred==0:
        pred="No"
    else:
        pred="Yes"
    return pred

def text_cleaning(text):
    """
    Returns cleaned text (Accented Characters, Expand Contractions, Special Characters)
    Parameters
    ----------
    text -> String
    """
    # remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # remove emails
    text = ' '.join([i for i in text.split() if '@' not in i])
    
    # remove urls
    text = re.sub('http[s]?://\S+', '', text)
    
    # expand contractions
    for word in text.split():
        if word.lower() in CONTRACTION_MAP:
            text = text.replace(word[1:], CONTRACTION_MAP[word.lower()][1:])
    
    # remove special characters
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    
    # remove extra white spaces
    text = re.sub('\s+', ' ', text)

    doc = ner_model(text)
    tokens = []
    
    for token in doc:
        if token.lemma_ != '-PRON-':
            tokens.append(token.lemma_.lower().strip())
        else:
            tokens.append(token.lower_)

    return ' '.join(tokens)

def get_disaster_prediction(raw_text_array):
    oov_token = '<unk>'
    padding_type = 'post'
    trunc_type = 'post'
    embedding_dim = 100
    max_len = 140

    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    print(type(raw_text_array))
    print(raw_text_array)

    cl_text_array=raw_text_array['text'].apply(text_cleaning)
    #test['text']

    test_seq = tokenizer.texts_to_sequences(cl_text_array) #might be false because of string vs dataframe
    test_pad = pad_sequences(test_seq, padding=padding_type, truncating=trunc_type, maxlen=max_len)

    pred_label_array = model.predict_classes(test_pad)
    #pred_label = (model.predict(test_pad) > 0.5).astype("int32")
    print(pred_label_array)
    return pred_label_array
    '''if "0" in pred_label:
        pred_label="No"
    else:
        pred_label="Yes"'''
