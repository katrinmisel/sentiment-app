###### import libraries

from fastapi import FastAPI
from pydantic import BaseModel
from keras.utils import pad_sequences
import preprocessor as p
import pickle
from keras.models import load_model
import streamlit as st
import json
import requests

###### load model and prediction function

model_dir = "perf_advanced_tweetprep_glove.h5"
tokenizer_dir = "keras_tokenizer.pickle"

model = load_model(model_dir)
with open(tokenizer_dir, 'rb') as handle:
    keras_tokenizer = pickle.load(handle)

def predict_sentiment(tweet):

    tweet_cleaned = [p.clean(tweet)]
    token_tweet = keras_tokenizer.texts_to_sequences(tweet_cleaned)
    pad_tweet = pad_sequences(token_tweet, padding='post', maxlen=100)

    prediction = model.predict(pad_tweet)

    if prediction>0.45: sentiment = 'Positive'
    else: sentiment = 'Negative'

    return sentiment

###### set up fastapi

class User_input(BaseModel):
    tweet : str

app = FastAPI()

@app.post('/predict_sentiment')
def operate(input:User_input):
    result = predict_sentiment(input.tweet)
    return result

###### set up streamlit

st.title("Sentiment prediction")

text = st.text_input(label="Please enter your text here to predict the sentiment")

st.write("")

inputs = {"tweet": text}

if st.button('Predict'):
    res = requests.post(url = "https://katrinmisel-sentimentapp-app-uylw02.streamlitapp.com/predict_sentiment", data = json.dumps(inputs))

    st.subheader(f"Response from API = {res.text}")
