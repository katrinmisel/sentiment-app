###### import libraries

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

    if prediction>0.45: 
        sentiment = 'Positive'
        face = ":slightly_smiling_face:"
    else: 
        sentiment = 'Negative'
        face = ":slightly_frowning_face:"

    return sentiment, face


###### set up streamlit

st.title("Sentiment prediction :robot_face:")

text = st.text_input(label="Please enter your text here to predict the sentiment")

st.write("")

# inputs = {"tweet": text}

if st.button('Predict'):
    st.write(f"Your text: {text}")
    sentiment, face = predict_sentiment(text)
    st.subheader(f"Sentiment prediction: {sentiment} {face}")
