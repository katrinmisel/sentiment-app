# Sentiment prediction with a deep learning model 

This is a streamlit app that's running at https://sentiment-prediction.streamlitapp.com/

The model used here is a simple Keras sequential model with: 
- a GloVe embedding layer
- a bidirectional LSTM layer
- a dense layer

The tweets are preprocessed thanks to the tweet-preprocessor library: https://pypi.org/project/tweet-preprocessor/
It has been trained on 40000 positive tweets and 40000 negative tweets from the Sentiment140 dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
