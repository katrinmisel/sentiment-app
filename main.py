import pandas as pd
import pickle
from fastapi import FastAPI, Request

api = FastAPI()

#model_filename = "perf_advanced_tweetprep_glove.h5"
#tokenizer_filename = "keras_tokenizer.pickle"

#model = load_model(model_filename)

#with open(tokenizer_filename,'rb') as handle:
#  keras_tokenizer = pickle.load(handle)

@app.get("/")

def root():
	return {'message': 'Hello friends!'}

#@api.post('/predict')

#async def predict(iris: Iris):
	
	# Converting input data into Pandas DataFrame
	#input_df = pd.DataFrame([iris.dict()])
	
	# Getting the prediction from the Logistic Regression model
	#pred = lr_model.predict(input_df)[0]
	
	#return pred
