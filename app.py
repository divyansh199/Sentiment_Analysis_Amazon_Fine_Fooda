from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import spacy
import re

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vect= pickle.load(open('vectorizer.pkl','rb'))
# nlp = spacy.load('en_core_web_sm', disable= ['parser','ner'])
def decontracted(phrase):
	# specific
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)

	# general
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
			"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
			'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
			'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
			'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
			'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
			'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
			'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
			'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
			'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
			's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
			've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
			"hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
			"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
			'won', "won't", 'wouldn', "wouldn't"])

def clean_text(sentance):
	sentance = re.sub(r"http\S+", "", sentance)
	sentance = BeautifulSoup(sentance, 'lxml').get_text()
	sentance = decontracted(sentance)
	sentance = re.sub("\S*\d\S*", "", sentance).strip()
	sentance = re.sub('[^A-Za-z]+', ' ', sentance)
	sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
	return sentance.strip()


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods = ["GET", "POST"])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		review_text= decontracted(message)
		review_text= clean_text(message)
		test_vect  = vect.transform(([review_text]))
		my_prediction = model.predict(test_vect)[0]
		if my_prediction == 1:
			url = 'https://media1.tenor.com/m/-8Uay6X3E3UAAAAC/gil-cat.gif'
			senti = 'Positive'
		else:
			url = "https://media1.tenor.com/m/pxHTd5NVlREAAAAd/grr.gif"
			senti = 'Negitive'

		return render_template('index.html', prediction = [url,senti] )

if __name__ == '__main__':
	app.run(debug=True)