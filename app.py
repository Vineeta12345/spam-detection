from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')

def home():
	return render_template('home.html')

@app.route('/youtube',methods=['POST'])
def youtube():
	return render_template('youtube.html')

@app.route('/text',methods=['POST'])
def text():
	return render_template('text.html')

@app.route('/email',methods=['POST'])
def email():
	return render_template('email.html')

@app.route('/sms',methods=['POST'])
def sms():
	return render_template('sms.html')


@app.route('/predictyoutube',methods=['POST'])
def predictyoutube():
	df= pd.read_csv("YoutubeSpamMergedData.csv")
	df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	df_x = df_data['CONTENT']
	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('resultyoutube.html',prediction = my_prediction)

@app.route('/predictemail',methods=['POST'])
def predictemail():
	df= pd.read_csv("spam_ham_dataset.csv")
	df_data = df[["text","label_num"]]
	# Features and Labels
	df_x = df_data['text']
	df_y = df_data.label_num
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('resultemail.html',prediction = my_prediction)

@app.route('/predictsms',methods=['POST'])
def predictsms():
	df= pd.read_csv("SMSSPAMDATA.csv")
	df_data = df[["SMS","CLASS"]]
	# Features and Labels
	df_x = df_data['SMS']
	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('resultsms.html',prediction = my_prediction)

@app.route('/predicttext',methods=['POST'])
def predicttext():
	df= pd.read_csv("textspam.csv")
	df_data = df[["text","CLASS"]]
	# Features and Labels
	df_x = df_data['text']
	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('resulttext.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)