from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('../Model/spamFinder.pickle', 'rb') as file:
    model = pickle.load(file)

with open('../Model/vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)
    
def classify(inputMail):
    inputMail_transformed = vectorizer.transform([inputMail])
    result = model.predict(inputMail_transformed)  # [0] or [1]
    return result[0]  # Return the first (and only) prediction

@app.route('/', methods=['POST', 'GET'])
def index():
    spamOrNot = 2
    if request.method == 'POST':
         inputMail = request.form['inputMail']
         spamOrNot = classify(inputMail)
    return render_template('index.html', spamOrNot=spamOrNot)

if __name__ == '__main__':
    app.run(debug=True)