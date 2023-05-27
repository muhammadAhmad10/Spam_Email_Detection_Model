from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

# Load the trained machine learning model
with open('MultinomialNB_KFold.pkl', 'rb') as f:
    MultinomialNB_KFold = pickle.load(f)

# Load the vectorizer for train test split
with open('vectorizer_train_test.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the vectorizer for KFold
with open('vectorizer_KFold.pkl', 'rb') as f:
    vectorizer_KFold = pickle.load(f)


@app.route('/')
def home():
    return render_template('prediction.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # Get the form data
        email = request.form.get('email')
        split = request.form.get('split')
        model1 = request.form.get('model')

        # Perform prediction
        email_vectorized_KFold = vectorizer_KFold.transform([email])
        email_vectorized = vectorizer.transform([email])
        predicted_label = None
        if(split == 'KFold' and model1 == 'MultinomialNB'):
            predicted_label = MultinomialNB_KFold.predict(
                email_vectorized_KFold)
        print(predicted_label[0])
        # Map predicted label to human-readable text
        # if predicted_label[0] == 0:
        #     prediction_text = 'Not Spam'
        # else:
        #     prediction_text = 'Spam'
        # Return the result to the template
        return render_template('prediction.html', email=email, prediction=predicted_label[0])
    else:
        return render_template('prediction.html')

# @app.route('/prediction', methods=['POST'])
# def predict():
#     # Get the input values from the HTML form
#     features = [float(x) for x in request.form.values()]

#     # Make the prediction using the loaded model
#     prediction = model.predict([features])

#     # Perform any additional processing with the prediction (e.g., convert to text label)
#     if prediction[0] == 0:
#         result = 'Not Spam'
#     else:
#         result = 'Spam'

#     return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
