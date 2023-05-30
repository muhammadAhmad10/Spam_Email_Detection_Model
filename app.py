from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

# Load the trained machine learning model
# Loading the MultinomialNB model
with open('./models/MultinomialNB_KFold.pkl', 'rb') as f:
    MultinomialNB_KFold = pickle.load(f)
with open('./models/MultinomialNB_Train_Test.pkl', 'rb') as f:
    MultinomialNB_Train_Test = pickle.load(f)
# Loading the AdaBoost Classifier model
with open('./models/AdaBoost_Classifier_KFold.pkl', 'rb') as f:
    AdaBoost_Classifier_KFold = pickle.load(f)
with open('./models/AdaBoost_Classifier_Train_Test.pkl', 'rb') as f:
    AdaBoost_Classifier_Train_Test = pickle.load(f)
# Loading the Bagging Classifier model
with open('./models/Bagging_Classifier_KFold.pkl', 'rb') as f:
    Bagging_Classifier_KFold = pickle.load(f)
with open('./models/Bagging_Classifier_Train_Test.pkl', 'rb') as f:
    Bagging_Classifier_Train_Test = pickle.load(f)
# Loading the Decision Tree model
with open('./models/Decision_Tree_KFold.pkl', 'rb') as f:
    Decision_Tree_KFold = pickle.load(f)
with open('./models/Decision_Tree_Train_Test.pkl', 'rb') as f:
    Decision_Tree_Train_Test = pickle.load(f)
# Loading the GaussianNB model
with open('./models/GaussianNB_KFold.pkl', 'rb') as f:
    GaussianNB_KFold = pickle.load(f)
with open('./models/GaussianNB_Train_Test.pkl', 'rb') as f:
    GaussianNB_Train_Test = pickle.load(f)
# Loading the Gradient Boosting Classifier model
with open('./models/Gradient_Boosting_Classifier.pkl', 'rb') as f:
    Gradient_Boosting_Classifier_KFold = pickle.load(f)
with open('./models/Gradient_Boosting_Classifier_Train_Test.pkl', 'rb') as f:
    Gradient_Boosting_Classifier_Train_Test = pickle.load(f)
# Loading the KNN model
with open('./models/KNN_KFold.pkl', 'rb') as f:
    KNN_KFold = pickle.load(f)
with open('./models/KNN_Train_Test.pkl', 'rb') as f:
    KNN_Train_Test = pickle.load(f)
# Loading the Logistic Regression model
with open('./models/Logistic_Regression_KFold.pkl', 'rb') as f:
    Logistic_Regression_KFold = pickle.load(f)
with open('./models/Logistic_Regression_Train_Test.pkl', 'rb') as f:
    Logistic_Regression_Train_Test = pickle.load(f)
# Loading the Random Forest model
with open('./models/Random_Forest_KFold.pkl', 'rb') as f:
    Random_Forest_KFold = pickle.load(f)
with open('./models/Random_Forest_Train_Test.pkl', 'rb') as f:
    Random_Forest_Train_Test = pickle.load(f)
# Loading the SVC model
with open('./models/SVC_KFold.pkl', 'rb') as f:
    SVC_KFold = pickle.load(f)
with open('./models/SVC_Train_Test.pkl', 'rb') as f:
    SVC_Train_Test = pickle.load(f)


# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


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

        # Vectorize the text using the loaded vectorizer
        email_vectorized = vectorizer.transform([email])
        predicted_label = None

        # Make a prediction using the loaded model
        if(split == 'KFold' and model1 == 'MultinomialNB'):
            print(split, model1)
            predicted_label = MultinomialNB_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'MultinomialNB'):
            print(split, model1)
            predicted_label = MultinomialNB_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'AdaBoost_Classifier'):
            predicted_label = AdaBoost_Classifier_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'AdaBoost_Classifier'):
            predicted_label = AdaBoost_Classifier_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'Bagging_Classifier'):
            predicted_label = Bagging_Classifier_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'Bagging_Classifier'):
            predicted_label = Bagging_Classifier_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'Decision_Tree'):
            predicted_label = Decision_Tree_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'Decision_Tree'):
            predicted_label = Decision_Tree_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'GaussianNB'):
            predicted_label = GaussianNB_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'GaussianNB'):
            predicted_label = GaussianNB_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'Gradient_Boosting_Classifier'):
            predicted_label = Gradient_Boosting_Classifier_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'Gradient_Boosting_Classifier'):
            predicted_label = Gradient_Boosting_Classifier_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'KNN'):
            predicted_label = KNN_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'KNN'):
            predicted_label = KNN_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'Logistic_Regression'):
            predicted_label = Logistic_Regression_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'Logistic_Regression'):
            predicted_label = Logistic_Regression_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'Random_Forest'):
            predicted_label = Random_Forest_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'Random_Forest'):
            predicted_label = Random_Forest_Train_Test.predict(
                email_vectorized)
        elif(split == 'KFold' and model1 == 'SVC'):
            predicted_label = SVC_KFold.predict(
                email_vectorized)
        elif(split == 'Train_Test' and model1 == 'SVC'):
            predicted_label = SVC_Train_Test.predict(
                email_vectorized)
        else:
            predicted_label = ['Invalid Input']

        print(predicted_label[0])
        return render_template('prediction.html', email=email, prediction=predicted_label[0], split=split, model=model1)
    else:
        return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug=True)
