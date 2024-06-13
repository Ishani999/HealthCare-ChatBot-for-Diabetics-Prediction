from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

app = Flask(__name__)

def load_data():
    df = pd.read_csv('diabetes.csv')
    return df

def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def predict_diabetes(model, features):
    return model.predict(np.array(features).reshape(1, -1))[0]

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_model(X_train, y_train)

name = None
gender = None
pregnancies = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global name, gender, pregnancies

    data = request.json
    message = data.get('message', '')
    tokens = word_tokenize(message.lower())

    if message.lower() == 'hi':
        # If the user says "hi", respond with a greeting and ask for gender
        name = None
        gender = None
        pregnancies = None
        return jsonify({'response': "Hi! What is your name?"})

    if not name:
        # If the name is not set, set the name and ask for gender
        name = message
        return jsonify({'response': f"Hi {name}! Are you male or female?"})

    elif not gender:
        # If the gender is not set, set the gender and ask for the remaining features
        gender = message.lower()
        if gender not in ['male', 'female']:
            return jsonify({'response': "Please enter 'male' or 'female'."})
        if gender == 'male':
            pregnancies = 0  # Automatically set pregnancies to 0 for male users
            return jsonify({'response': "You are male. Please provide comma seperated values for the remaining 7 features."})
        elif gender == 'female':
            return jsonify({'response': "You are female. Please provide commoa seperated values for all 8 features."})

    elif gender == 'female':
        # If the user is female, set the pregnancies and proceed with the prediction
        try:
            # Assuming message is a comma-separated string of 8 numerical values
            features = [float(val) for val in message.split(',')]
            if len(features) != 8:
                return jsonify({'response': "Please provide values for all 8 features."})
            pregnancies = int(features[0])  # Assuming pregnancies is the first feature
            prediction = predict_diabetes(model, features)
            result = "Diabetes prediction: " + ("Positive" if prediction == 1 else "Negative")
            return jsonify({'response': result})
        except ValueError:
            return jsonify({'response': "Please enter valid numerical values."})

    elif gender == 'male':
        # If the user is male, proceed with the prediction
        try:
            # Assuming message is a comma-separated string of 7 numerical values
            features = [float(val) for val in message.split(',')]
            if len(features) != 7:
                return jsonify({'response': "Please provide values for all 7 features."})
            # Insert 0 at the beginning of the feature list
            features.insert(0, 0)
            prediction = predict_diabetes(model, features)
            result = "Diabetes prediction: " + ("Positive" if prediction == 1 else "Negative")
            return jsonify({'response': result})
        except ValueError:
            return jsonify({'response': "Please enter valid numerical values."})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)