import numpy as np
import joblib
from flask import Flask, request, jsonify

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('symptom_checker_model.joblib')

# Function to predict disorders based on symptoms
def predict_disorders(symptoms, top_n=3):
    symptoms_vectorized = vectorizer.transform([symptoms])
    predictions = model.predict(symptoms_vectorized)
    
    # Get the top predicted disorders
    top_predictions = []
    for prediction in np.argsort(model.predict_proba(symptoms_vectorized), axis=1)[0][::-1][:top_n]:
        top_predictions.append(model.classes_[prediction])

    return top_predictions

# Flask web server
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Symptom Checker Bot! Use /predict to check symptoms."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data['symptoms']
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    try:
        top_predictions = predict_disorders(symptoms)
        return jsonify({'predictions': top_predictions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
