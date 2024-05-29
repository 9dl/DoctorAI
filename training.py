import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset from JSON file
with open('dataset/data.json', 'r') as file:
    data = json.load(file)

# Extract symptoms and corresponding disorders from dataset
symptoms = [entry['query'] for entry in data]
disorders = [entry['response'] for entry in data]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms, disorders, test_size=0.2, random_state=42)

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")

# Save the trained model and vectorizer for later use in the chatbot
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'symptom_checker_model.joblib')
