import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

# Download necessary NLTK data files
nltk.download('punkt')

# Training data: pairs of medical-related questions and responses
training_data = [
    ("Hey","Hello How can I help you ?"),
    ("What are the symptoms of COVID-19?", "Common symptoms include fever, dry cough, and tiredness."),
    ("How can I prevent the spread of COVID-19?", "Wear a mask, maintain social distancing, and wash your hands regularly."),
    ("What is the treatment for COVID-19?", "Treatment includes supportive care to relieve symptoms. Severe cases may require hospitalization."),
    ("What should I do if I think I have COVID-19?", "Isolate yourself and contact a healthcare provider for advice."),
    ("Can children get COVID-19?", "Yes, children can get COVID-19, but they generally have milder symptoms."),
    ("What are the side effects of the COVID-19 vaccine?", "Common side effects include pain at the injection site, tiredness, and mild fever."),
    ("What is hypertension?", "Hypertension is high blood pressure, a condition in which the force of the blood against the artery walls is too high."),
    ("What are the symptoms of hypertension?", "Symptoms include headaches, shortness of breath, and nosebleeds, but many people have no symptoms."),
    ("How can I manage my blood pressure?", "Eat a healthy diet, exercise regularly, reduce salt intake, and take prescribed medications."),
    ("What is diabetes?", "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high."),
    ("What are the symptoms of diabetes?", "Symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision."),
    ("How can I manage diabetes?", "Monitor blood sugar levels, eat a healthy diet, exercise regularly, and take prescribed medications."),
]

# Split training data into questions and responses
questions, responses = zip(*training_data)

# Vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Create a simple logistic regression model and train it
model = LogisticRegression()
model.fit(X, range(len(questions)))

context = {}

def get_response(user_message):
    user_message_vectorized = vectorizer.transform([user_message])
    prediction = model.predict(user_message_vectorized)[0]
    response = responses[prediction]
    
    # Add context handling
    if "COVID-19" in user_message:
        context["topic"] = "COVID-19"
    elif "hypertension" in user_message:
        context["topic"] = "hypertension"
    elif "diabetes" in user_message:
        context["topic"] = "diabetes"
    else:
        context["topic"] = None
    
    if context.get("topic") == "COVID-19":
        if "symptoms" in user_message:
            response += " Make sure to monitor your health and consult a doctor if symptoms persist."
        elif "prevent" in user_message:
            response += " Stay informed and follow public health guidelines."
    elif context.get("topic") == "hypertension":
        if "manage" in user_message:
            response += " Regular check-ups with your doctor are important."
    elif context.get("topic") == "diabetes":
        if "manage" in user_message:
            response += " Keeping a healthy lifestyle is crucial."
    
    return response

if __name__ == "__main__":
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit", "goodbye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = get_response(user_message)
        print("Chatbot:", response)
