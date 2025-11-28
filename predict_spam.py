import os
import joblib

VEC_PATH = os.path.join("models", "spam_tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join("models", "spam_classifier.pkl")

# Load saved objects
vectorizer = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)

def predict_message(text: str):
    """
    Takes a text message and returns:
    - label: 'Spam' or 'Not Spam'
    - proba: probability that it's spam (0 to 1)
    """
    X = vectorizer.transform([text])   # transform text -> TF-IDF
    pred = model.predict(X)[0]         # 0 or 1
    proba = model.predict_proba(X)[0][1]  # probability of spam (class 1)
    label = "Spam" if pred == 1 else "Not Spam"
    return label, proba

if __name__ == "__main__":
    print("Spam Detector – type a message (q to quit)\n")
    while True:
        msg = input("Message: ")
        if msg.lower() == "q":
            break
        label, p = predict_message(msg)
        print(f"Prediction: {label} (spam probability: {p:.3f})\n")
