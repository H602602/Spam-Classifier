import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
# If you get a Unicode error, change encoding to "latin-1"
df = pd.read_csv("spam.csv", encoding="latin-1")

# Some versions have extra useless columns. Keep only text + label.
df = df[['v1', 'v2']]
df.columns = ['label', 'text']  # rename columns

# Convert labels: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Sample data:")
print(df.head())

# -----------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------
X = df['text']   # messages
y = df['label']  # 0 or 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep same spam/ham ratio
)

# -----------------------------
# 3. CONVERT TEXT TO NUMBERS (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',   # remove common words like "the", "and"
    max_features=5000       # keep top 5000 important words
)

X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocabulary + transform
X_test_tfidf = vectorizer.transform(X_test)        # only transform using same vocab

# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)

# -----------------------------
# 5. EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# -----------------------------
# 6. SAVE MODEL + VECTORIZER
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, os.path.join("models", "spam_tfidf_vectorizer.pkl"))
joblib.dump(model, os.path.join("models", "spam_classifier.pkl"))

print("\nSaved model and vectorizer in 'models/' folder.")
