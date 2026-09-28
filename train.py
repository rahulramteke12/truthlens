import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Loading datasets...")
df_fake = pd.read_csv("dataset/Fake.csv")
df_true = pd.read_csv("dataset/True.csv")

# Clean publisher watermarks to avoid dataset bias leakage
def clean_source_watermarks(t, is_true=True):
    if not isinstance(t, str):
        return ""
    if is_true:
        # Strip location tags and Reuters attributions like "WASHINGTON (Reuters) -"
        t = re.sub(r'^[A-Z\s,/-]+?\(Reuters\)\s*[-–—:]\s*', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\bReuters\b', '', t, flags=re.IGNORECASE)
    else:
        # Strip common fake news outlet signatures and image tags
        t = re.sub(r'21st Century Wire\s*(says)?', '', t, flags=re.IGNORECASE)
        t = re.sub(r'Featured image via.*', '', t, flags=re.IGNORECASE)
        t = re.sub(r'Image via.*', '', t, flags=re.IGNORECASE)
        t = re.sub(r'READ MORE.*?$', '', t, flags=re.IGNORECASE)
    return t

print("Removing publisher leakage...")
df_true['text'] = df_true['text'].apply(lambda x: clean_source_watermarks(x, is_true=True))
df_fake['text'] = df_fake['text'].apply(lambda x: clean_source_watermarks(x, is_true=False))

df_fake["class"] = 0
df_true["class"] = 1

print("Merging and preparing dataset...")
df_merge = pd.concat([df_fake, df_true], axis=0, ignore_index=True)
# Combine title and text so headlines and full articles are well supported
df_merge["content"] = df_merge["title"].fillna('') + ' ' + df_merge["text"].fillna('')

def wordopt(text):
    """Cleans and preprocesses input text."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

print("Cleaning text data...")
df_merge["clean_content"] = df_merge["content"].apply(wordopt)
df = df_merge[df_merge["clean_content"].str.len() > 10].reset_index(drop=True)

X = df["clean_content"]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
xv_train = vectorizer.fit_transform(X_train)
xv_test = vectorizer.transform(X_test)

print("Training Ensemble Classifier (Logistic Regression + Multinomial Naive Bayes)...")
lr = LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced')
nb = MultinomialNB(alpha=0.1)
model = VotingClassifier(estimators=[('lr', lr), ('nb', nb)], voting='soft', weights=[1.0, 1.5])
model.fit(xv_train, y_train)

# Evaluation
y_pred = model.predict(xv_test)
print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained Model and Vectorizer
print("Saving model.pkl and vectorizer.pkl...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Success! You can now run app.py.")