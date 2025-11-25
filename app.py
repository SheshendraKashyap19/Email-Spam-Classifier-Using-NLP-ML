import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK packages
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load model + vectorizer
model = joblib.load("email_spam_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    # Use split() instead of word_tokenize → avoids Streamlit NLTK errors
    tokens = text.split()

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)

st.title("📧 Email Spam Classifier (NLP + ML)")

email_text = st.text_area("Enter email content to classify")

if st.button("Predict"):
    clean = preprocess(email_text)
    vector = tfidf.transform([clean])
    result = model.predict(vector)[0]

    if result == 1:
        st.error("🚨 This email is *SPAM*! Beware of scams.")
    else:
        st.success("✅ This email is *GENUINE*. No spam detected.")
