import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)

# Set environment path
nltk.data.path.append(nltk_data_dir)


nltk.download('punkt')
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# Preprocess input text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.title("🎭 Emotion Detection from Text")
st.write("Enter your message below, and the AI will predict the emotion:")

user_input = st.text_area("Your Message")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = preprocess(user_input)
        vectorized = vectorizer.transform([clean])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Emotion: **{prediction.upper()}**")
 
