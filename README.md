# 🧠 Emotion Detection from Text

This is a machine learning project where we built a text-based emotion detection system. Given a user's message, the model predicts the emotional tone — such as joy, sadness, anger, love, etc. The trained model is also deployed in a Streamlit web app for real-time predictions.

## 📌 Project Overview

- **Goal:** Detect human emotions from short text messages.
- **Model Used:** Logistic Regression with TF-IDF vectorization.
- **Dataset:** Emotion-labeled text samples (joy, sadness, anger, etc.)
- **Deployment:** Streamlit app with user input and emotion prediction.

## 🧪 Detected Emotions

- 😄 Joy  
- 😢 Sadness  
- 😠 Anger  
- 😨 Fear  
- ❤️ Love  
- 😲 Surprise

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- Streamlit  

## ⚙️ How It Works

The dataset is cleaned using NLTK (tokenization, stopword removal). Text is vectorized using TF-IDF. A Logistic Regression classifier is trained with class balancing. The model and vectorizer are saved using joblib, and the app is built with Streamlit. Users input a sentence, which is cleaned, vectorized, and predicted in real-time with a confidence score.

## 🚀 How to Run

### 1. Install required libraries

```bash
pip install streamlit scikit-learn pandas nltk joblib
```

### 2. Download NLTK resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 📂 Project Files

- `app.py` – Streamlit app code  
- `emotion_model.pkl` – Trained ML model  
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer  
- `emotion.csv` – Dataset used for training  
- `README.md` – Project documentation  


## 🌟 Outcome

- A fast and lightweight emotion classifier with real-time UI
- Can be used in chatbots, feedback analyzers, or mental health tools

## Future Enhancements
- Upgrade to BERT or LSTM for better accuracy
- Add more emotions like neutral or frustration

## ✍️ Credentials

Made for Tamizhan Skills RISE Internship  - June 2025 Batch


