
import streamlit as st
import joblib
import numpy as np

def tokenizer_porter(text):
    from nltk.stem import PorterStemmer
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

# Load the trained TF-IDF Vectorizer and Random Forest model
tfidf = joblib.load("./model/tfidf_vectorizer.pkl")
NB = joblib.load("./model/navie_bayes.pkl")

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🎬 Movie Review Sentiment Classifier")
st.write("Enter a movie review to analyze its sentiment.")

# User Input
user_input = st.text_area("Enter your movie review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Transform input using the TF-IDF vectorizer
        input_vectorized = tfidf.transform([user_input])
        
        # Predict sentiment
        prediction = NB.predict(input_vectorized)[0]
        pred_prob = NB.predict_proba(input_vectorized)[0]
        confidence = np.max(pred_prob) * 100
        
        # Display result
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.subheader(f"Predicted Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Scikit-learn")