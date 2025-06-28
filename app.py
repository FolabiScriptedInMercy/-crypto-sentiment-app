import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("model/crypto_sentiment_model_10000feat_v2.pth")
vectorizer = joblib.load("model/tfidf_vectorizer_10000feat.pkl")

# Map label IDs back to sentiment names
label_map = {
    0: "Bullish 🚀",
    1: "FUD 😡",
    2: "Neutral 😐",
    3: "Scam 🤡",
    4: "Shilling 🤑"
}

# UI Title
st.set_page_config(page_title="Crypto Sentiment Classifier")
st.title("🧠 Crypto Sentiment Classifier")
st.caption("Paste a crypto tweet or message and get its sentiment instantly!")

# Text input
message = st.text_area("Paste your message here...", "")

if st.button("✅ Classify Sentiment"):
    if message.strip() != "":
        # Vectorize the input
        X_input = vectorizer.transform([message])
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        sentiment = label_map[prediction]
        
        # Show result
        st.success(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a message before classification.")
