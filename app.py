import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Paste a news article below to check whether it is **Real or Fake**")

news_text = st.text_area("Enter News Text", height=200)

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter news text")
    else:
        news_vec = vectorizer.transform([news_text])
        pred = model.predict(news_vec)[0]
        prob = model.predict_proba(news_vec)[0]

        confidence = max(prob) * 100

        if pred == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

        st.info(f"Confidence: {confidence:.2f}%")
