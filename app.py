import streamlit as st
import pickle

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="wide"
)

# Load model and vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

emotion_emoji = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

st.title("🧠 Emotion Detection using NLP")

st.write(
    "Enter any sentence and the model will predict the emotion."
)

text = st.text_area(
    "Enter Text",
    height=200
)

if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        emotion_map = {
            0: "sadness",
            1: "anger",
            2: "love",
            3: "surprise",
            4: "fear",
            5: "joy"
        }
        emotion = emotion_map[int(prediction)]

        emoji = emotion_emoji.get(emotion, "🤔")

        st.success(
            f"{emoji} Predicted Emotion: {emotion.upper()}"
        )