from pathlib import Path
import re

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_knn_regressor.joblib"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.joblib"


def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def interpret_rating(value):
    if value < 2.5:
        return "Negatif"
    if value < 3.5:
        return "Netral"
    return "Positif"


def format_rating(value):
    return f"{value:.2f}".replace(".", ",")


def reset_state():
    st.session_state.review_text = ""
    st.session_state.prediction = None
    st.session_state.interpretation = None
    st.session_state.message = None


st.set_page_config(
    page_title="Prediksi Rating Ulasan Pelanggan Lazada Indonesia",
)

if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "interpretation" not in st.session_state:
    st.session_state.interpretation = None
if "message" not in st.session_state:
    st.session_state.message = None

st.title("Prediksi Rating Ulasan Pelanggan Lazada Indonesia")
st.write("Masukkan teks ulasan pelanggan untuk memperoleh estimasi rating.")

review_text = st.text_area("Masukkan teks ulasan", key="review_text")

predict_button, reset_button = st.columns([1, 1])

with predict_button:
    predict_clicked = st.button("Prediksi Rating", type="primary")

with reset_button:
    st.button("Reset", on_click=reset_state)

if predict_clicked:
    st.session_state.prediction = None
    st.session_state.interpretation = None
    st.session_state.message = None

    if not review_text.strip():
        st.session_state.message = "Masukkan teks ulasan terlebih dahulu."
    else:
        cleaned_text = clean_text(review_text)
        if not cleaned_text:
            st.session_state.message = "Teks ulasan tidak berisi kata yang dapat diproses."
        else:
            model_knn, tfidf_vectorizer = load_assets()
            features = tfidf_vectorizer.transform([cleaned_text])
            prediction = float(model_knn.predict(features)[0])
            st.session_state.prediction = prediction
            st.session_state.interpretation = interpret_rating(prediction)

if st.session_state.message:
    st.warning(st.session_state.message)

if st.session_state.prediction is not None and st.session_state.interpretation:
    st.success(
        f"Prediksi rating: {format_rating(st.session_state.prediction)} dari 5"
    )
    st.write(f"Interpretasi: {st.session_state.interpretation}")
