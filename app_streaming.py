from pathlib import Path
import re

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_knn_regressor.joblib"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.joblib"

EXAMPLE_REVIEWS = [
    "Barang sangat bagus, kualitas mantap dan sesuai pesanan!",
    "Kualitas kurang bagus, barang biasa saja, pengiriman lama.",
    "Kualitas sangat buruk, barang rusak, tidak sesuai deskripsi.",
]

INTERPRETATION_STYLES = {
    "Positif": {
        "background": "#e8f7ee",
        "border": "#2e9d58",
        "text": "#14532d",
    },
    "Netral": {
        "background": "#fff6df",
        "border": "#d48a05",
        "text": "#713f12",
    },
    "Negatif": {
        "background": "#fdecec",
        "border": "#d64242",
        "text": "#7f1d1d",
    },
}


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


def rating_stars(value):
    filled_stars = int(round(value))
    filled_stars = max(0, min(5, filled_stars))
    return "★" * filled_stars + "☆" * (5 - filled_stars)


def select_example(text):
    st.session_state.review_text = text
    st.session_state.prediction = None
    st.session_state.interpretation = None
    st.session_state.message = None


def reset_state():
    st.session_state.review_text = ""
    st.session_state.prediction = None
    st.session_state.interpretation = None
    st.session_state.message = None


st.set_page_config(
    page_title="Prediksi Rating Ulasan Pelanggan Lazada Indonesia",
)

st.markdown(
    """
    <style>
    .main-title-note {
        color: #4b5563;
        font-size: 1rem;
        margin-bottom: 1.25rem;
    }
    .input-hint {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        color: #1f2937;
        padding: 0.85rem 1rem;
        border-radius: 0.4rem;
        margin: 0.75rem 0 1rem;
    }
    .result-card {
        border: 1px solid;
        border-left-width: 0.45rem;
        border-radius: 0.5rem;
        padding: 1.05rem 1.15rem;
        margin-top: 1rem;
    }
    .result-card h3 {
        margin: 0 0 0.35rem;
        font-size: 1.15rem;
    }
    .rating-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
        margin: 0.25rem 0;
    }
    .stars {
        color: #f59e0b;
        font-size: 1.55rem;
        letter-spacing: 0.08rem;
        margin-top: 0.45rem;
    }
    .footer-note {
        color: #6b7280;
        font-size: 0.9rem;
        text-align: center;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
st.markdown(
    '<p class="main-title-note">Masukkan teks ulasan pelanggan untuk memperoleh estimasi rating.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="input-hint">
        Tulis ulasan seperti komentar pembeli asli. Hindari input kosong atau hanya simbol agar teks dapat diproses.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Contoh ulasan siap coba:")
example_columns = st.columns(3)
example_labels = ["Ulasan positif", "Ulasan biasa", "Ulasan negatif"]

for column, label, example in zip(example_columns, example_labels, EXAMPLE_REVIEWS):
    with column:
        st.button(
            label,
            on_click=select_example,
            args=(example,),
            use_container_width=True,
        )

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
    style = INTERPRETATION_STYLES[st.session_state.interpretation]
    rating_display = format_rating(st.session_state.prediction)
    visual_progress = min(max(st.session_state.prediction / 5, 0.0), 1.0)

    st.markdown(
        f"""
        <div class="result-card" style="background: {style['background']}; border-color: {style['border']}; color: {style['text']};">
            <h3>Hasil prediksi</h3>
            <div class="rating-value">Prediksi rating: {rating_display} dari 5</div>
            <div>Interpretasi: <strong>{st.session_state.interpretation}</strong></div>
            <div class="stars" aria-label="Visual rating bintang">{rating_stars(st.session_state.prediction)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(visual_progress, text="Skala rating 1-5")

st.markdown(
    """
    <div class="footer-note">
        Aplikasi ini digunakan untuk prediksi rating ulasan pelanggan berdasarkan teks yang dimasukkan.
    </div>
    """,
    unsafe_allow_html=True,
)
