from pathlib import Path
import re

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_knn_regressor.joblib"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.joblib"

SIDEBAR_EXAMPLES = [
    ("Contoh Positif", "Barang sangat bagus, kualitas mantap dan sesuai pesanan."),
    ("Contoh Netral", "Produknya lumayan dan sesuai dengan harga."),
    ("Contoh Negatif", "Barang rusak, pengiriman lama dan sangat mengecewakan."),
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
    return "\u2605" * filled_stars + "\u2606" * (5 - filled_stars)


def input_quality(word_count):
    if word_count == 0:
        return "Input belum tersedia"
    if word_count <= 2:
        return "Ulasan terlalu singkat"
    if word_count <= 5:
        return "Ulasan cukup singkat"
    return "Ulasan cukup jelas"


def analyze_vocabulary(cleaned_text, vectorizer):
    words = cleaned_text.split()
    vocabulary = getattr(vectorizer, "vocabulary_", {})
    known_count = sum(1 for word in words if word in vocabulary)
    unknown_words = [word for word in words if word not in vocabulary]
    unknown_examples = list(dict.fromkeys(unknown_words))[:5]
    return known_count, len(unknown_words), unknown_examples


def truncate_text(text, limit=72):
    clean_preview = " ".join(str(text).split())
    if len(clean_preview) <= limit:
        return clean_preview
    return f"{clean_preview[: limit - 3]}..."


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


def add_prediction_history(text, prediction, interpretation):
    st.session_state.prediction_history.append(
        {
            "text": text,
            "rating": prediction,
            "interpretation": interpretation,
        }
    )


def clear_session_history():
    st.session_state.prediction_history = []


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
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

current_cleaned_text = clean_text(st.session_state.review_text)
current_cleaned_words = current_cleaned_text.split()

with st.sidebar:
    st.title("Panel Analisis Ulasan")
    st.caption("Ringkasan ini hanya membantu membaca input dan sesi saat ini.")

    st.divider()
    st.subheader("Analisis Input Saat Ini")
    st.metric("Jumlah karakter", len(st.session_state.review_text))
    st.metric("Jumlah kata setelah cleaning", len(current_cleaned_words))
    st.caption("Preview hasil cleaning")
    st.code(current_cleaned_text if current_cleaned_text else "-", language="text")
    st.caption(f"Status kualitas input: {input_quality(len(current_cleaned_words))}")

    st.divider()
    st.subheader("Keterbacaan Kata")
    _, sidebar_vectorizer = load_assets()
    known_count, unknown_count, unknown_examples = analyze_vocabulary(
        current_cleaned_text,
        sidebar_vectorizer,
    )
    word_metric_columns = st.columns(2)
    with word_metric_columns[0]:
        st.metric("Dikenali", known_count)
    with word_metric_columns[1]:
        st.metric("Tidak dikenali", unknown_count)

    with st.expander("Contoh kata tidak dikenali", expanded=unknown_count > 0):
        if unknown_examples:
            for word in unknown_examples:
                st.caption(f"- {word}")
            st.info(
                'Gunakan kata yang dipisahkan dengan spasi dan kalimat yang lebih jelas. Contoh: tulis "ga suka", bukan "gasuka".'
            )
        else:
            st.caption("Tidak ada kata tidak dikenali untuk input saat ini.")

    st.divider()
    st.subheader("Contoh Ulasan")
    for label, example in SIDEBAR_EXAMPLES:
        st.button(
            label,
            on_click=select_example,
            args=(example,),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Analisis Sesi")
    history = st.session_state.prediction_history
    total_predictions = len(history)
    average_rating = (
        sum(item["rating"] for item in history) / total_predictions
        if total_predictions
        else None
    )
    positive_count = sum(1 for item in history if item["interpretation"] == "Positif")
    neutral_count = sum(1 for item in history if item["interpretation"] == "Netral")
    negative_count = sum(1 for item in history if item["interpretation"] == "Negatif")

    st.metric("Total prediksi", total_predictions)
    st.metric(
        "Rata-rata rating",
        format_rating(average_rating) if average_rating is not None else "-",
    )

    session_metric_columns = st.columns(3)
    with session_metric_columns[0]:
        st.metric("Positif", positive_count)
    with session_metric_columns[1]:
        st.metric("Netral", neutral_count)
    with session_metric_columns[2]:
        st.metric("Negatif", negative_count)

    with st.expander("Riwayat prediksi terakhir", expanded=bool(history)):
        if history:
            for item in reversed(history[-5:]):
                st.caption(truncate_text(item["text"]))
                st.write(
                    f"{format_rating(item['rating'])} dari 5 - {item['interpretation']}"
                )
        else:
            st.caption("Belum ada prediksi pada sesi ini.")

    st.button(
        "Hapus Semua Data Sesi",
        on_click=clear_session_history,
        use_container_width=True,
    )

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
            add_prediction_history(
                review_text,
                prediction,
                st.session_state.interpretation,
            )

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
