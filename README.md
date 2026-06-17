# Prediksi Rating Ulasan Pelanggan Lazada Indonesia

Aplikasi Streamlit sederhana untuk memperkirakan rating dari teks ulasan pelanggan Lazada Indonesia.

## Tujuan Aplikasi

Project ini dibuat untuk menjalankan inference dari model machine learning yang sudah dilatih sebelumnya. Pengguna memasukkan teks ulasan, lalu aplikasi menampilkan estimasi rating dan interpretasi sederhana.

## Struktur Folder

```text
TUBES/
|-- model/
|   |-- model_knn_regressor.joblib
|   |-- model_decision_tree_regressor.joblib
|   |-- model_svm_regressor.joblib
|   |-- model_nn_regressor.joblib
|   `-- tfidf_vectorizer.joblib
|-- venv/
|-- .gitignore
|-- app_streaming.py
|-- README.md
|-- requirements.txt
|-- lazada_reviews_clean.csv
`-- CODEX_NOTES.md
```

## Versi Python

Gunakan Python 3.12. Project ini diuji dengan Python 3.12.13.

## Membuat dan Mengaktifkan Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Jika perintah `python` belum tersedia di PATH, gunakan path Python 3.12 yang tersedia di komputer.

## Instal Dependency

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependency machine learning yang digunakan:

```text
pandas==2.2.2
numpy==2.0.2
scipy==1.16.3
scikit-learn==1.6.1
joblib==1.5.3
streamlit==1.58.0
```

## Menjalankan Streamlit

```powershell
.\venv\Scripts\python.exe -m streamlit run app_streaming.py
```

## Catatan Inference

Aplikasi hanya melakukan inference menggunakan model dan TF-IDF vectorizer yang sudah tersedia. Tidak ada training ulang, perubahan vocabulary, atau perubahan file model saat aplikasi berjalan.

File `lazada_reviews_clean.csv` disimpan sebagai dokumentasi dataset bersih, tetapi tidak dibaca oleh aplikasi saat melakukan prediksi.

Model utama project ini adalah KNN Regressor versi HPO dengan dokumentasi evaluasi: MAE 0.132118, MSE 0.249679, RMSE 0.499679, dan R2 0.796911.

## Deployment Streamlit Community Cloud

1. Push seluruh source code, folder `model/`, file `.joblib`, `requirements.txt`, `README.md`, `app_streaming.py`, dan `lazada_reviews_clean.csv` ke repository GitHub.
2. Pastikan folder `venv/` tidak ikut masuk repository.
3. Buka Streamlit Community Cloud dan pilih repository GitHub.
4. Pilih branch utama.
5. Gunakan `app_streaming.py` sebagai entrypoint.
6. Gunakan Python 3.12.
7. Periksa build log dan uji URL publik setelah deployment selesai.
