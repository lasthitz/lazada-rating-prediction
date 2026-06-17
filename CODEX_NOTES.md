Kerjakan project Streamlit yang berada di folder lokal `TUBES`.

Sebelum membuat atau mengubah file apa pun, periksa isi folder project dan laporkan kondisi awalnya secara singkat.

## KONDISI PROJECT AKTUAL

Struktur awal yang seharusnya tersedia:

```text
TUBES/
├── model/
│   ├── model_knn_regressor.joblib
│   ├── model_decision_tree_regressor.joblib
│   ├── model_svm_regressor.joblib
│   ├── model_nn_regressor.joblib
│   └── tfidf_vectorizer.joblib
└── lazada_reviews_clean.csv
└──CODEX_NOTES.md
```

Pastikan kelima file `.joblib` dan satu file CSV tersedia.

Periksa dan laporkan:

1. nama file;
2. path relatif;
3. ukuran file;
4. tipe objek hasil `joblib.load()` untuk kelima file `.joblib`.

Struktur file aktual bukan bundle dictionary:

* `model_knn_regressor.joblib` berisi objek `KNeighborsRegressor` yang sudah fit;
* `model_decision_tree_regressor.joblib` berisi objek `DecisionTreeRegressor` yang sudah fit;
* `model_svm_regressor.joblib` berisi objek `SVR` yang sudah fit;
* `model_nn_regressor.joblib` berisi objek `MLPRegressor` yang sudah fit;
* `tfidf_vectorizer.joblib` berisi objek `TfidfVectorizer` yang sudah fit.

Jangan mengubah, membungkus ulang, menimpa, memindahkan, mengganti nama, atau menyimpan ulang file `.joblib`.

## MODEL YANG DIGUNAKAN APLIKASI

Aplikasi hanya boleh menggunakan:

```text
model/model_knn_regressor.joblib
model/tfidf_vectorizer.joblib
```

KNN Regressor adalah model terbaik berdasarkan evaluasi Google Colab:

```text
Versi final : HPO
MAE          : 0.132118
MSE          : 0.249679
RMSE         : 0.499679
R²           : 0.796911
```

Parameter final model:

```python
{
    "algorithm": "brute",
    "metric": "euclidean",
    "n_neighbors": 15,
    "weights": "distance"
}
```

Tiga model lain tetap harus berada di folder `model`, tetapi tidak digunakan oleh aplikasi:

```text
model_decision_tree_regressor.joblib
model_svm_regressor.joblib
model_nn_regressor.joblib
```

Jangan membuat dropdown pemilihan model, empat hasil prediksi, voting, ensemble, atau rata-rata hasil model.

## LINGKUNGAN GOOGLE COLAB

Model dibuat menggunakan:

```text
Python         3.12.13
pandas         2.2.2
numpy          2.0.2
scipy          1.16.3
scikit-learn   1.6.1
joblib         1.5.3
matplotlib     3.10.0
```

Gunakan Python 3.12 untuk virtual environment apabila tersedia.

Jika Python 3.12 tidak tersedia, jangan menggunakan versi lain secara diam-diam. Laporkan versi Python yang tersedia dan jelaskan kendalanya.

## TUGAS YANG HARUS DIKERJAKAN

1. Buat virtual environment bernama `venv` di root project.
2. Aktifkan `venv`.
3. Instal dependency yang diperlukan ke dalam `venv`.
4. Gunakan versi berikut untuk library machine learning:

```text
pandas==2.2.2
numpy==2.0.2
scipy==1.16.3
scikit-learn==1.6.1
joblib==1.5.3
```

5. Instal Streamlit yang kompatibel dengan Python 3.12.
6. Setelah berhasil, pin versi Streamlit yang benar-benar terinstal ke `requirements.txt`.
7. Jangan memasukkan `matplotlib` apabila aplikasi tidak mengimpornya.
8. Buat file:

```text
app_streaming.py
requirements.txt
README.md
.gitignore
```

9. Jangan membuat folder `src`, folder `tests`, database, API, FastAPI, login, upload CSV, atau struktur aplikasi tambahan.
10. Folder `venv` wajib masuk `.gitignore`.
11. File `lazada_reviews_clean.csv` tetap berada di project sebagai dokumentasi, tetapi aplikasi tidak boleh membacanya ketika melakukan prediksi.

## STRUKTUR AKHIR

Struktur project akhir harus sederhana:

```text
TUBES/
├── model/
│   ├── model_knn_regressor.joblib
│   ├── model_decision_tree_regressor.joblib
│   ├── model_svm_regressor.joblib
│   ├── model_nn_regressor.joblib
│   └── tfidf_vectorizer.joblib
├── venv/
├── .gitignore
├── app_streaming.py
├── README.md
├── requirements.txt
└── lazada_reviews_clean.csv
```

## APLIKASI STREAMLIT

Buat aplikasi satu halaman sederhana.

Judul:

```text
Prediksi Rating Ulasan Pelanggan Lazada Indonesia
```

Deskripsi:

```text
Masukkan teks ulasan pelanggan untuk memperoleh estimasi rating.
```

Komponen wajib:

1. text area dengan label `Masukkan teks ulasan`;
2. tombol `Prediksi Rating`;
3. tombol `Reset`;
4. satu hasil prediksi rating;
5. satu interpretasi hasil;
6. validasi input mentah kosong;
7. validasi teks yang menjadi kosong setelah cleaning.

Contoh tampilan hasil:

```text
Prediksi rating: 4,35 dari 5
Interpretasi: Positif
```

Gunakan maksimal dua angka desimal dan format desimal Indonesia dengan koma pada tampilan.

Jangan menampilkan:

* nama model;
* nama file model;
* parameter model;
* MAE;
* MSE;
* RMSE;
* R²;
* informasi TF-IDF;
* probabilitas;
* confidence;
* detail teknis machine learning.

Tidak boleh ada sidebar navigasi, banyak halaman, dashboard evaluasi, atau pilihan model.

## FUNGSI CLEANING WAJIB

Gunakan fungsi cleaning yang sama persis dengan Google Colab:

```python
import re

def clean_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

Jangan menambahkan:

* stemming;
* stopword removal;
* kamus slang;
* koreksi typo;
* normalisasi huruf berulang;
* terjemahan;
* preprocessing NLP tambahan.

## ALUR PREDIKSI

Urutannya wajib:

```text
input teks
→ validasi input mentah
→ clean_text()
→ validasi hasil cleaning
→ tfidf_vectorizer.transform()
→ model_knn.predict()
→ format maksimal dua desimal
→ interpretasi
→ tampilkan hasil
```

Gunakan hanya:

```python
tfidf_vectorizer.transform(...)
```

Dilarang menggunakan:

```python
fit()
fit_transform()
TfidfVectorizer()
```

Jangan melakukan training ulang, mengubah vocabulary, mengubah parameter model, atau memodifikasi file `.joblib`.

Gunakan path yang aman dan relatif terhadap lokasi `app_streaming.py`, misalnya menggunakan `pathlib.Path`, agar aplikasi dapat berjalan secara lokal dan di Streamlit Community Cloud.

Gunakan cache Streamlit yang sesuai untuk memuat model dan TF-IDF satu kali.

## INTERPRETASI RATING

Gunakan aturan:

```text
prediksi < 2.5              → Negatif
2.5 <= prediksi < 3.5       → Netral
prediksi >= 3.5             → Positif
```

Jangan melakukan clipping atau pembulatan sebelum menentukan interpretasi. Pembulatan hanya untuk tampilan.

## TOMBOL RESET

Tombol `Reset` harus:

1. mengosongkan text area;
2. menghapus hasil prediksi;
3. menghapus interpretasi;
4. melakukan rerun dengan aman.

Gunakan `st.session_state` apabila diperlukan.

## PENGUJIAN

Lakukan syntax check:

```bash
python -m py_compile app_streaming.py
```

Lakukan pengujian load file dan prediksi tanpa training ulang menggunakan tiga teks berikut:

```text
Barang sangat bagus, kualitas mantap dan sesuai pesanan!
Produknya lumayan dan sesuai dengan harga.
Barang rusak, pengiriman lama dan sangat mengecewakan.
```

Referensi hasil KNN dari Google Colab:

```text
Teks positif : sekitar 5.00
Teks biasa   : sekitar 4.94
Teks negatif : sekitar 2.89
```

Gunakan toleransi kecil untuk verifikasi, tetapi jangan meng-hardcode nilai prediksi.

Pastikan juga aplikasi menangani:

```text
input kosong
input hanya spasi
input hanya simbol seperti !!!
```

Jalankan aplikasi lokal menggunakan Python dari `venv`.

Contoh perintah Windows yang dapat digunakan:

```powershell
.\venv\Scripts\python.exe -m streamlit run app_streaming.py
```

Tunggu sampai Streamlit memberikan URL lokal dan laporkan URL tersebut.

Jangan menyatakan aplikasi lokal berhasil sebelum:

1. seluruh file berhasil di-load;
2. syntax check berhasil;
3. tiga teks pengujian berhasil diprediksi;
4. validasi input kosong berhasil;
5. halaman Streamlit berhasil dibuka.

## README

Buat `README.md` yang berisi:

1. judul project;
2. deskripsi singkat;
3. tujuan aplikasi;
4. struktur folder;
5. versi Python;
6. cara membuat dan mengaktifkan `venv`;
7. cara instal dependency;
8. cara menjalankan Streamlit;
9. penjelasan singkat bahwa aplikasi hanya melakukan inference;
10. keterangan bahwa `lazada_reviews_clean.csv` tidak dibaca saat prediksi;
11. instruksi deployment ke Streamlit Community Cloud secara singkat.

Jangan menampilkan metrik model di halaman aplikasi, tetapi metrik boleh disebut secara singkat di README sebagai dokumentasi project.

## .GITIGNORE

Minimal harus mengecualikan:

```text
venv/
.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.DS_Store
```

Jangan mengecualikan:

```text
model/
*.joblib
lazada_reviews_clean.csv
requirements.txt
app_streaming.py
README.md
```

GitHub dan deployment

Kerjakan tahap lokal terlebih dahulu sampai aplikasi Streamlit benar-benar berjalan dan bisa diuji.

Urutan wajib:

Periksa isi folder TUBES dan baca CODEX_NOTES.md.
Buat venv, instal dependency, dan buat file aplikasi.
Jalankan syntax check.
Uji load model dan TF-IDF.
Uji prediksi dengan contoh teks.
Jalankan Streamlit secara lokal.
Pastikan URL lokal dapat dibuka dan fitur prediksi serta reset berfungsi.

Untuk saat ini jangan:

membuat atau menghubungkan repository GitHub;
melakukan commit atau push;
melakukan deployment Streamlit Community Cloud;
mengubah file .joblib;
mengubah file di luar folder TUBES.

Setelah aplikasi lokal berhasil, berhenti dan laporkan:

file yang dibuat;
dependency yang terinstal;
hasil pengujian;
URL lokal;
error atau warning yang muncul.

Tunggu instruksi saya sebelum lanjut ke GitHub dan deployment publik.

Sedikit INFO  tentang github dan deployment Setelah aplikasi lokal berhasil:

pastikan venv tidak ikut Git;
commit source code, dataset bersih, TF-IDF, dan empat file model;
push ke GitHub apabila remote dan autentikasi tersedia;
jika repository belum tersedia, minta URL repository kepada pengguna;
deploy melalui Streamlit Community Cloud;
gunakan branch utama dan entrypoint app_streaming.py;
gunakan Python 3.12;
periksa build log jika gagal;
perbaiki dependency atau path apabila diperlukan;
uji URL publik menggunakan beberapa teks.

Jangan menyatakan deployment berhasil sebelum URL publik benar-benar dapat dibuka dan digunakan untuk prediksi.

Keempat file model, TF-IDF, dan dataset bersih harus tetap dapat masuk repository.

## LARANGAN

Jangan membuat:

* FastAPI;
* API;
* database;
* login;
* upload CSV;
* model tambahan;
* training ulang;
* dropdown model;
* empat output prediksi;
* ensemble;
* voting;
* probabilitas palsu;
* confidence palsu;
* dashboard evaluasi;
* banyak halaman;
* folder `src`;
* folder `tests`;
* Docker;
* file model baru.

Jangan mengubah lima file `.joblib`.

## LAPORAN AKHIR CODEX

Setelah selesai, laporkan:

1. kondisi awal folder;
2. tipe dan ukuran setiap file `.joblib`;
3. versi Python yang dipakai;
4. versi dependency yang terinstal;
5. daftar file yang dibuat;
6. file yang diubah;
7. hasil syntax check;
8. hasil tiga prediksi pengujian;
9. hasil validasi input kosong dan hasil cleaning kosong;
10. URL lokal Streamlit;
11. isi akhir struktur folder;
12. konfirmasi bahwa file model tidak diubah;
13. konfirmasi bahwa aplikasi hanya menggunakan KNN dan TF-IDF;
14. kendala atau warning yang muncul.

Jangan melakukan deployment publik atau push GitHub pada tahap ini. Berhenti setelah aplikasi lokal berhasil dan tunggu instruksi berikutnya.
