# ASL Detection With NLP sebagai Rekomendasi Kata

## 1. Pendahuluan

ASL Detection With NLP adalah sebuah sistem yang dirancang untuk mendeteksi bahasa isyarat Amerika (American Sign Language/ASL) menggunakan teknologi computer vision dan memberikan rekomendasi kata secara otomatis dengan bantuan Natural Language Processing (NLP). Sistem ini bertujuan untuk membantu individu tunarungu maupun masyarakat umum dalam berkomunikasi lebih efektif melalui pemahaman dan penerjemahan gestur ASL.

## 2. Latar Belakang

Bahasa isyarat merupakan media komunikasi utama bagi penyandang tunarungu. Namun, tidak semua orang mampu memahami bahasa isyarat, sehingga diperlukan teknologi yang dapat menjadi jembatan komunikasi. Dengan kemajuan artificial intelligence, khususnya pada bidang computer vision dan NLP, deteksi otomatis gestur ASL dan rekomendasi kata yang relevan dapat diwujudkan sebagai solusi inovatif.

## 3. Tujuan

- Membangun sistem pendeteksi gestur ASL berbasis kamera.
- Mengintegrasikan NLP untuk memberikan rekomendasi kata/kalimat dari hasil deteksi gestur.
- Mempermudah interaksi antara penyandang tunarungu dan masyarakat luas.

## 4. Metodologi

1. **Akuisisi Data**  
   Mengumpulkan dataset gambar atau video gestur ASL.
2. **Pra-pemrosesan**  
   Melakukan preprocessing data seperti resize, augmentasi, dan normalisasi.
3. **Pelatihan Model Deteksi**  
   Menggunakan deep learning (CNN) untuk mendeteksi gestur ASL.
4. **Integrasi NLP**  
   Menggunakan model NLP untuk memberikan rekomendasi kata dari hasil deteksi.
5. **Evaluasi Sistem**  
   Mengukur akurasi deteksi dan relevansi rekomendasi kata.

## 5. Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/KangFeeder090403/ASL-Detection-With-NLP.git
   cd ASL-Detection-With-NLP
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 6. Cara Penggunaan

1. Jalankan program utama (`finalpred.py` atau notebook terkait).
2. Sistem akan mengakses kamera untuk mendeteksi gestur ASL.
3. Lakukan gestur ASL di depan kamera.
4. Hasil deteksi serta rekomendasi kata akan muncul di layar.

## 7. Teknologi & Library

- Python 3.9.21
- OpenCV (pengolahan citra & video)
- TensorFlow/Keras atau PyTorch (deep learning)
- NLTK, spaCy, atau HuggingFace Transformers (NLP)
- Streamlit/Gradio (opsional, untuk antarmuka)

## 8. Kontribusi

Kontribusi sangat terbuka! Silakan buat *issue* atau *pull request* untuk penambahan fitur, saran, maupun perbaikan bug.

## 9. Lisensi

MIT License.

## 10. Referensi

- [ASL Alphabet Dataset - https://drive.google.com/drive/folders/1CtSOwl10TNaQ3L9a46m_WsDaBMr1Os3y?hl=ID]
