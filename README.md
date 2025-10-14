# Sistem Prediksi Kelayakan Kredit

## Tugas Akhir 3 (TA-03): Aplikasi Web untuk Prediksi Risiko Gagal Bayar Kartu Kredit dengan Regresi Logistik

### Deskripsi Proyek

Aplikasi web ini dibangun untuk memprediksi risiko gagal bayar kartu kredit menggunakan algoritma **Regresi Logistik**. Sistem ini membantu lembaga keuangan dalam mengambil keputusan kredit yang lebih baik dengan memprediksi kemungkinan seorang nasabah akan gagal bayar pada bulan berikutnya.

### Fitur Utama

- 🎯 **Prediksi Real-time**: Prediksi instan risiko kredit dengan akurasi tinggi
- 🛡️ **Manajemen Risiko**: Membantu mengurangi kerugian akibat gagal bayar
- 👥 **User Friendly**: Interface yang mudah digunakan untuk analis kredit
- 📊 **Visualisasi Hasil**: Tampilan hasil prediksi yang informatif dan mudah dipahami

### Dataset

**UCI Credit Card Dataset**
- **Total Records**: 30,000 nasabah
- **Features**: 23 fitur (demografis, riwayat pembayaran, tagihan)
- **Target**: Default payment next month
- **Class Distribution**: 
  - Non-default: 77.88%
  - Default: 22.12%

### Performa Model

| Metrik | Nilai |
|--------|-------|
| **AUC Score** | 71.04% |
| **Accuracy** | 67% |
| **Recall** | 63% |
| **Precision** | 36% |

### Teknologi yang Digunakan

#### Machine Learning
- Python 3.10
- Scikit-Learn (Logistic Regression)
- Pandas & NumPy (Data Processing)
- Imbalanced-Learn (SMOTE)
- Matplotlib & Seaborn (Visualization)

#### Web Application
- Flask (Web Framework)
- Bootstrap 5 (Frontend)
- HTML5 & CSS3
- JavaScript (AJAX)
- Joblib (Model Persistence)

### Instalasi dan Setup

1. **Clone atau download proyek ini**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan analisis data (opsional):**
   ```bash
   python data_analysis.py
   ```

4. **Jalankan aplikasi web:**
   ```bash
   python app.py
   ```

5. **Buka browser dan akses:**
   ```
   http://localhost:5000
   ```

### Struktur Proyek

```
├── app.py                          # Aplikasi Flask utama
├── data_analysis.py                # Script analisis data dan training model
├── requirements.txt                # Dependencies Python
├── README.md                       # Dokumentasi proyek
├── UCI_Credit_Card.csv            # Dataset
├── credit_risk_model.pkl          # Model terlatih
├── credit_risk_scaler.pkl         # Scaler untuk preprocessing
├── templates/                     # Template HTML
│   ├── index.html                 # Halaman utama
│   └── about.html                 # Halaman tentang
└── static/                        # File statis
    └── style.css                  # Custom CSS
```

### Cara Menggunakan Aplikasi

1. **Akses halaman utama** di `http://localhost:5000`

2. **Isi form dengan data nasabah:**
   - **Informasi Demografis**: Limit kredit, jenis kelamin, pendidikan, status pernikahan, usia
   - **Riwayat Pembayaran**: Status pembayaran 6 bulan terakhir
   - **Jumlah Tagihan**: Tagihan 6 bulan terakhir
   - **Jumlah Pembayaran**: Pembayaran 6 bulan terakhir

3. **Klik "Prediksi Risiko Kredit"** untuk mendapatkan hasil

4. **Lihat hasil prediksi** yang mencakup:
   - Status risiko (Berisiko/Tidak Berisiko)
   - Probabilitas gagal bayar
   - Interpretasi tingkat risiko

### Interpretasi Hasil

- **Risiko Rendah** (< 30%): Nasabah dengan probabilitas gagal bayar rendah
- **Risiko Sedang** (30-60%): Nasabah dengan probabilitas gagal bayar sedang
- **Risiko Tinggi** (> 60%): Nasabah dengan probabilitas gagal bayar tinggi

### Interpretasi Bisnis

#### False Positives (31.5%)
- Nasabah yang diprediksi berisiko tapi sebenarnya tidak
- **Dampak**: Kehilangan revenue dari nasabah yang baik

#### False Negatives (37.2%)
- Nasabah yang diprediksi tidak berisiko tapi sebenarnya berisiko
- **Dampak**: Potensi kerugian dari nasabah yang gagal bayar

> **Catatan Penting**: Dalam manajemen risiko kredit, False Negatives biasanya lebih merugikan karena dapat menyebabkan kerugian finansial yang signifikan.

### Metodologi

1. **Exploratory Data Analysis (EDA)**
   - Analisis distribusi data
   - Visualisasi hubungan antar variabel
   - Identifikasi pola dan outlier

2. **Data Preprocessing**
   - Handling missing values
   - Feature scaling dengan StandardScaler
   - Encoding variabel kategorikal

3. **Class Imbalance Handling**
   - Menggunakan SMOTE (Synthetic Minority Over-sampling Technique)
   - Menyeimbangkan distribusi kelas

4. **Model Training**
   - Logistic Regression dengan Scikit-Learn
   - Train-test split (80:20)
   - Cross-validation untuk evaluasi

5. **Model Evaluation**
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - AUC-ROC Score
   - Business impact analysis

### Kontribusi

Proyek ini dikembangkan sebagai bagian dari Tugas Akhir 3 (TA-03) untuk mata kuliah Machine Learning.

### Lisensi

Proyek ini dibuat untuk tujuan edukasi dan pembelajaran.

---

**Tugas Akhir 3 (TA-03) - Sistem Prediksi Kelayakan Kredit**  
*Menggunakan Regresi Logistik untuk Prediksi Risiko Gagal Bayar Kartu Kredit*

