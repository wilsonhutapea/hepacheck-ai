# 📋 RINGKASAN INTEGRASI KNN MODEL - HEPACHECK.AI

## ✅ Status Integrasi
**SUKSES** - Model KNN berhasil diintegrasikan ke aplikasi HepaCheck.AI

---

## 🔄 Alur Kerja Aplikasi

### **Langkah 1-11: Pengumpulan Gejala Klinis**
- User menjawab 11 pertanyaan tentang gejala hepatitis
- Jawaban disimpan dalam session state

### **Langkah 12: Upload Foto Mata**
- User mengupload foto mata
- Foto disimpan untuk analisis

### **Langkah 13: Upload Foto Urine**
- User mengupload foto urine
- Foto disimpan untuk analisis

### **Langkah 14: Analisis & Hasil**

#### **🤖 Analisis Mata (3 Langkah dengan KNN Model):**

**Step 1: Deteksi Mata**
- Menggunakan: `utils/eye_detection_module_v3.py`
- Fungsi: `multi_stage_detection(image)`
- Metode: MediaPipe → Haar Cascade → Segmentation
- Output: Dictionary dengan eye_image dan eye_mask

**Step 2: Ekstraksi RGB Sclera**
- Menggunakan: `extract_sclera_rgb_from_detection()` di `app.py`
- Input: eye_image dan eye_mask dari Step 1
- Output: Tuple (R, G, B) - nilai median RGB

**Step 3: Klasifikasi KNN**
- Menggunakan: `classify_jaundice_knn()` di `app.py`
- Model: `knn_jaundice_model.pkl`
- Input: RGB values [R, G, B]
- Output: 
  - Class 0 = Normal (tidak jaundice)
  - Class 2 = Jaundice terdeteksi
  - Confidence score
  - Probabilities untuk setiap class

#### **🧪 Analisis Urine:**
- Analisis warna dan brightness urine
- Deteksi urine gelap/abnormal

#### **📊 Kesimpulan Akhir:**
- Kombinasi dari:
  - Skor gejala klinis (40% bobot)
  - Hasil KNN mata (30% bobot)
  - Hasil analisis urine (30% bobot)
- Risiko: RENDAH / SEDANG / TINGGI

---

## 📁 Struktur File

```
hepacheck-ai/
├── app.py                              # Main application (UPDATED)
├── knn_jaundice_model.pkl              # KNN Model
├── requirements.txt                    # Dependencies (UPDATED)
├── utils/
│   ├── __init__.py
│   ├── eye_detection_module_v3.py      # Eye detection (USED)
│   ├── extract_sclera_rgb.py           # Standalone RGB extractor
│   └── hepatitis_analysis.py           # Old analysis (NOT USED in new version)
└── assets/
    └── images/
```

---

## 🔧 Fungsi-Fungsi Utama

### **app.py**

#### `load_knn_model()`
```python
@st.cache_resource
def load_knn_model():
    """Load KNN model from pkl file"""
```
- Menggunakan `@st.cache_resource` untuk efisiensi
- Load model hanya sekali saat startup

#### `extract_sclera_rgb_from_detection(eye_image, eye_mask)`
```python
def extract_sclera_rgb_from_detection(eye_image, eye_mask):
    """Extract median RGB from detected sclera region"""
    # Returns: (R, G, B) tuple
```
- Input: Eye region (BGR) dan binary mask
- Konversi BGR → RGB
- Ekstrak pixels dari area sclera (mask > 0)
- Hitung median RGB
- Return: Tuple (R, G, B)

#### `classify_jaundice_knn(rgb_values, knn_model)`
```python
def classify_jaundice_knn(rgb_values, knn_model):
    """Classify jaundice using KNN model"""
    # Returns: dict with prediction results
```
- Input: RGB values dan KNN model
- Prepare input array: [[R, G, B]]
- Prediksi dengan model
- Return dict dengan:
  - `prediction`: 0 atau 2
  - `is_jaundice`: Boolean
  - `status`: String status
  - `confidence`: Percentage
  - `probabilities`: Array probabilitas
  - `classes`: Array kelas

---

## 📦 Dependencies (requirements.txt)

### **Core Libraries:**
- `streamlit==1.28.0` - Web app framework
- `opencv-python==4.8.1.78` - Computer vision
- `numpy==1.24.3` - Numerical operations

### **Eye Detection:**
- `mediapipe==0.10.8` - Face mesh detection

### **Machine Learning:**
- `scikit-learn==1.5.1` - KNN model (ADDED)
- `pandas==2.0.3` - Data handling (ADDED)

### **Utilities:**
- `pillow==10.1.0` - Image handling
- `matplotlib==3.8.0` - Plotting (optional)

---

## 🔬 Detail Model KNN

**File:** `knn_jaundice_model.pkl`

### Spesifikasi:
- **Algorithm:** K-Nearest Neighbors
- **Parameter k:** 3 neighbors
- **Library:** scikit-learn v1.5.1
- **Metric:** Euclidean Distance
- **Weights:** Uniform

### Input:
- **Format:** Array [[R, G, B]]
- **R:** Red value (0-255)
- **G:** Green value (0-255)
- **B:** Blue value (0-255)

### Output:
- **Class 0:** Normal (tidak jaundice)
- **Class 2:** Jaundice terdeteksi

### Training Data:
- **Total samples:** 40
- **Class 0 (Normal):** 20 samples
- **Class 2 (Jaundice):** 20 samples
- **RGB Range:**
  - Red: ~133-219
  - Green: ~98-184
  - Blue: ~48-167

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi:
```bash
streamlit run app.py
```

### 3. Akses Browser:
```
http://localhost:8501
```

---

## ✨ Fitur Baru

### ✅ **Integrasi KNN Model:**
- Model machine learning untuk klasifikasi jaundice
- Akurasi tinggi berdasarkan warna sclera
- Probabilitas prediksi untuk setiap class

### ✅ **Proses 3 Langkah yang Jelas:**
- Step 1: Deteksi mata otomatis
- Step 2: Ekstraksi warna RGB
- Step 3: Klasifikasi dengan KNN

### ✅ **Visualisasi Lengkap:**
- Tampilan mata yang terdeteksi
- Area sclera yang dianalisis
- Detail RGB dan probabilitas
- Confidence score

---

## 📊 Output Aplikasi

### **Untuk Setiap Mata:**

1. **Status KNN:**
   - ✅ Normal, atau
   - ⚠️ Jaundice Terdeteksi

2. **Confidence:** 
   - Persentase kepercayaan prediksi

3. **Prediksi Kelas:**
   - Class 0 (Normal) atau
   - Class 2 (Jaundice)

4. **Detail Warna Sclera:**
   - RGB (Median): R, G, B values
   - Probabilitas untuk setiap class
   - Model info

---

## 🔗 Koneksi Antar Komponen

```
┌─────────────────────────────────────────────────┐
│               USER UPLOADS PHOTO                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: Eye Detection                          │
│  Module: eye_detection_module_v3.py             │
│  Function: multi_stage_detection()              │
│  Output: eye_image + eye_mask                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: RGB Extraction                         │
│  Function: extract_sclera_rgb_from_detection()  │
│  Output: (R, G, B) tuple                        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: KNN Classification                     │
│  Model: knn_jaundice_model.pkl                  │
│  Function: classify_jaundice_knn()              │
│  Output: Class (0/2) + Confidence               │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         DISPLAY RESULTS TO USER                 │
│  - Status (Normal/Jaundice)                     │
│  - Confidence score                             │
│  - RGB values                                   │
│  - Probabilities                                │
└─────────────────────────────────────────────────┘
```

---

## 🧪 Testing

### **Manual Testing Steps:**

1. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

2. **Jawab 11 pertanyaan**

3. **Upload foto mata:**
   - Gunakan foto dari `assets/images/fadey contoh_compressed.jpg`

4. **Upload foto urine:**
   - Gunakan foto dari `assets/images/Urine Normal.jpg`

5. **Verifikasi output:**
   - ✅ Mata terdeteksi
   - ✅ RGB values ditampilkan
   - ✅ KNN prediction muncul
   - ✅ Confidence score muncul
   - ✅ Probabilitas untuk setiap class muncul

---

## ⚠️ Troubleshooting

### **Error: Model tidak ditemukan**
- Pastikan file `knn_jaundice_model.pkl` ada di root directory
- Path: `hepacheck-ai/knn_jaundice_model.pkl`

### **Error: Import module failed**
- Install semua dependencies: `pip install -r requirements.txt`
- Pastikan virtual environment aktif

### **Error: Mata tidak terdeteksi**
- Gunakan foto dengan pencahayaan baik
- Pastikan wajah/mata terlihat jelas
- Coba foto yang berbeda

---

## 📝 Catatan Penting

1. **Model KNN sudah terintegrasi penuh** dengan aplikasi
2. **Semua komponen terhubung** dan berfungsi sebagai satu kesatuan
3. **File `hepatitis_analysis.py` tidak digunakan** dalam versi baru ini
4. **Fungsi ekstraksi RGB custom** dibuat khusus untuk integrasi dengan app
5. **File `extract_sclera_rgb.py`** tetap ada untuk keperluan standalone/training

---

## 👥 Developer Notes

**Created by:** Fadey Ezra & Diandra Almeira  
**Updated:** Integration with KNN Model  
**Version:** 2.0 (KNN Integration)

---

## 📄 License & Disclaimer

⚠️ **DISCLAIMER:** Hasil analisis BUKAN diagnosis medis final. Hanya sebagai alat bantu skrining awal. Wajib konsultasi dengan dokter untuk diagnosis yang akurat.

---

**🎉 INTEGRASI SELESAI - SEMUA KOMPONEN TERHUBUNG DAN SIAP DIGUNAKAN! 🎉**

