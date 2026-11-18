# Klasifikasi Lesi Kulit Mpox - Kelompok 6

Project klasifikasi citra lesi kulit mpox menggunakan deep learning (PyTorch). Model dilatih dan diuji dengan **5-fold cross validation** untuk mengenali 6 kelas kondisi kulit, lalu dibandingkan antar-model.

> Project ini dibuat untuk keperluan pembelajaran dan **bukan alat diagnosis medis**. Hasil prediksi tidak menggantikan pemeriksaan oleh tenaga kesehatan.

## Fitur

- Training dengan 5-fold cross validation, dengan model baru untuk setiap fold.
- Penyimpanan model terbaik per fold berdasarkan *validation loss* terendah.
- Early stopping (opsional, diatur per model di `config.py`).
- Evaluasi: accuracy, precision, recall, F1-score (macro), dan ROC-AUC (one-vs-rest, macro).
- Confusion matrix per fold dan gabungan semua fold, serta laporan klasifikasi per kelas.
- Penambahan model baru cukup lewat satu entri di `config.py`.

## Struktur Project

```
Kelompok-6/
├── Models/              # definisi arsitektur model (CNN.py, CNN2.py, ...)
├── Utils/
│   └── getData.py       # pemuat dataset (getImageLabel, get_dataset_root, CLASSES)
├── Checkpoints/         # model terbaik per fold (dibuat otomatis saat training)
├── Results/             # grafik, riwayat, dan metrik (dibuat otomatis)
├── config.py            # daftar model dan pengaturannya
├── train.py             # script training
├── test.py              # script pengujian
├── requirements.txt     # daftar library
└── README.md
```

## Dataset

Dataset diunduh otomatis lewat `kagglehub` oleh fungsi `get_dataset_root()` di `Utils/getData.py`. Data dibagi per fold menjadi `Train`, `Valid`, dan `Test`.

Daftar 6 kelas dapat dilihat pada variabel `CLASSES` di `Utils/getData.py`.

## Persiapan Environment

### 1. Clone repo

```bash
git clone https://github.com/auliyasalwahayati-web/Kelompok-6.git
cd Kelompok-6
```

### 2. Buat virtual environment (disarankan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install library umum

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch

PyTorch (`torch` dan `torchvision`) tidak ada di `requirements.txt` karena cara instalnya berbeda untuk GPU NVIDIA dan CPU. Pilih salah satu:

**Laptop dengan GPU NVIDIA (CUDA):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Laptop tanpa GPU (CPU saja):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Perintah untuk versi CUDA lain tersedia di [pytorch.org](https://pytorch.org/get-started/locally/).

Cek apakah GPU terbaca:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Model

Semua model didaftarkan di `config.py`.

| Nama | Arsitektur | Ukuran gambar | Batch size | Epoch | Learning rate | Early stopping |
|------|-----------|---------------|-----------|-------|---------------|----------------|
| `model1` | `SimpleCNN` | 100 | 32 | 100 | 0.001 | tidak dipakai |
| `model2` | `SimpleCNNReg` | 244 | 16 | 15 | 0.001 | patience 7 |

Optimizer yang dipakai adalah Adam dengan loss `NLLLoss`, karena model mengeluarkan log-probabilitas (`LogSoftmax`) untuk 6 kelas.

### Menambah model baru

1. Buat file model di folder `Models/`. Model harus menerima input `(B, 3, H, W)` dan mengeluarkan log-probabilitas 6 kelas.
2. Impor model di `config.py`.
3. Tambahkan satu entri baru di `CONFIGS`:

```python
"model3": {
    "model": NamaModel,
    "model_args": {"output": 6},
    "img_size": 128,
    "batch_size": 32,
    "epoch": 50,
    "lr": 0.001,
    "patience": 5,
},
```

## Cara Menjalankan

### Training

```bash
python train.py --model model1
```

Opsi:

| Argumen | Fungsi | Default |
|---------|--------|---------|
| `--model` | nama model di `config.py` (wajib) | - |
| `--folds` | jumlah fold yang dilatih, mulai dari fold 1 | 5 |
| `--epochs` | mengganti jumlah epoch di `config.py` | sesuai config |

Contoh uji coba cepat (1 fold, 2 epoch):

```bash
python train.py --model model1 --folds 1 --epochs 2
```

Hasil training:

- `Checkpoints/<model>/<model>_fold<N>.pt`: model terbaik per fold
- `Results/<model>/training_fold<N>.png`: grafik loss training dan validasi
- `Results/<model>/history_fold<N>.csv`: riwayat loss dan akurasi per epoch
- `Results/<model>/training_summary.csv`: ringkasan epoch terbaik dan durasi tiap fold

### Testing

Jalankan setelah training selesai:

```bash
python test.py --model model1
```

Opsi `--folds` bekerja sama seperti pada training, misalnya `python test.py --model model1 --folds 1`.

Hasil testing di `Results/<model>/`:

- `test_metrics.csv`: accuracy, precision, recall, F1, dan ROC-AUC per fold, beserta rata-rata dan simpangan baku
- `confusion_matrix_fold<N>.png`: confusion matrix per fold
- `confusion_matrix_all.png`: confusion matrix gabungan semua fold
- `classification_report_fold<N>.txt`: laporan per kelas

## Metrik Evaluasi

Precision, recall, dan F1-score memakai rata-rata **macro** karena jumlah data antar kelas tidak seimbang. ROC-AUC dihitung dengan skema one-vs-rest dari probabilitas prediksi. Confusion matrix dinormalisasi per baris, sehingga setiap baris menunjukkan proporsi prediksi untuk satu kelas asli.

## Hasil

Isi tabel ini dari `Results/<model>/test_metrics.csv` setelah testing selesai.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| `model1` (SimpleCNN) | - | - | - | - | - |
| `model2` (SimpleCNNReg) | - | - | - | - | - |

## Anggota Kelompok 6

| Nama | NIM | Peran |
|------|-----|-------|
| Auliya Salwa Hayati | - | - |
| - | - | - |
