# mpox-skin-lesion-classification-kelompok 6

## EfficientNet-B0

Model EfficientNet-B0 digunakan untuk melakukan klasifikasi enam kelas lesi kulit:

- Chickenpox
- Cowpox
- HFMD
- Healthy
- Measles
- Monkeypox

Implementasi model dapat dilihat pada file:

[efficientnet_b0.py](Models/efficientnet_b0.py)

### Konfigurasi Training

- Ukuran input: 224 × 224 piksel
- Batch size: 16
- Maksimum epoch: 20
- Learning rate: 0.0001
- Optimizer: Adam
- Early stopping patience: 5
- Pretrained weight: ImageNet
- Metode evaluasi: 5-fold cross-validation

### Hasil Evaluasi EfficientNet-B0

| Metrik | Rata-rata ± Simpangan Baku |
|---|---:|
| Accuracy | 79.99% ± 3.91% |
| Precision | 81.34% ± 5.26% |
| Recall | 79.53% ± 3.85% |
| F1-score | 78.99% ± 3.20% |
| ROC-AUC | 0.9653 ± 0.0179 |

Hasil pengujian diperoleh dari lima fold menggunakan data test. Model terbaik pada setiap fold dipilih berdasarkan validation loss terendah.
