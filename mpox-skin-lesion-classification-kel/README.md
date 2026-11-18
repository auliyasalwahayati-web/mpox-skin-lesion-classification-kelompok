# mpox-skin-lesion-classification-kelompok 6

## EfficientNet-B0

Implementasi model EfficientNet-B0 dapat dilihat pada file berikut:

[EfficientNetB0.py](Models/EfficientNetB0.py)


EfficientNet-B0 digunakan untuk klasifikasi enam kelas lesi kulit, yaitu:

- Chickenpox
- Cowpox
- HFMD
- Healthy
- Measles
- Monkeypox

Model dilatih menggunakan 5-fold cross-validation dengan konfigurasi:

- Input image size: 224 × 224 piksel
- Batch size: 16
- Maksimum epoch: 20
- Optimizer: Adam
- Learning rate: 0.0001
- Early stopping patience: 5
- Pretrained weight: ImageNet

### Hasil Evaluasi 5-Fold

| Metrik | Rata-rata ± Simpangan Baku |
|---|---:|
| Accuracy | 79.99% ± 3.91% |
| Precision | 81.34% ± 5.26% |
| Recall | 79.53% ± 3.85% |
| F1-score | 78.99% ± 3.20% |
| ROC-AUC | 0.9653 ± 0.0179 |

Hasil tersebut diperoleh dari pengujian lima fold menggunakan file
`test.py`. Model terbaik setiap fold disimpan berdasarkan nilai validation loss
terendah.
