# -*- coding: utf-8 -*-
"""
preprocess.py

Fungsi ini akan:
1. Muat file CSV SaYoPillow.
2. Rename kolom (agar nama konsisten, jika belum di-rename).
3. Pisahkan fitur (X) dan label (y).
4. Isi (impute) missing values dengan mean (jika ada).
5. Standarisasi fitur menggunakan StandardScaler.
6. Kembalikan X_scaled dan y untuk dipakai di train_models.py.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    """
    Memuat CSV, melakukan rename kolom, impute missing, dan standarisasi.

    Parameter:
    - csv_path: string, path lengkap ke file SaYoPillow.csv.

    Return:
    - X_scaled: array NumPy (n_samples, n_features) yang sudah diskalakan.
    - y: pandas Series (n_samples,) berisi label 'stress levels'.
    """
    # 1. Muat dataset
    df = pd.read_csv(csv_path)

    # 2. Rename kolom (jika belum di-rename di eda.ipynb)
    df.rename(columns={
        'sr': 'snoring rate',
        'rr': 'respiration rate',
        't': 'body temperature',
        'lm': 'limb movement',
        'bo': 'blood oxygen',
        'rem': 'eye movement rate',
        'sr.1': 'sleeping hours',
        'hr': 'heart rate',
        'sl': 'stress levels'
    }, inplace=True)

    # 3. Pisahkan fitur (X) dan label (y)
    #    Kita gunakan semua kolom kecuali 'stress levels' sebagai fitur
    fitur = [
        'snoring rate',
        'respiration rate',
        'body temperature',
        'limb movement',
        'blood oxygen',
        'eye movement rate',
        'sleeping hours',
        'heart rate'
    ]
    target = 'stress levels'

    X = df[fitur].copy()
    y = df[target].copy()

    # 4. Tangani missing values (jika ada)
    #    Isi nilai kosong dengan nilai rata-rata setiap kolom
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    # 5. Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Kembalikan X_scaled dan y
    return X_scaled, y
