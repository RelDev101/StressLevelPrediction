# -*- coding: utf-8 -*-
"""
train_models.py

Script ini akan:
1. Memuat data yang sudah diproses (X_scaled, y) dari preprocess.py.
2. Membagi data menjadi train/test.
3. Mendefinisikan dan melatih tiga model: KNN, SVM, Decision Tree.
4. Mengevaluasi performa menggunakan cross‐validation dan test set.
5. Mencetak metrik akurasi, classification report, dan confusion matrix untuk masing‐masing model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import load_and_preprocess_data

def main():
    # 1. Load & preprocess data
    #    Pastikan path ke CSV benar (jika CSV berada di satu level di atas folder code/,
    #    gunakan "../SaYoPillow.csv"; atau jika CSV ada di root proyek, cukup "SaYoPillow.csv").
    csv_path = "../data/SaYoPillow.csv"
    X_scaled, y = load_and_preprocess_data(csv_path)

    # 2. Bagi data menjadi training dan testing (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Definisikan model
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=None, random_state=42)
    }

    # 4. Latih & Evaluasi setiap model
    for name, model in models.items():
        print(f"\n====== Model: {name} ======\n")

        # 4a. Cross‐Validation (5‐fold) pada data training
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross‐Validation (5‐fold) Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

        # 4b. Latih model pada seluruh data training
        model.fit(X_train, y_train)

        # 4c. Evaluasi pada test set
        y_pred = model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        print(f"Test Set Accuracy: {acc_test:.4f}\n")

        # 4d. Tampilkan classification report (precision, recall, f1‐score per kelas)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # 4e. Tampilkan confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("-" * 40)

if __name__ == "__main__":
    main()
