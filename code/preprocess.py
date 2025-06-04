import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Rename kolom (kalau belum dilakukan di eda)
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

    # Fitur dan target
    X = df.drop('stress levels', axis=1)
    y = df['stress levels']

    # Normalisasi (standarisasi)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
