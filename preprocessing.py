# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path="data.csv", sequence_length=60):
    df = pd.read_csv(file_path)
    df = df.ffill()  # Fill missing values

    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    x = np.array(x)
    y = np.array(y)
    return x, y, scaler
