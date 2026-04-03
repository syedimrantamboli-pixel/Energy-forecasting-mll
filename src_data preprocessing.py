import pandas as pd
import numpy as np

def generate_data():
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=500, freq='H')

    data = pd.DataFrame({
        'datetime': dates,
        'temperature': np.random.uniform(15, 35, size=500),
        'humidity': np.random.uniform(30, 90, size=500),
        'wind_speed': np.random.uniform(0, 10, size=500),
    })

    data['energy_consumption'] = (
        50 +
        data['temperature'] * 2 +
        data['humidity'] * 0.5 +
        np.random.normal(0, 5, size=500)
    )

    return data


def preprocess_data(data):
    data['hour'] = data['datetime'].dt.hour
    data['day'] = data['datetime'].dt.day
    data['month'] = data['datetime'].dt.month

    data = data.drop(columns=['datetime'])

    return data