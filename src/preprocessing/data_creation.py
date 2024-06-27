import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import config
from src.helpers.csv_file_manager import save_csv_files
from src.helpers.decorators import preserve_custom_attribute, timer


def accessibility_score(G: float, R: float) -> float:
    N = G + R
    if N == 0:
        return 0
    return G / N * np.log(N + 1)


def shannon_enthropy(G: float, R: float) -> float:
    N = G + R
    if N == 0:
        return 0

    p_G = G / N
    p_R = R / N

    if p_G > 0:
        H_G = -p_G * np.log2(p_G)
    else:
        H_G = 0

    if p_R > 0:
        H_R = -p_R * np.log2(p_R)
    else:
        H_R = 0

    H = H_G + H_R

    score = H

    return score


@preserve_custom_attribute('filename')
def compute_accessibility_score(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    scores = [accessibility_score(row['wheelchair_yes'], row['wheelchair_no']) for _, row in df.iterrows()]
    scores = MinMaxScaler().fit_transform(np.array(scores).reshape(-1, 1))
    df_copy['accessibility_score'] = scores
    return df_copy


def craft_wheelchair_data(car_data: pd.DataFrame, accessibility_scores: List[float]) -> pd.DataFrame:

    def map(value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    wheelchair_data = car_data.copy()

    if len(car_data.columns) != len(accessibility_scores):
        raise ValueError(f"Expected {len(accessibility_scores)} columns, but got {len(car_data.columns)}")

    car_speeds = car_data.values

    # Linear transformation to scale car speeds to wheelchair speeds
    wheelchair_speeds = map(car_speeds, car_speeds.min(), car_speeds.max(), config.accessibility.minimum_speed, config.accessibility.maximum_speed)

    # Adding some randomness to simulate realistic wheelchair speed variation
    wheelchair_speeds += np.random.normal(0, config.accessibility.random_max, wheelchair_speeds.shape)

    # Apply the accessibility scores transformation
    for i, score in enumerate(accessibility_scores):
        wheelchair_speeds[:, i] *= config.accessibility.score_multiplier * score + 1

    # Ensure that the wheelchair speeds are within the range of the car speeds with linear transformation
    wheelchair_speeds = map(wheelchair_speeds, wheelchair_speeds.min(), wheelchair_speeds.max(), config.accessibility.minimum_speed, config.accessibility.maximum_speed)

    wheelchair_data[:] = wheelchair_speeds

    return wheelchair_data


@timer
def create_csv_files() -> None:
    # Define paths
    adj_mx_path = Path('data/original/adj_mx.pkl')
    metr_la_path = Path('data/original/metr-la.h5')
    accessibility_path = Path('data/accessibility/_get_accessibility.csv')
    save_dir = Path('data/crafted')

    # Read data
    car_data = pd.read_hdf(metr_la_path)
    accessibility_data = pd.read_csv(accessibility_path)

    # Process data
    car_data = car_data.iloc[:5000]  # FIXME: Because of RAM issue, limit the number of rows
    accessibility_data = compute_accessibility_score(accessibility_data)
    wheelchair_data = craft_wheelchair_data(car_data, accessibility_data['accessibility_score'])

    # Save data
    car_data.filename = "car_data"
    accessibility_data.filename = "accessibility_data"
    wheelchair_data.filename = "wheelchair_data"

    save_csv_files([car_data, accessibility_data, wheelchair_data], save_dir, verbose=config.verbose)

    with open(adj_mx_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)

    np.save(save_dir / 'adj_mx.npy', adj_mx)
