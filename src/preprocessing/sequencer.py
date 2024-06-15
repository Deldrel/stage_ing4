from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.helpers.csv_file_manager import load_csv_files
from src.helpers.decorators import timer


def create_sequences(dfs: List[pd.DataFrame], verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if len(set([df.shape for df in dfs])) != 1:
        raise ValueError("All DataFrames must have the same shape")

    for df in dfs:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

    # array shape (number_of_sequences, sequence_length, number_of_sensors, number_of_features)
    sequence_length = 12
    time_of_day = True
    day_of_week = True

    number_of_sequences = dfs[0].shape[0] - 2 * sequence_length + 1
    number_of_sensors = dfs[0].shape[1] - 1  # because 1st column is timestamp
    number_of_features = len(dfs) + time_of_day + day_of_week

    x = np.zeros((number_of_sequences, sequence_length, number_of_sensors, number_of_features))
    y = np.zeros((number_of_sequences, sequence_length, number_of_sensors, number_of_features))

    for i, df in enumerate(dfs):
        sensor_data = df.iloc[:, 1:].values

        for t in range(number_of_sequences):
            x[t, :, :, i] = sensor_data[t:t + sequence_length, :]
            y[t, :, :, i] = sensor_data[t + sequence_length:t + 2 * sequence_length, :]

    if time_of_day:
        time_of_day_feature = (dfs[0].index - dfs[0].index.normalize()) / np.timedelta64(1, "D")
        time_of_day_feature = np.tile(time_of_day_feature.values[:, np.newaxis], (1, number_of_sensors))

        for t in range(number_of_sequences):
            x[t, :, :, len(dfs)] = time_of_day_feature[t:t + sequence_length, :]
            y[t, :, :, len(dfs)] = time_of_day_feature[t + sequence_length:t + 2 * sequence_length, :]

    if day_of_week:
        day_of_week_feature = np.zeros((dfs[0].shape[0], 7))
        day_of_week_feature[np.arange(dfs[0].shape[0]), dfs[0].index.dayofweek] = 1
        day_of_week_feature = np.tile(day_of_week_feature[:, np.newaxis, :], (1, number_of_sensors, 1))

        for t in range(number_of_sequences):
            x[t, :, :, len(dfs) + time_of_day] = day_of_week_feature[t:t + sequence_length, :, 0]
            y[t, :, :, len(dfs) + time_of_day] = day_of_week_feature[t + sequence_length:t + 2 * sequence_length, :, 0]

    if verbose:
        print(f"X shape: {x.shape}, estimated size: {x.nbytes / 1e9:.2f} GB")
        print(f"Y shape: {y.shape}, estimated size: {y.nbytes / 1e9:.2f} GB")

    return x, y


def split_data(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = x.shape[0]
    num_train = int(num_samples * train_ratio)
    num_val = (num_samples - num_train) // 2

    x_train = x[:num_train]
    y_train = y[:num_train]
    x_val = x[num_train:num_train + num_val]
    y_val = y[num_train:num_train + num_val]
    x_test = x[num_train + num_val:]
    y_test = y[num_train + num_val:]

    return x_train, y_train, x_val, y_val, x_test, y_test


@timer
def sequence() -> None:
    crafted_path = Path('data/crafted')
    sequences_path = Path('data/sequences')
    sequences_path.mkdir(parents=True, exist_ok=True)

    dfs = load_csv_files(crafted_path, verbose=True)
    x, y = create_sequences(dfs)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y, train_ratio=0.8)

    print(f"Saving sequences to {sequences_path}...")
    np.savez_compressed(sequences_path / 'train.npz', x=x_train, y=y_train)
    np.savez_compressed(sequences_path / 'val.npz', x=x_val, y=y_val)
    np.savez_compressed(sequences_path / 'test.npz', x=x_test, y=y_test)
