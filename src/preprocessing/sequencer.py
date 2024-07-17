from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import config
from src.helpers.csv_file_manager import load_csv_files
from src.helpers.decorators import timer


def map(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def create_sequences(dfs: List[pd.DataFrame], verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(set([df.shape for df in dfs])) != 1:
        raise ValueError("All DataFrames must have the same shape")

    for df in dfs:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

    sequence_length = config.sequencer.sequence_length
    time_of_day = config.sequencer.time_of_day
    day_of_week = config.sequencer.day_of_week

    number_of_sequences = dfs[0].shape[0] - 2 * sequence_length + 1
    number_of_sensors = dfs[0].shape[1] - 1  # because 1st column is timestamp
    number_of_features = 3  # speed, time of day, day of week

    results = []
    in_min, in_max = 0, 80
    out_min, out_max = 0, 1

    for vehicle_type_index, df in enumerate(dfs):
        sensor_data = df.iloc[:, 1:].values
        # sensor_data = map(sensor_data, in_min, in_max, out_min, out_max)

        x = np.memmap(f"temp_x_{vehicle_type_index}.dat", dtype='float32', mode='w+',
                      shape=(number_of_sequences, sequence_length, number_of_sensors, number_of_features + 1))
        y = np.memmap(f"temp_y_{vehicle_type_index}.dat", dtype='float32', mode='w+',
                      shape=(number_of_sequences, sequence_length, number_of_sensors, number_of_features + 1))

        for t in range(number_of_sequences):
            x[t, :, :, 0] = sensor_data[t:t + sequence_length, :]
            y[t, :, :, 0] = sensor_data[t + sequence_length:t + 2 * sequence_length, :]

        if time_of_day:
            time_of_day_feature = (df.index - df.index.normalize()) / np.timedelta64(1, "D")
            time_of_day_feature = np.tile(time_of_day_feature.values[:, np.newaxis], (1, number_of_sensors))

            for t in range(number_of_sequences):
                x[t, :, :, 1] = time_of_day_feature[t:t + sequence_length, :]
                y[t, :, :, 1] = time_of_day_feature[t + sequence_length:t + 2 * sequence_length, :]

        if day_of_week:
            day_of_week_feature = np.zeros((df.shape[0], 7))
            day_of_week_feature[np.arange(df.shape[0]), df.index.dayofweek] = 1
            day_of_week_feature = np.tile(day_of_week_feature[:, np.newaxis, :], (1, number_of_sensors, 1))

            for t in range(number_of_sequences):
                x[t, :, :, 2] = day_of_week_feature[t:t + sequence_length, :, 0]
                y[t, :, :, 2] = day_of_week_feature[t + sequence_length:t + 2 * sequence_length, :, 0]

        # Adding vehicle type as a feature
        vehicle_type_feature = np.full((sequence_length, number_of_sensors), vehicle_type_index)

        for t in range(number_of_sequences):
            x[t, :, :, 3] = vehicle_type_feature
            y[t, :, :, 3] = vehicle_type_feature

        if verbose:
            print(f"Vehicle type {vehicle_type_index}:")
            print(f"X shape: {x.shape}, estimated size: {x.nbytes / 1e9:.3f} GB")
            print(f"Y shape: {y.shape}, estimated size: {y.nbytes / 1e9:.3f} GB")
            print(f"X example: {x[0, 0, 0, :]}")
            print(f"Y example: {y[0, 0, 0, :]}")

        results.append((x, y))

    return results


def split_data(x_all, y_all, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of train_ratio, val_ratio, and test_ratio must be 1.0"

    # Shuffle the data
    indices = np.arange(x_all.shape[0])
    np.random.shuffle(indices)

    x_all = x_all[indices]
    y_all = y_all[indices]

    total_samples = x_all.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    x_train, y_train = x_all[:train_end], y_all[:train_end]
    x_val, y_val = x_all[train_end:val_end], y_all[train_end:val_end]
    x_test, y_test = x_all[val_end:], y_all[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test


@timer
def sequence() -> None:
    crafted_path = Path(config.sequencer.crafted_path)
    sequences_path = Path(config.sequencer.sequences_path)
    sequences_path.mkdir(parents=True, exist_ok=True)

    dfs = load_csv_files(crafted_path, verbose=config.verbose)
    dfs = dfs[1:]  # tej score accessibilité

    print(f"Creating sequences...")
    sequences = create_sequences(dfs, verbose=config.verbose)
    x_all = np.concatenate([np.memmap(f"temp_x_{i}.dat", dtype='float32', mode='r', shape=seq[0].shape) for i, seq in
                            enumerate(sequences)], axis=0)
    y_all = np.concatenate([np.memmap(f"temp_y_{i}.dat", dtype='float32', mode='r', shape=seq[1].shape) for i, seq in
                            enumerate(sequences)], axis=0)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_all, y_all, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    print(f"Train shape: {x_train.shape}, {y_train.shape}")
    print(f"Val shape: {x_val.shape}, {y_val.shape}")
    print(f"Test shape: {x_test.shape}, {y_test.shape}")

    print(f"Saving sequences to {sequences_path}...")
    np.savez_compressed(sequences_path / 'train.npz', x=x_train, y=y_train)
    print(f"Saved train.npz")
    np.savez_compressed(sequences_path / 'val.npz', x=x_val, y=y_val)
    print(f"Saved val.npz")
    np.savez_compressed(sequences_path / 'test.npz', x=x_test, y=y_test)
    print(f"Saved test.npz")

