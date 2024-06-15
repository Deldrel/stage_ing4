import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.helpers.csv_file_manager import process_csv_files, save_csv_files
from src.helpers.decorators import timer


def craft_wheelchair_data(car_data: pd.DataFrame) -> pd.DataFrame:
    wheelchair_data = car_data.copy()

    minimum_speed = 0
    maximum_speed = 5

    car_speeds = car_data.values
    car_speed_min = car_speeds[car_speeds > 0].min()
    car_speed_max = car_speeds[car_speeds > 0].max()

    # Linear transformation to scale car speeds to wheelchair speeds
    wheelchair_speeds = minimum_speed + (car_speeds - car_speed_min) * (maximum_speed - minimum_speed) / (
            car_speed_max - car_speed_min)

    # Adding some randomness to simulate realistic wheelchair speed variation
    wheelchair_speeds += np.random.normal(0, 0.5, wheelchair_speeds.shape)

    # Ensuring speeds are within the defined range
    wheelchair_speeds = np.clip(wheelchair_speeds, minimum_speed, maximum_speed)

    wheelchair_data[:] = wheelchair_speeds

    return wheelchair_data


@timer
def create_csv_files() -> None:
    adj_mx_path = Path('data/original/adj_mx.pkl')
    metr_la_path = Path('data/original/metr-la.h5')
    save_dir = Path('data/crafted')

    car_data = pd.read_hdf(metr_la_path)
    car_data, wheelchair_data = process_csv_files(craft_wheelchair_data, car_data, keep_originals=True, verbose=True)

    car_data.filename = "car_data"
    wheelchair_data.filename = "wheelchair_data"

    save_csv_files([car_data, wheelchair_data], save_dir, verbose=True)

    with open(adj_mx_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)

    np.save(save_dir / 'adj_mx.npy', adj_mx)
