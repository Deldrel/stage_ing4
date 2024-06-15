from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd

from src.helpers.csv_file_manager import load_csv_files


def visualize_speed(dfs: Union[pd.DataFrame, List[pd.DataFrame]] = None,
                    sensor_index: Union[int, List[int]] = 0,
                    day_index: int = 0,
                    time_period: int = 24 * 12) -> None:
    if dfs is None:
        return

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    if isinstance(sensor_index, int):
        sensor_index = [sensor_index]

    for idx in sensor_index:
        for df in dfs:
            plt.plot(df.iloc[day_index * time_period: (day_index + 1) * time_period, idx])

    plt.title(f'Speed of sensor(s) {sensor_index} from day {day_index} during {time_period // (24 * 12)} days')
    plt.xlabel('Time')
    plt.ylabel('Speed (mph)')
    plt.show()


def visualize():
    dfs = load_csv_files('data/crafted', verbose=True)
    dfs = [df.drop(df.columns[0], axis=1) for df in dfs]
    visualize_speed(dfs, sensor_index=[0, 1], day_index=0, time_period=24 * 12)
