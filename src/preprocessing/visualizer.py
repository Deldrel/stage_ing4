from pathlib import Path
from typing import List, Union

import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

from src.config import config
from src.helpers.csv_file_manager import load_csv_files
from src.helpers.decorators import timer
from src.preprocessing.data_creation import accessibility_score, shannon_enthropy


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


def visualize_distribution(dfs: List[pd.DataFrame]) -> None:
    def optimal_bins(data):
        n = len(data)
        if n > 0:
            return int(np.ceil(np.sqrt(n)))
        return 10

    for idx, df in enumerate(dfs):
        data = df.iloc[:, 1:].values.flatten()
        mean = np.mean(data)
        std = np.std(data)

        plt.hist(data, bins=optimal_bins(data)//2, alpha=0.5, label=f'Dataframe {idx + 1}')

        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(mean + std, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(mean - std, color='r', linestyle='dashed', linewidth=1)

        plt.title('Distribution of Speeds with Mean and Standard Deviation')
        plt.xlabel('Speed')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


@timer
def create_accessibility_map(score_column: str = 'accessibility_score'):
    def get_color_from_value(value: float) -> str:
        if value < 0 or value > 1:
            raise ValueError('Value must be between 0 and 1')
        if value == 0:
            return 'black'
        gradient = mcolors.LinearSegmentedColormap.from_list('red_green', ['red', 'green'])
        color = mcolors.to_hex(gradient(value))
        return color

    path = Path('data/crafted/accessibility_data_compute_accessibility_score.csv')
    output_path = Path('data/accessibility/accessibility_map.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)

    latitude = df['latitude'].mean()
    longitude = df['longitude'].mean()
    accessibility_map = folium.Map(location=[latitude, longitude], zoom_start=12)

    radius = config.accessibility.query_radius

    for idx, row in df.iterrows():
        wheelchair = row[score_column]
        color = get_color_from_value(wheelchair)

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=f"Sensor ID: {row['sensor_id']}\n Wheelchair: {wheelchair}"
        ).add_to(accessibility_map)

        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            popup=f"Sensor ID: {row['sensor_id']}\n Wheelchair: {wheelchair}\n Radius: {radius}"
        ).add_to(accessibility_map)

    accessibility_map.save(output_path)
    print(f'Accessibility map saved at {output_path}')


def visualize_accessibility_score():
    wheelchair_yes = np.linspace(0, 20, 100)
    wheelchair_no = np.linspace(0, 20, 100)
    wheelchair_yes, wheelchair_no = np.meshgrid(wheelchair_yes, wheelchair_no)

    scores_accessibility_score = np.zeros_like(wheelchair_yes)
    scores_shannon_enthropy = np.zeros_like(wheelchair_yes)
    for i in range(wheelchair_yes.shape[0]):
        for j in range(wheelchair_yes.shape[1]):
            scores_accessibility_score[i, j] = accessibility_score(wheelchair_yes[i, j], wheelchair_no[i, j])
            scores_shannon_enthropy[i, j] = shannon_enthropy(wheelchair_yes[i, j], wheelchair_no[i, j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(wheelchair_yes, wheelchair_no, scores_accessibility_score, cmap='viridis')
    # ax.plot_surface(wheelchair_yes, wheelchair_no, scores_shannon_enthropy, cmap='viridis')
    ax.set_xlabel('Wheelchair Yes')
    ax.set_ylabel('Wheelchair No')
    ax.set_zlabel('Accessibility Score')
    plt.show()


@timer
def visualize():
    dfs = load_csv_files('data/crafted', verbose=True)
    dfs = dfs[1:]
    dfs = [df.drop(df.columns[0], axis=1) for df in dfs]
    visualize_speed(dfs, sensor_index=[0, 1, 6, 14], day_index=0, time_period=24 * 12)
    visualize_speed(dfs[1], sensor_index=[0, 1, 6, 14], day_index=0, time_period=24 * 12)
    #visualize_distribution(dfs)
    visualize_accessibility_score()
