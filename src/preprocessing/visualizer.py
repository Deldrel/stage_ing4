from pathlib import Path
from typing import List, Union

import folium
import matplotlib.pyplot as plt
import pandas as pd

from src.config import config
from src.helpers.csv_file_manager import load_csv_files
from src.helpers.decorators import timer


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


def create_sensor_map():
    path = Path('data/original/graph_sensor_locations.csv')
    output_path = Path('data/accessibility/sensor_map.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)

    latitude = df['latitude'].mean()
    longitude = df['longitude'].mean()
    sensor_map = folium.Map(location=[latitude, longitude], zoom_start=12)

    for idx, row in df.iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=f"Sensor ID: {row['sensor_id']}").add_to(sensor_map)

    sensor_map.save(output_path)
    print(f'Sensor map saved at {output_path}')


def create_accessibility_map():
    path = Path('data/accessibility/access_get_accessibility.csv')
    output_path = Path('data/accessibility/accessibility_map.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)

    latitude = df['latitude'].mean()
    longitude = df['longitude'].mean()
    accessibility_map = folium.Map(location=[latitude, longitude], zoom_start=12)

    # Define the radius (in meters)
    radius = config.accessibility.query_radius

    for idx, row in df.iterrows():
        wheelchair = row['wheelchair']
        color = 'green' if wheelchair > config.accessibility.green_threshold else 'orange' if wheelchair > config.accessibility.orange_threshold else 'red'

        # Add a marker for each sensor
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"Sensor ID: {row['sensor_id']}\n Wheelchair: {wheelchair}",
            icon=folium.Icon(color=color)
        ).add_to(accessibility_map)

        # Add a circle to indicate the 'around' radius
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            popup=f"Radius: {radius} meters"
        ).add_to(accessibility_map)

    accessibility_map.save(output_path)
    print(f'Accessibility map saved at {output_path}')


@timer
def create_maps():
    create_sensor_map()
    create_accessibility_map()


@timer
def visualize():
    dfs = load_csv_files('data/crafted', verbose=True)
    dfs = [df.drop(df.columns[0], axis=1) for df in dfs]
    visualize_speed(dfs, sensor_index=[0, 1], day_index=0, time_period=24 * 12)
