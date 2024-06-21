from pathlib import Path
from typing import Tuple

import overpy
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.config import config
from src.helpers.csv_file_manager import process_csv_files
from src.helpers.decorators import preserve_custom_attribute, timer


def get_wheelchair_accessibility(api, lat, lon) -> Tuple[int, int, int]:
    radius = config.accessibility.query_radius
    query = f"""
    [out:json];
    (
      node["wheelchair"](around:{radius},{lat},{lon});
      way["wheelchair"](around:{radius},{lat},{lon});
      relation["wheelchair"](around:{radius},{lat},{lon});
    );
    out center;
    """
    result = api.query(query)

    elements = result.nodes + result.ways + result.relations
    wheelchair_yes = sum(1 for element in elements if element.tags.get("wheelchair") == "yes")
    wheelchair_limited = sum(1 for element in elements if element.tags.get("wheelchair") == "limited")
    wheelchair_no = sum(1 for element in elements if element.tags.get("wheelchair") == "no")

    return wheelchair_yes, wheelchair_limited, wheelchair_no


@preserve_custom_attribute('filename')
def get_accessibility(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    api = overpy.Overpass()

    max_count = len(df)

    for idx, row in tqdm(df.iterrows(), total=max_count, desc="Processing", unit="row"):
        wheelchair_yes, wheelchair_limited, wheelchair_no = get_wheelchair_accessibility(api, row['latitude'], row['longitude'])
        df_copy.loc[idx, 'radius'] = config.accessibility.query_radius
        df_copy.loc[idx, 'wheelchair_yes'] = wheelchair_yes
        df_copy.loc[idx, 'wheelchair_limited'] = wheelchair_limited
        df_copy.loc[idx, 'wheelchair_no'] = wheelchair_no

    return df_copy


@timer
def create_accessibility_df() -> None:
    locations_path = Path('data/original/graph_sensor_locations.csv')
    save_dir = Path('data/accessibility')
    df = pd.read_csv(locations_path)
    df.filename = ''
    process_csv_files(get_accessibility, df, output_dir=save_dir, verbose=config.verbose)
