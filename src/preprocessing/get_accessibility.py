from pathlib import Path

import overpy
import pandas as pd

from src.config import config
from src.helpers.csv_file_manager import process_csv_files
from src.helpers.decorators import preserve_custom_attribute, timer


def get_wheelchair_accessibility(api, lat, lon):
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

    # Count accessible nodes, ways, and relations
    accessible_count = 0
    for element in result.nodes + result.ways + result.relations:
        wheelchair = element.tags.get("wheelchair")
        if wheelchair == "yes":
            accessible_count += config.accessibility.score_yes
        elif wheelchair == "limited":
            accessible_count += config.accessibility.score_limited
        elif wheelchair == "no":
            accessible_count += config.accessibility.score_no

    # Return a score based on the number of accessible elements
    return accessible_count


@preserve_custom_attribute('filename')
def get_accessibility(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    api = overpy.Overpass()

    counter = 0
    for idx, row in df.iterrows():
        result = get_wheelchair_accessibility(api, row['latitude'], row['longitude'])
        df_copy.loc[idx, 'wheelchair'] = result
        counter += 1
        if counter % 10 == 0:
            print(f'Processed {idx} rows')

    return df_copy


@timer
def create_accessibility_df() -> None:
    locations_path = Path('data/original/graph_sensor_locations.csv')
    save_dir = Path('data/accessibility')
    df = pd.read_csv(locations_path)
    df.filename = 'access'
    process_csv_files(get_accessibility, df, output_dir=save_dir, verbose=config.verbose)
