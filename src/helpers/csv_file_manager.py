from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
from tqdm import tqdm


def process_csv_files(
        process_function: Callable[[pd.DataFrame], pd.DataFrame],
        input_data: Union[pd.DataFrame, List[pd.DataFrame], Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        parallel: Optional[str] = None,
        keep_originals: bool = False,
        verbose: bool = False
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Process CSV files using a specified processing function either from a list of DataFrames or from a directory of CSV files.

    :param process_function: Function to process a single DataFrame. (Warning: no error handling is done in this function!)
    :param input_data: Dataframe, list of dataframes or path to the directory containing CSV files.
    :param output_dir: Path to save processed CSV files.
    :param parallel: Specifies parallel computing mode: 'thread', 'process', or None. (Warning: order is not guaranteed!)
    :param keep_originals: If True, retains original DataFrames and appends processed versions.
    :param verbose: If True, prints additional information.
    :return: Dataframe or list of Dataframes.
    """
    if isinstance(input_data, (str, Path)):
        dataframes = load_csv_files(input_data, verbose=verbose)
    else:
        dataframes = input_data

    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    if parallel in ("thread", "process"):
        executor = ThreadPoolExecutor() if parallel == "thread" else ProcessPoolExecutor()
        if verbose:
            print(f"Using {parallel} parallelism")
        with executor as exe:
            processed_dfs = list(tqdm(exe.map(process_function, dataframes),
                                      total=len(dataframes),
                                      desc=f"Processing {len(dataframes)} DataFrames",
                                      disable=not verbose))
    else:
        processed_dfs = [process_function(df) for df in tqdm(dataframes,
                                                             desc=f"Processing {len(dataframes)} DataFrames",
                                                             disable=not verbose)]

    for df in processed_dfs:
        if hasattr(df, "filename"):
            df.filename = f"{df.filename}_{process_function.__name__}"

    if verbose:
        print(f"Processed {len(processed_dfs)} DataFrames")

    final_dfs = dataframes + processed_dfs if keep_originals else processed_dfs

    if output_dir:
        save_csv_files(final_dfs, output_dir, verbose=verbose)

    if len(final_dfs) == 1:
        return final_dfs[0]

    return final_dfs


def load_csv_files(
        path: Union[str, Path],
        verbose: bool = False,
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load CSV files from a directory of CSV files.

    :param path: Path to the directory containing CSV files.
    :param verbose: If True, prints additional information.
    :return: Dataframe or list of DataFrames.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    files = sorted(path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    dataframes = []
    successful_loads = 0
    for file in tqdm(files, desc=f"Loading CSV files from {path}", disable=not verbose):
        try:
            df = pd.read_csv(file)
            filename = file.name.split(".")[0]
            df.filename = filename
            dataframes.append(df)
            successful_loads += 1
        except Exception as e:
            if verbose:
                print(f"Error loading {file}: {e}")

    if verbose:
        print(f"Loaded {successful_loads} CSV files from {path}")

    if len(dataframes) == 1:
        return dataframes[0]

    return dataframes


def save_csv_files(
        dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
        path: Union[str, Path],
        verbose: bool = False
) -> None:
    """
    Save DataFrames to a directory of CSV files.

    :param dataframes: List of DataFrames.
    :param path: Path to the directory to save CSV files.
    :param verbose: If True, prints additional information.
    :return: None
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    successful_saves = 0
    for i, df in enumerate(tqdm(dataframes, desc=f"Saving CSV files to {path}", disable=not verbose)):
        filename = df.filename if hasattr(df, "filename") else f"{i:09d}"
        filename = f"{filename}.csv" if not filename.endswith(".csv") else filename
        try:
            df.to_csv(path / filename, index=True)
            successful_saves += 1
        except Exception as e:
            if verbose:
                print(f"Error saving DataFrame {filename}: {e}")

    if verbose:
        print(f"Saved {successful_saves} CSV files to {path}")
