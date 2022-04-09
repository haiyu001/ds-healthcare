from typing import List, Tuple, Optional, Dict
from pathlib import Path
import pandas as pd
import csv
import os


def get_filepaths_recursively(input_dir: str, file_formats: List[str], sort: bool = True) -> List[str]:
    filepaths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_format = os.path.splitext(file)[-1][1:]
            if file_format in file_formats:
                filepaths.append(os.path.join(root, file))
    if sort:
        filepaths.sort()
    return filepaths


def split_filepath(filepath: str) -> Tuple[str]:
    filepath = Path(filepath)
    file_dir = str(filepath.parent)
    file_name = filepath.stem
    file_format = filepath.suffix[1:]
    return file_dir, file_name, file_format


def save_pandas_dataframe(dataframe: pd.DataFrame,
                          save_filepath: str,
                          rename_columns: Optional[Dict] = None,
                          csv_index: bool = False,
                          csv_index_label: Optional[str] = None,
                          csv_quoting: int = csv.QUOTE_MINIMAL):
    if rename_columns is not None:
        dataframe.columns = [rename_columns.get(i, i) for i in dataframe.columns]
    file_format = Path(save_filepath).suffix[1:] if save_filepath else None
    if file_format == "csv":
        dataframe.to_csv(save_filepath, index=csv_index, index_label=csv_index_label, quoting=csv_quoting)
    elif file_format == "json":
        dataframe.to_json(save_filepath, orient="records", lines=True, force_ascii=False)
    elif file_format is not None:
        raise ValueError(f"Unsupported file format of {file_format}")
