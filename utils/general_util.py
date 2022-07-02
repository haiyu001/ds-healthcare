from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
import logging
import json
import csv
import os


def get_repo_dir() -> str:
    return str(Path(__file__).parent.parent)


def make_dir(dir: str) -> str:
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


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


def save_pdf(pdf: pd.DataFrame,
             save_filepath: str,
             rename_columns: Optional[Dict] = None,
             csv_index: bool = False,
             csv_index_label: Optional[str] = None,
             csv_quoting: int = csv.QUOTE_MINIMAL):
    if rename_columns is not None:
        pdf.columns = [rename_columns.get(i, i) for i in pdf.columns]
    file_format = Path(save_filepath).suffix[1:] if save_filepath else None
    if file_format == "csv":
        pdf.to_csv(save_filepath, index=csv_index, index_label=csv_index_label, quoting=csv_quoting)
    elif file_format == "json":
        pdf.to_json(save_filepath, orient="records", lines=True, force_ascii=False)
    elif file_format is not None:
        raise ValueError(f"Unsupported file format of {file_format}")


def load_json_file(input_filepath: str) -> Dict[Any, Any]:
    with open(input_filepath, "r", encoding="utf-8") as input:
        json_dict = json.load(input)
        return json_dict


def dump_json_file(json_dict: Dict[Any, Any], output_filepath: str):
    with open(output_filepath, "w", encoding="utf-8") as output:
        json.dump(json_dict, output, ensure_ascii=False, indent=4)


def load_pickle_file(pickle_file_path):
    with open(pickle_file_path, "rb") as input:
        item = pickle.load(input)
    return item


def dump_pickle_file(item, pickle_file_path: str):
    with open(pickle_file_path, "wb") as input:
        pickle.dump(item, input)


def setup_logger(logs_dir: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # console logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(handler_formatter)
    logger.addHandler(handler)

    # file logging
    logs_dir = logs_dir or os.path.join(get_repo_dir(), "logs")
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filepath = os.path.join(logs_dir, f"{log_filename}.log")
    handler = logging.FileHandler(log_filepath, "w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(handler_formatter)
    logger.addHandler(handler)