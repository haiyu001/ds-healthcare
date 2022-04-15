from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import json
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
    with open(input_filepath, 'r', encoding='utf-8') as input:
        json_dict = json.load(input)
        return json_dict


def dump_json_file(json_dict: Dict[Any, Any], ouput_filepath: str):
    with open(ouput_filepath, 'w', encoding='utf-8') as output:
        json.dump(json_dict, output, ensure_ascii=False, indent=4)

