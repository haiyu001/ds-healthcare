from typing import List, Tuple
from pathlib import Path
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



