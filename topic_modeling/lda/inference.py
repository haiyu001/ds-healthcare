from typing import List, Tuple

from topic_modeling.lda.mallet_wrapper import LdaMallet
from topic_modeling.lda_utils.pipeline_util import get_model_filename, get_model_folder_name
import os

from utils.general_util import load_pickle_file


def extract_topics(mallet_corpus_filepath: str,
                   mallet_model_filepath: str) -> List[List[Tuple[int, float]]]:
    mallet_corpus: List[Tuple[str, List[Tuple[int, int]]]] = load_pickle_file(mallet_corpus_filepath)
    mallet_model = LdaMallet.load(mallet_model_filepath)
    return mallet_model[mallet_corpus]
