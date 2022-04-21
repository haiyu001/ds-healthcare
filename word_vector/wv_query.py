from typing import Tuple, List, Optional, Union
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
from utils.general_util import save_pdf
from IPython.core.display import display
import pandas as pd


def get_result_pdf(tuple_list: List[Tuple[str, float, int]]) -> pd.DataFrame:
    result_json = []
    for (neighbour, similarity, count) in tuple_list:
        result_json.append({"word": " ".join(neighbour.split("_")),
                            "similarity": round(similarity, 3),
                            "count": count, })
    result_df = pd.DataFrame(result_json)
    return result_df


def query_word(word: str,
               wv_model: Union[FastTextKeyedVectors, KeyedVectors],
               topn: int = 50,
               min_count: int = 5,
               save_filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
    wv_word = "_".join(word.strip().lower().split())
    if wv_word not in wv_model.key_to_index:
        print(f"word {word} not in vocabulary")
    else:
        neighbours = wv_model.similar_by_word(wv_word, 100)
        neighbours = [[word, similarity, wv_model.get_vecattr(word, "count")]
                      for word, similarity in neighbours if wv_model.get_vecattr(word, "count") >= min_count]
        neighbours = neighbours[:topn]
        result_pdf = get_result_pdf(neighbours)
        if save_filepath is not None:
            save_pdf(result_pdf, save_filepath)
        return result_pdf


def query_words(words: List[str],
                wv_model: Union[FastTextKeyedVectors, KeyedVectors],
                topn: int = 50,
                min_count: int = 5):
    for word in words:
        print(f"{'=' * 10} {word} {'=' * 10}")
        result_df = query_word(word, wv_model, topn=topn, min_count=min_count)
        display(result_df)
