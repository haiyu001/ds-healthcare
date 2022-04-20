from typing import Tuple, List, Optional, Union
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
from utils.general_util import save_pdf
from IPython.core.display import display
import pandas as pd


def get_result_pdf(tuple_list: List[Tuple[str, float, int]]) -> pd.DataFrame:
    result_json = []
    for (neighbour, similarity, count) in tuple_list:
        result_json.append({"word": round(similarity, 3),
                            "similarity": " ".join(neighbour.split("_")),
                            "count": count, })
    result_df = pd.DataFrame(result_json)
    return result_df


def query_word(word: str,
               word_vector: Union[FastTextKeyedVectors, KeyedVectors],
               topn: int = 50,
               min_count: int = 5,
               save_filepath: Optional[str] = None) -> pd.DataFrame:
    wv_word = "_".join(word.strip().lower().split())
    neighbours = None
    if wv_word not in word_vector.key_to_index:
        print(f"word {word} not in vocabulary")
    else:
        neighbours = word_vector.similar_by_word(wv_word, 100)
        neighbours = [[" ".join(t[0].split("_")), t[1], word_vector.get_vecattr(t[0], "count")]
                      for t in neighbours if word_vector.get_vecattr(t[0], "count") >= min_count]
        neighbours = neighbours[:topn]
    if neighbours:
        result_pdf = get_result_pdf(neighbours)
        if save_filepath is not None:
            save_pdf(result_pdf, save_filepath)
        return result_pdf


def query_words(words: List[str],
                word_vector: Union[FastTextKeyedVectors, KeyedVectors],
                topn: int = 50,
                min_count: int = 5):
    for word in words:
        print(word)
        result_df = query_word(word, word_vector, topn=topn, min_count=min_count)
        display(result_df)
        print()


if __name__ == "__main__":
    from gensim.models import FastText

    model_path = "/Users/haiyang/data/drug_test/canonicalization/canonicalizer_wv/model/fastText"
    word2vec_model = FastText.load(model_path).wv
    query_words(["gynocologist"], word2vec_model)
