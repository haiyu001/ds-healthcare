from typing import List, Optional, Dict
from annotation.annotation_utils.corpus_util import is_valid_token, pudf_get_corpus_line
from utils.spark_util import write_sdf_to_file
from pyspark import Row
from pyspark.sql import Column, functions as F, DataFrame
from pyspark.sql.types import StringType
import string


def udf_get_doc_text_by_token_texts(tokens: Column, to_lower: bool = True) -> Column:
    def get_doc_text_by_token_texts(tokens: List[Row]) -> str:
        doc_token_texts = []
        for token in tokens:
            token_text = token.text
            if to_lower:
                token_text = token_text.lower()
            token_text = token_text.strip(string.punctuation)
            if is_valid_token(token_text):
                doc_token_texts.append(token_text)
        return " ".join(doc_token_texts)

    return F.udf(get_doc_text_by_token_texts, StringType())(tokens)


def build_wv_corpus_by_annotation(annotation_sdf: DataFrame,
                                  lang: str,
                                  spacy_package: str,
                                  corpus_filepath: str,
                                  ngram_match_dict: Optional[Dict[str, str]] = None,
                                  match_lowercase: bool = True,
                                  num_partitions: Optional[int] = None):
    text_sdf = annotation_sdf.select(udf_get_doc_text_by_token_texts(F.col("tokens"), match_lowercase).alias("text"))
    corpus_sdf = text_sdf.select(pudf_get_corpus_line(F.col("text"),
                                                      lang,
                                                      spacy_package,
                                                      ngram_match_dict))
    write_sdf_to_file(corpus_sdf, corpus_filepath, num_partitions)