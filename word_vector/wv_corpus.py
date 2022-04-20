from typing import Optional, Dict, List, Iterator
from annotation.annotation_utils.annotation_util import load_blank_nlp
from utils.spark_util import write_sdf_to_file
from pyspark import Row
from pyspark.sql import DataFrame, Column
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from spacy import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Token
from spacy.util import filter_spans
import pandas as pd


def udf_get_text(tokens: Column, to_lower: bool = True) -> Column:
    def get_words(tokens: List[Row]) -> str:
        return " ".join([token.text.lower() if to_lower else token.text for token in tokens])

    return F.udf(get_words, StringType())(tokens)


def pudf_get_wv_corpus_line(text_iter: Column,
                            lang: str,
                            spacy_package: str,
                            ngram_match_dict: Optional[Dict[str, str]] = None,
                            ngram_match_lowercase: bool = True) -> Column:
    def get_wv_corpus_line(text_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        nlp = load_blank_nlp(lang, spacy_package, whitespace_tokenizer=True)
        ngram_matcher = None
        if ngram_match_dict is not None:
            ngram_matcher = get_ngram_matcher(nlp, list(ngram_match_dict.keys()), ngram_match_lowercase)
        for text in text_iter:
            doc = text.apply(nlp)
            if ngram_matcher is not None:
                doc = doc.apply(match_ngram, ngram_matcher=ngram_matcher)
            wv_corpus_line = doc.apply(doc_to_wv_corpus_line, ngram_match_dict=ngram_match_dict)
            yield wv_corpus_line

    return F.pandas_udf(get_wv_corpus_line, StringType())(text_iter)


def _is_valid_token(token: Token) -> bool:
    return not token.like_email and not token.like_url and not token.is_punct


def doc_to_wv_corpus_line(doc: Doc, ngram_match_dict: Optional[Dict[str, str]] = None) -> str:
    token_texts = []
    for token in doc:
        if _is_valid_token(token):
            token_text = token.text
            if ngram_match_dict is not None:
                token_text = ngram_match_dict.get(token_text, token_text)
            if token_text:
                token_texts.append(token_text)
    wv_corpus_line = " ".join(token_texts)
    return wv_corpus_line


def match_ngram(doc: Doc, ngram_matcher):
    matches = ngram_matcher(doc)
    spans = []
    for _, start, end in matches:
        span = doc[start: end]
        spans.append(span)
    with doc.retokenize() as retokenizer:
        for span in filter_spans(spans):
            retokenizer.merge(span)
    return doc


def get_ngram_matcher(nlp: Language, ngrams: List[str], match_lowercase: bool = True):
    ngram_matcher = PhraseMatcher(nlp.vocab)
    ngrams = [ngram.lower() for ngram in ngrams] if match_lowercase else ngrams
    ngrams_docs = list(nlp.tokenizer.pipe(ngrams))
    ngram_matcher.add("ngram_match", ngrams_docs)
    return ngram_matcher


def extact_wv_corpus_from_annotation(annotation_sdf: DataFrame,
                                     lang: str,
                                     spacy_package: str,
                                     wv_corpus_filepath: str,
                                     ngram_match_dict: Optional[Dict[str, str]] = None,
                                     ngram_match_lowercase: bool = True,
                                     num_partitions: int = 1):
    text_sdf = annotation_sdf.select(udf_get_text(F.col("tokens"), ngram_match_lowercase).alias("text"))
    wv_corpus_sdf = text_sdf.select(pudf_get_wv_corpus_line(F.col("text"),
                                                            lang,
                                                            spacy_package,
                                                            ngram_match_dict,
                                                            ngram_match_lowercase))
    write_sdf_to_file(wv_corpus_sdf, wv_corpus_filepath, num_partitions)


if __name__ == "__main__":
    from annotation.annotation_utils.annotation_util import read_nlp_model_config, read_annotation_config
    from utils.resource_util import get_repo_dir, get_data_filepath
    from annotation.annotation_utils.annotation_util import load_annotation
    from annotation.components.canonicalizer import get_bigram_match_dict
    from utils.spark_util import get_spark_session
    import os

    nlp_model_config_filepath = os.path.join(get_repo_dir(), "conf", "nlp_model_template.cfg")
    nlp_model_config = read_nlp_model_config(nlp_model_config_filepath)
    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)
    canonicalization_folder = annotation_config["canonicalization_folder"]
    canonicalization_wv_folder = annotation_config["canonicalization_wv_folder"]

    domain_dir = get_data_filepath(annotation_config["domain"])
    annotation_dir = os.path.join(domain_dir, annotation_config["annotation_folder"])
    bigram_norm_candidates_filepath = os.path.join(domain_dir, canonicalization_folder,
                                                   annotation_config["bigram_norm_candidates_filename"])
    wv_corpus_filepath = os.path.join(domain_dir, canonicalization_folder, canonicalization_wv_folder,
                                      annotation_config["canonicalization_wv_corpus_filename"])

    spark = get_spark_session("test", config_updates={}, master_config="local[4]", log_level="WARN")

    extact_wv_corpus_from_annotation(
        annotation_sdf=load_annotation(spark, annotation_dir, annotation_config["drop_non_english"]),
        lang=nlp_model_config["lang"],
        spacy_package=nlp_model_config["spacy_package"],
        wv_corpus_filepath=wv_corpus_filepath,
        ngram_match_dict=get_bigram_match_dict(bigram_norm_candidates_filepath),
        ngram_match_lowercase=True,
        num_partitions=4
    )
