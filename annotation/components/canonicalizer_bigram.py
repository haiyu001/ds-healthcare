from typing import Set, Dict, Union, Optional, Iterator, List, Tuple
from utils.general_util import save_pdf
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors, FastText
from hunspell.hunspell import HunspellWrap
from pyspark.sql.types import BooleanType, ArrayType, StringType
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
import pandas as pd


def pudf_is_valid_bigram(bigram: Column) -> Column:
    def is_valid_bigram(bigram: pd.Series) -> pd.Series:
        words = bigram.apply(lambda x: x.split())
        valid_candidate = words.apply(lambda x: len(x[0]) >= 2 and len(x[1]) >= 3)
        return valid_candidate

    return F.pandas_udf(is_valid_bigram, BooleanType())(bigram)


def pudf_get_valid_suggestions(suggestions: Column, unigram: Set[str]) -> Column:
    def get_valid_suggestions(suggestions: pd.Series) -> pd.Series:
        valid_suggestions = suggestions.apply(lambda x: [i for i in x if i.lower() in unigram])
        return valid_suggestions

    return F.pandas_udf(get_valid_suggestions, ArrayType(StringType()))(suggestions)


def _get_unigram_bigram_similarity(unigram: str,
                                   bigram: str,
                                   wv_model: Union[FastTextKeyedVectors, KeyedVectors]) -> Optional[float]:
    similarity = None
    concat_bigram = "_".join(bigram.split())
    if unigram in wv_model.key_to_index and concat_bigram in wv_model.key_to_index:
        similarity = wv_model.similarity(unigram, concat_bigram)
    return similarity


def _get_bigram_canonicalization_canonical(unigram: str, bigram: str, unigram_count: int, bigram_count: int,
                                           hunspell_checker: HunspellWrap) -> str:
    if unigram_count > bigram_count:
        canonical = unigram
    elif unigram_count < bigram_count:
        canonical = bigram
    else:
        canonical = unigram if hunspell_checker.spell(unigram) else bigram
    return canonical


def get_bigram_canonicalization_candidates_match_dict(bigram_canonicalization_candidates_filepath: str,
                                                      match_lowercase: bool = True) -> Dict[str, str]:
    bigram_canonicalization_candidates_pdf = pd.read_csv(bigram_canonicalization_candidates_filepath, encoding="utf-8",
                                                         keep_default_na=False, na_values="")
    bigrams = bigram_canonicalization_candidates_pdf["bigram"].str.lower().tolist() if match_lowercase \
        else bigram_canonicalization_candidates_pdf["bigram"].tolist()
    bigram_match_dict = {bigram: "_".join(bigram.strip().split()) for bigram in bigrams}
    return bigram_match_dict


def get_bigram_canonicalization_candidates(unigram_sdf: DataFrame,
                                           bigram_sdf: DataFrame,
                                           bigram_canonicalization_candidates_filepath: str,
                                           num_partitions: int = 1):
    unigram_sdf = unigram_sdf.select(F.col("word").alias("unigram"),
                                     F.col("count").alias("unigram_count"))
    bigram_sdf = bigram_sdf.select(F.regexp_replace(F.col("ngram"), " ", "").alias("unigram"),
                                   F.col("ngram").alias("bigram"),
                                   F.col("count").alias("bigram_count"))
    bigram_canonicalization_candidates_sdf = unigram_sdf.join(bigram_sdf, on="unigram", how="inner")
    bigram_canonicalization_candidates_sdf = bigram_canonicalization_candidates_sdf \
        .select("unigram", "bigram", "unigram_count", "bigram_count")
    bigram_canonicalization_candidates_sdf = bigram_canonicalization_candidates_sdf
    bigram_canonicalization_candidates_sdf = bigram_canonicalization_candidates_sdf.filter(
        pudf_is_valid_bigram(F.col("bigram")))
    write_sdf_to_file(bigram_canonicalization_candidates_sdf, bigram_canonicalization_candidates_filepath,
                      num_partitions)


def get_bigram_canonicalization(bigram_canonicalization_candidates_filepath: str,
                                wv_model_filepath: str,
                                bigram_canonicalization_filepath: str,
                                wv_bigram_canonicalization_filter_min_similarity: float = 0.8):
    pdf = pd.read_csv(bigram_canonicalization_candidates_filepath, encoding="utf-8", keep_default_na=False,
                      na_values="")
    wv_model = FastText.load(wv_model_filepath).wv
    pdf["similarity"] = pdf.apply(lambda x: _get_unigram_bigram_similarity(x["unigram"], x["bigram"], wv_model), axis=1)
    pdf["max_count"] = pdf[["unigram_count", "bigram_count"]].max(axis=1)
    negative_auxiliary_pdf = pdf[(pdf["unigram"].str.endswith("n't")) | (pdf["unigram"].str.endswith("n`t"))]
    other_pdf = pdf[~((pdf["unigram"].str.endswith("n't")) | (pdf["unigram"].str.endswith("n`t")))]
    other_pdf = other_pdf.dropna(subset=["similarity"])
    other_pdf = other_pdf[other_pdf["similarity"] >= wv_bigram_canonicalization_filter_min_similarity]
    pdf = pd.concat([negative_auxiliary_pdf, other_pdf])
    pdf = pdf.sort_values(by="max_count", ascending=False).drop(columns=["max_count"])
    save_pdf(pdf, bigram_canonicalization_filepath)


if __name__ == "__main__":
    from annotation.annotation_utils.annotator_util import read_annotation_config
    from utils.resource_util import get_data_filepath, get_repo_dir
    from utils.spark_util import get_spark_session, write_sdf_to_file
    import os

    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)

    domain_dir = get_data_filepath(annotation_config["domain"])
    canonicalization_dir = os.path.join(domain_dir, annotation_config["canonicalization_folder"])
    canonicalization_extraction_dir = os.path.join(
        canonicalization_dir, annotation_config["canonicalization_extraction_folder"])
    canonicalization_wv_folder = annotation_config["canonicalization_wv_folder"]

    unigram_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["canonicalization_unigram_filename"])
    bigram_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["canonicalization_bigram_filename"])
    bigram_canonicalization_candidates_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["bigram_canonicalization_candidates_filename"])
    bigram_canonicalization_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["bigram_canonicalization_filename"])

    # ================================ bigram canonicalization candidates =======================================

    spark_cores = 6
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="WARN")

    unigram_sdf = spark.read.csv(unigram_filepath, header=True, quote='"', escape='"', inferSchema=True)
    bigram_sdf = spark.read.csv(bigram_filepath, header=True, quote='"', escape='"', inferSchema=True)
    get_bigram_canonicalization_candidates(unigram_sdf,
                                           bigram_sdf,
                                           bigram_canonicalization_candidates_filepath)

    # ===================================== bigram canonicalization ============================================

    wv_model_filepath = os.path.join(canonicalization_dir, canonicalization_wv_folder, "model", "fasttext")
    get_bigram_canonicalization(bigram_canonicalization_candidates_filepath,
                                wv_model_filepath,
                                bigram_canonicalization_filepath,
                                annotation_config["wv_bigram_canonicalization_filter_min_similarity"])
