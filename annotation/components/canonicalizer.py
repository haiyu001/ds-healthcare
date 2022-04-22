from typing import Set, Dict, Union, Optional
from annotation.pipes.spell_detector import get_hunspell_checker
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


def pudf_get_valid_suggestions(suggestions: Column, vocab: Set[str]) -> Column:
    def get_valid_suggestions(suggestions: pd.Series) -> pd.Series:
        valid_suggestions = suggestions.apply(lambda x: [i for i in x if i.lower() in vocab])
        return valid_suggestions

    return F.pandas_udf(get_valid_suggestions, ArrayType(StringType()))(suggestions)


def _get_unigram_bigram_similarity(unigram: str,
                                   bigram: str,
                                   wv_model: Union[FastTextKeyedVectors, KeyedVectors]) -> Optional[float]:
    concat_bigram = '_'.join(bigram.split())
    if unigram in wv_model.key_to_index and concat_bigram in wv_model.key_to_index:
        return wv_model.similarity(unigram, concat_bigram)


def _get_bigram_norm_canonical(unigram: str, bigram: str, unigram_count: int, bigram_count: int,
                               hunspell_checker: HunspellWrap) -> Optional[float]:
    if unigram_count > bigram_count:
        canonical = unigram
    elif unigram_count < bigram_count:
        canonical = bigram
    else:
        canonical = unigram if hunspell_checker.spell(unigram) else bigram
    return canonical


def get_bigram_norm_candidates_match_dict(bigram_norm_candidates_filepath: str,
                                          match_lowercase: bool = True) -> Dict[str, str]:
    bigram_norm_candidates_pdf = pd.read_csv(bigram_norm_candidates_filepath, encoding="utf-8",
                                             keep_default_na=False, na_values="")
    bigrams = bigram_norm_candidates_pdf["bigram"].str.lower().tolist() if match_lowercase \
        else bigram_norm_candidates_pdf["bigram"].tolist()
    bigram_match_dict = {bigram: "_".join(bigram.strip().split()) for bigram in bigrams}
    return bigram_match_dict


def get_bigram_norm_candidates(vocab_sdf: DataFrame,
                               bigram_sdf: DataFrame,
                               bigram_norm_candidates_filepath: str,
                               num_partitions: int = 1):
    vocab_sdf = vocab_sdf.select(F.col("word").alias("unigram"),
                                 F.col("count").alias("unigram_count"))
    bigram_sdf = bigram_sdf.select(F.regexp_replace(F.col("ngram"), " ", "").alias("unigram"),
                                   F.col("ngram").alias("bigram"),
                                   F.col("count").alias("bigram_count"))
    bigram_norm_candidates_sdf = vocab_sdf.join(bigram_sdf, on="unigram", how="inner")
    bigram_norm_candidates_sdf = bigram_norm_candidates_sdf \
        .select("unigram", "bigram", "unigram_count", "bigram_count")
    bigram_norm_candidates_sdf = bigram_norm_candidates_sdf
    bigram_norm_candidates_sdf = bigram_norm_candidates_sdf.filter(pudf_is_valid_bigram(F.col("bigram")))
    write_sdf_to_file(bigram_norm_candidates_sdf, bigram_norm_candidates_filepath, num_partitions)


def get_spell_norm_candidates(vocab_sdf: DataFrame,
                              annotation_sdf: DataFrame,
                              spell_checking_candidates_filepath: str,
                              vocab_filter_min_count: int = 5,
                              num_partitions: int = 1):
    vocab_sdf = vocab_sdf.filter(F.col("count") >= vocab_filter_min_count)
    vocab = set([x.word for x in vocab_sdf.select("word").distinct().collect()])

    misspelling_sdf = annotation_sdf.select(F.explode(annotation_sdf._.misspellings).alias("misspelling"))
    misspelling_sdf = misspelling_sdf.select(F.lower(F.col("misspelling").text).alias("misspelling"),
                                             F.size(F.col("misspelling").ids).alias("count"),
                                             F.col("misspelling").suggestions.alias("suggestions"))

    misspelling_sdf = misspelling_sdf.groupby(["misspelling"]).agg(
        F.sum("count").alias("count"),
        F.array_distinct(F.flatten(F.collect_set("suggestions"))).alias("suggestions"))

    spell_checking_candidates_sdf = misspelling_sdf.withColumn("suggestions",
                                                               pudf_get_valid_suggestions(F.col("suggestions"), vocab))
    spell_checking_candidates_sdf = spell_checking_candidates_sdf.filter(F.size("suggestions") > 0) \
        .orderBy(F.desc("count"))
    spell_checking_candidates_sdf = spell_checking_candidates_sdf.withColumn("suggestions",
                                                                             F.to_json(F.col("suggestions")))
    write_sdf_to_file(spell_checking_candidates_sdf, spell_checking_candidates_filepath, num_partitions)


def get_bigram_norm(bigram_norm_candidates_filepath: str,
                    wv_model_filepath: str,
                    bigram_norm_filepath: str,
                    wv_filter_min_similarity: int = 0.8):
    pdf = pd.read_csv(bigram_norm_candidates_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    wv_model = FastText.load(wv_model_filepath).wv
    hunspell_checker = get_hunspell_checker()
    pdf["similarity"] = pdf.apply(lambda x: _get_unigram_bigram_similarity(x["unigram"], x["bigram"], wv_model), axis=1)
    pdf = pdf.dropna(subset=["similarity"])
    pdf["canonical"] = pdf.apply(lambda x: _get_bigram_norm_canonical(x["unigram"], x["bigram"], x["unigram_count"],
                                                                      x["bigram_count"], hunspell_checker), axis=1)
    pdf = pdf[pdf["similarity"] >= wv_filter_min_similarity]
    pdf = pdf.sort_values(by="similarity", ascending=False)
    save_pdf(pdf, bigram_norm_filepath)


def get_spelling_norm(vocab_sdf: DataFrame):
    pass


if __name__ == "__main__":
    from annotation.annotation_utils.annotation_util import read_annotation_config, load_annotation
    from utils.resource_util import get_data_filepath, get_repo_dir
    from utils.spark_util import get_spark_session, write_sdf_to_file
    import os

    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)

    domain_dir = get_data_filepath(annotation_config["domain"])
    extraction_dir = os.path.join(domain_dir, annotation_config["extraction_folder"])
    canonicalization_dir = os.path.join(domain_dir, annotation_config["canonicalization_folder"])
    canonicalization_wv_folder = annotation_config["canonicalization_wv_folder"]

    vocab_filepath = os.path.join(extraction_dir, annotation_config["vocab_filename"])
    bigram_filepath = os.path.join(extraction_dir, annotation_config["bigram_filename"])
    canonicalization_annotation_dir = os.path.join(canonicalization_dir,
                                                   annotation_config["canonicalization_annotation_folder"])
    bigram_norm_candidates_filepath = os.path.join(canonicalization_dir,
                                                   annotation_config["bigram_norm_candidates_filename"])
    spelling_norm_candidates_filepath = os.path.join(canonicalization_dir,
                                                     annotation_config["spelling_norm_candidates_filename"])
    bigram_norm_filepath = os.path.join(canonicalization_dir, annotation_config["bigram_norm_filename"])

    wv_filter_min_similarity = annotation_config["wv_filter_min_similarity"]

    spark = get_spark_session("test", master_config="local[4]", log_level="WARN")

    # vocab_sdf = spark.read.csv(vocab_filepath, header=True, quote='"', escape='"', inferSchema=True)
    # bigram_sdf = spark.read.csv(bigram_filepath, header=True, quote='"', escape='"', inferSchema=True)
    # canonicalization_annotation_sdf = load_annotation(spark, canonicalization_annotation_dir,
    #                                                   annotation_config["drop_non_english"])
    #
    # get_bigram_norm_candidates(vocab_sdf, bigram_sdf, bigram_norm_candidates_filepath)
    # get_spell_norm_candidates(vocab_sdf, canonicalization_annotation_sdf, spelling_norm_candidates_filepath)

    # # ==========================================================================================================

    wv_model_filepath = os.path.join(canonicalization_dir, canonicalization_wv_folder, "model", "fasttext")
    get_bigram_norm(bigram_norm_candidates_filepath,
                    wv_model_filepath,
                    bigram_norm_filepath,
                    wv_filter_min_similarity)
