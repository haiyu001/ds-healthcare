from typing import Set, Dict
from pyspark.sql.types import BooleanType, ArrayType, StringType
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
import pandas as pd


def pudf_is_valid_bigram(bigram: Column, min_word_char_count: int = 2) -> Column:
    def is_valid_bigram(bigram: pd.Series) -> pd.Series:
        valid_candidate = bigram.apply(lambda x: all([len(word) >= min_word_char_count for word in x.split()]))
        return valid_candidate

    return F.pandas_udf(is_valid_bigram, BooleanType())(bigram)


def pudf_get_valid_suggestions(suggestions: Column, vocab: Set[str]) -> Column:
    def get_valid_suggestions(suggestions: pd.Series) -> pd.Series:
        valid_suggestions = suggestions.apply(lambda x: [i for i in x if i.lower() in vocab])
        return valid_suggestions

    return F.pandas_udf(get_valid_suggestions, ArrayType(StringType()))(suggestions)


def get_bigram_match_dict(bigram_norm_candidates_filepath: str) -> Dict[str, str]:
    bigram_norms_pdf = pd.read_csv(bigram_norm_candidates_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    bigrams = bigram_norms_pdf["bigram"].tolist()
    bigram_match_dict = {bigram.lower(): "_".join(bigram.strip().lower().split()) for bigram in bigrams}
    return bigram_match_dict


def get_bigram_norm_candidates(vocab_sdf: DataFrame,
                               bigram_sdf: DataFrame,
                               bigram_norm_candidates_filepath: str,
                               bigram_norm_min_word_char_count: int = 2,
                               num_partitions: int = 1):
    vocab_sdf = vocab_sdf.select(F.col("word"),
                                 F.col("count").alias("word_count"),
                                 F.col("top_three_pos").alias("word_top_three_pos"))
    bigram_sdf = bigram_sdf.select(F.regexp_replace(F.col("ngram"), " ", "").alias("word"),
                                   F.col("ngram").alias("bigram"),
                                   F.col("count").alias("bigram_count"))
    bigram_norm_candidates_sdf = vocab_sdf.join(bigram_sdf, on="word", how="inner")
    bigram_norm_candidates_sdf = bigram_norm_candidates_sdf \
        .select("word", "bigram", "word_count", "bigram_count", "word_top_three_pos")
    bigram_norm_candidates_sdf = bigram_norm_candidates_sdf
    bigram_norm_candidates_sdf = bigram_norm_candidates_sdf.filter(pudf_is_valid_bigram(F.col("bigram"),
                                                                                        bigram_norm_min_word_char_count))
    write_sdf_to_file(bigram_norm_candidates_sdf, bigram_norm_candidates_filepath, num_partitions)


def get_spell_check_candidates(vocab_sdf: DataFrame,
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


if __name__ == "__main__":
    from annotation.annotation_utils.annotation_util import read_annotation_config, load_annotation
    from utils.resource_util import get_data_filepath, get_repo_dir
    from utils.spark_util import get_spark_session, write_sdf_to_file
    import os

    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)

    domain_dir = get_data_filepath(annotation_config["domain"])
    extraction_folder = annotation_config["extraction_folder"]
    canonicalization_folder = annotation_config["canonicalization_folder"]

    annotation_dir = os.path.join(domain_dir, annotation_config["annotation_folder"])
    vocab_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["vocab_filename"])
    bigram_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["bigram_filename"])
    bigram_norm_candidates_filepath = os.path.join(domain_dir, canonicalization_folder,
                                                   annotation_config["bigram_norm_candidates_filename"])
    spell_check_candidates_filepath = os.path.join(domain_dir, canonicalization_folder,
                                                   annotation_config["spell_check_candidates_filename"])

    spark = get_spark_session("test", master_config="local[4]", log_level="WARN")

    vocab_sdf = spark.read.csv(vocab_filepath, header=True, quote='"', escape='"', inferSchema=True)
    bigram_sdf = spark.read.csv(bigram_filepath, header=True, quote='"', escape='"', inferSchema=True)
    bigram_norm_min_word_char_count = annotation_config["bigram_norm_min_word_char_count"]
    get_bigram_norm_candidates(vocab_sdf, bigram_sdf, bigram_norm_candidates_filepath, bigram_norm_min_word_char_count)

    annotation_sdf = load_annotation(spark, annotation_dir, annotation_config["drop_non_english"])
    get_spell_check_candidates(vocab_sdf, annotation_sdf, spell_check_candidates_filepath)
