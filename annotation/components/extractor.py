from typing import List, Optional
from utils.general_util import dump_json_file
from utils.spark_util import extract_topn_common, write_sdf_to_file, pudf_get_most_common_text
from pyspark.sql.types import ArrayType, StringType, Row, BooleanType
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
from pyspark.ml.feature import NGram
from string import punctuation
import pandas as pd
import collections


def pudf_is_valid_ngram(ngrams: Column) -> Column:
    def is_valid_ngram(ngrams: pd.Series) -> pd.Series:
        ngrams_words = ngrams.apply(lambda x: x.split())
        is_valid = ngrams_words.apply(lambda x: x[0] not in punctuation and x[-1] not in punctuation)
        return is_valid

    return F.pandas_udf(is_valid_ngram, BooleanType())(ngrams)


def udf_get_words(tokens: Column) -> Column:
    def get_words(tokens: List[Row]) -> List[str]:
        return [token.text.lower() for token in tokens]

    return F.udf(get_words, ArrayType(StringType()))(tokens)


def extract_unigram(annotation_sdf: DataFrame,
                    unigram_filepath: str,
                    num_partitions: Optional[int] = None) -> DataFrame:
    tokens_sdf = annotation_sdf.select(F.explode(annotation_sdf.tokens).alias("token")).cache()
    tokens_sdf = tokens_sdf.select(F.lower(F.col("token").text).alias("word"),
                                   F.lower(F.col("token").lemma).alias("lemma"),
                                   F.col("token").pos.alias("pos"))
    count_sdf = tokens_sdf.groupby("word").agg(F.count("*").alias("count"))
    pos_sdf = tokens_sdf.groupby(["word", "pos"]).agg(F.count("*").alias("pos_count"))
    pos_sdf = extract_topn_common(pos_sdf, partition_by="word", key_by="pos", value_by="pos_count", topn=3)
    pos_sdf = pos_sdf.withColumnRenamed("pos", "top_three_pos")
    lemma_sdf = tokens_sdf.groupby(["word", "lemma"]).agg(F.count("*").alias("lemma_count"))
    lemma_sdf = extract_topn_common(lemma_sdf, partition_by="word", key_by="lemma", value_by="lemma_count", topn=3)
    lemma_sdf = lemma_sdf.withColumnRenamed("lemma", "top_three_lemma")
    unigram_sdf = count_sdf.join(pos_sdf, on="word", how="inner").join(lemma_sdf, on="word", how="inner")
    unigram_sdf = unigram_sdf.orderBy(F.asc("word"))
    write_sdf_to_file(unigram_sdf, unigram_filepath, num_partitions)
    return unigram_sdf


def extract_ngram(annotation_sdf: DataFrame,
                  ngram_filepath: str, n: int,
                  ngram_filter_min_count: Optional[int] = None,
                  num_partitions: Optional[int] = None) -> DataFrame:
    tokens_sdf = annotation_sdf.select(F.col("tokens"))
    words_sdf = tokens_sdf.withColumn("words", udf_get_words(F.col("tokens")))
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    ngram_sdf = ngram.transform(words_sdf).select("ngrams")
    ngram_sdf = ngram_sdf.select(F.explode(F.col("ngrams")).alias("ngram"))
    ngram_sdf = ngram_sdf.groupby(["ngram"]).agg(F.count("*").alias("count"))
    ngram_sdf = ngram_sdf.orderBy(F.desc("count"))
    ngram_sdf = ngram_sdf.filter(pudf_is_valid_ngram(F.col("ngram")))
    if ngram_filter_min_count:
        ngram_sdf = ngram_sdf.filter(F.col("count") >= ngram_filter_min_count)
    write_sdf_to_file(ngram_sdf, ngram_filepath, num_partitions)
    return ngram_sdf


def extract_phrase(annotation_sdf: DataFrame,
                   phrase_filepath: str,
                   phrase_filter_min_count: Optional[int] = None,
                   num_partitions: Optional[int] = None):
    phrase_sdf = annotation_sdf.select(F.explode(annotation_sdf._.phrases).alias("phrase"))
    phrase_sdf = phrase_sdf.select(F.lower(F.col("phrase").text).alias("phrase"),
                                   F.col("phrase").phrase_count.alias("count"),
                                   F.col("phrase").phrase_rank.alias("rank"),
                                   F.to_json(F.col("phrase").phrase_poses).alias("phrase_poses"),
                                   F.to_json(F.col("phrase").phrase_lemmas).alias("phrase_lemmas"),
                                   F.to_json(F.col("phrase").phrase_deps).alias("phrase_deps"))
    phrase_sdf = phrase_sdf.groupby(["phrase"]) \
        .agg(F.sum("count").alias("count"),
             F.mean("rank").alias("rank"),
             pudf_get_most_common_text(F.collect_list("phrase_poses")).alias("phrase_poses"),
             pudf_get_most_common_text(F.collect_list("phrase_lemmas")).alias("phrase_lemmas"),
             pudf_get_most_common_text(F.collect_list("phrase_deps")).alias("phrase_deps")) \
        .orderBy(F.desc("count"))
    if phrase_filter_min_count:
        phrase_sdf = phrase_sdf.filter(F.col("count") >= phrase_filter_min_count)
    write_sdf_to_file(phrase_sdf, phrase_filepath, num_partitions)


def extract_entity(annotation_sdf: DataFrame,
                   entity_filepath: str,
                   entity_filter_min_count: Optional[int] = None,
                   num_partitions: Optional[int] = None):
    entity_sdf = annotation_sdf.select(F.explode(annotation_sdf.entities).alias("entity"))
    entity_sdf = entity_sdf.select(F.lower(F.col("entity").text).alias("text_lower"),
                                   F.col("entity").entity.alias("entity"),
                                   F.col("entity").text.alias("text"),
                                   F.col("entity").negation.alias("negation"))
    entity_sdf = entity_sdf.groupby(["text_lower"]) \
        .agg(pudf_get_most_common_text(F.collect_list("entity")).alias("entity"),
             pudf_get_most_common_text(F.collect_list("text")).alias("text"),
             F.count(F.col("text")).alias("count"),
             F.sum(F.col("negation").cast("int")).alias("negation_count")) \
        .orderBy(F.asc("entity"), F.desc("count"))
    if entity_filter_min_count:
        entity_sdf = entity_sdf.filter(F.col("count") >= entity_filter_min_count)
    entity_sdf = entity_sdf.select("text", "count", "negation_count", "entity")
    write_sdf_to_file(entity_sdf, entity_filepath, num_partitions)


def extract_umls_concept(annotation_sdf: DataFrame,
                         umls_concept_filepath: str,
                         umls_concept_filter_min_count: Optional[int] = None,
                         num_partitions: Optional[int] = None):
    umls_concept_sdf = annotation_sdf.select(F.explode(annotation_sdf._.umls_concepts).alias("umls_concept"))
    umls_concept_sdf = umls_concept_sdf.select(F.lower(F.col("umls_concept").text).alias("text_lower"),
                                               F.col("umls_concept").concepts.alias("concepts"),
                                               F.col("umls_concept").text.alias("text"),
                                               F.col("umls_concept").negation.alias("negation"))
    umls_concept_sdf = umls_concept_sdf.groupby(["text_lower"]) \
        .agg(pudf_get_most_common_text(F.collect_list("text")).alias("text"),
             F.to_json(F.first(F.col("concepts"))).alias("concepts"),
             F.count(F.col("text")).alias("count"),
             F.sum(F.col("negation").cast("int")).alias("negation_count"))\
        .orderBy(F.desc("count"))
    if umls_concept_filter_min_count:
        umls_concept_sdf = umls_concept_sdf.filter(F.col("count") >= umls_concept_filter_min_count)
    umls_concept_sdf = umls_concept_sdf.select("text", "count", "negation_count", "concepts")
    write_sdf_to_file(umls_concept_sdf, umls_concept_filepath, num_partitions)


def _get_multi_noun_or_propn_ids(pos_list: List[str], max_sequence_count: int) -> List[Tuple[int, int]]:
    res = []
    start = -1
    for i, pos in enumerate(pos_list):
        if pos == "NOUN" or pos == "PROPN":
            if start >= 0:
                continue
            else:
                start = i
        else:
            if start >= 0 and 2 <= i - start <= max_sequence_count:
                res.append((start, i))
            start = -1
    return res


def filter_phrase(phrase_filepath: str,
                  filtered_phrase_filepath: str,
                  phrase_words_max_count: int = 4,
                  filter_min_count: int = 5):
    phrase_pdf = pd.read_csv(phrase_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    phrase_pdf["phrase_words"] = phrase_pdf["phrase"].str.split()
    phrase_pdf["phrase_poses"] = phrase_pdf["phrase_poses"].apply(json.loads)
    phrase_pdf = phrase_pdf[phrase_pdf["count"] >= filter_min_count]
    phrase_to_count = collections.defaultdict(lambda: int)
    for _, row in phrase_pdf.iterrows():
        phrase_words, phrase_poses, count = phrase_pdf["phrase_words"], phrase_pdf["phrase_poses"], phrase_pdf["count"]
        for start, end in _get_multi_noun_or_propn_ids(phrase_poses, phrase_words_max_count):
            phrase_to_count[" ".join(phrase_words[start: end])] += count
    phrase_to_count = dict(sorted(phrase_to_count.items(), key=lambda item: item[1]))
    dump_json_file(phrase_to_count, filtered_phrase_filepath)