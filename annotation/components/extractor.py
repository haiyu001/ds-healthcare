from typing import List, Optional
from utils.spark_util import extract_topn_common, write_sdf_to_file, pudf_get_most_common_text
from pyspark.sql.types import ArrayType, StringType, Row, BooleanType
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
from pyspark.ml.feature import NGram
from string import punctuation
import pandas as pd


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
                  ngram_extraction_min_count: Optional[int] = None,
                  num_partitions: Optional[int] = None) -> DataFrame:
    tokens_sdf = annotation_sdf.select(F.col("tokens"))
    words_sdf = tokens_sdf.withColumn("words", udf_get_words(F.col("tokens")))
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    ngram_sdf = ngram.transform(words_sdf).select("ngrams")
    ngram_sdf = ngram_sdf.select(F.explode(F.col("ngrams")).alias("ngram"))
    ngram_sdf = ngram_sdf.groupby(["ngram"]).agg(F.count("*").alias("count"))
    ngram_sdf = ngram_sdf.orderBy(F.desc("count"))
    ngram_sdf = ngram_sdf.filter(pudf_is_valid_ngram(F.col("ngram")))
    if ngram_extraction_min_count:
        ngram_sdf = ngram_sdf.filter(F.col("count") >= ngram_extraction_min_count)
    write_sdf_to_file(ngram_sdf, ngram_filepath, num_partitions)
    return ngram_sdf


def extract_phrase(annotation_sdf: DataFrame,
                   phrase_filepath: str,
                   phrase_extraction_min_count: Optional[int] = None,
                   num_partitions: Optional[int] = None):
    phrase_sdf = annotation_sdf.select(F.explode(annotation_sdf._.phrases).alias("phrase"))
    phrase_sdf = phrase_sdf.select(F.lower(F.col("phrase").text).alias("phrase_lower"),
                                   F.col("phrase").text.alias("text"),
                                   F.to_json(F.col("phrase").phrase_words).alias("phrase_words"),
                                   F.to_json(F.col("phrase").phrase_poses).alias("phrase_poses"),
                                   F.to_json(F.col("phrase").phrase_lemmas).alias("phrase_lemmas"),
                                   F.to_json(F.col("phrase").phrase_deps).alias("phrase_deps"))
    phrase_sdf = phrase_sdf.groupby(["phrase_lower"]).agg(
        pudf_get_most_common_text(F.collect_list("text")).alias("text"),
        F.count(F.col("text")).alias("count"),
        pudf_get_most_common_text(F.collect_list("phrase_words")).alias("phrase_words"),
        pudf_get_most_common_text(F.collect_list("phrase_poses")).alias("phrase_poses"),
        pudf_get_most_common_text(F.collect_list("phrase_lemmas")).alias("phrase_lemmas"),
        pudf_get_most_common_text(F.collect_list("phrase_deps")).alias("phrase_deps")) \
        .orderBy(F.desc("count"))
    if phrase_extraction_min_count:
        phrase_sdf = phrase_sdf.filter(F.col("count") >= phrase_extraction_min_count)
    phrase_sdf = phrase_sdf.select("text", "count", "phrase_words", "phrase_poses", "phrase_lemmas", "phrase_deps")
    write_sdf_to_file(phrase_sdf, phrase_filepath, num_partitions)


def extract_entity(annotation_sdf: DataFrame,
                   entity_filepath: str,
                   entity_extraction_min_count: Optional[int] = None,
                   num_partitions: Optional[int] = None):
    entity_sdf = annotation_sdf.select(F.explode(annotation_sdf.entities).alias("entity"))
    entity_data = entity_sdf.select(F.col("entity")).schema.jsonValue()
    entity_data_fields = [field["name"] for field in entity_data["fields"][0]["type"]["fields"]]
    select_columns = [
        F.lower(F.col("entity").text).alias("text_lower"),
        F.col("entity").entity.alias("entity"),
        F.col("entity").text.alias("text"),
    ]
    agg_columns = [
        pudf_get_most_common_text(F.collect_list("text")).alias("text"),
        pudf_get_most_common_text(F.collect_list("entity")).alias("entity"),
        F.count(F.col("text")).alias("count"),
    ]
    if "negation" in entity_data_fields:
        select_columns.append(F.col("entity").negation.alias("negation"))
        agg_columns.append(F.sum(F.col("negation").cast("int")).alias("negation_count"))

    entity_sdf = entity_sdf.select(*select_columns)
    entity_sdf = entity_sdf.groupby(["text_lower"]).agg(*agg_columns).orderBy(F.asc("entity"), F.desc("count"))
    if entity_extraction_min_count:
        entity_sdf = entity_sdf.filter(F.col("count") >= entity_extraction_min_count)
    entity_sdf = entity_sdf.select("text", "entity", "count", "negation_count")
    write_sdf_to_file(entity_sdf, entity_filepath, num_partitions)


def extract_umls_concept(annotation_sdf: DataFrame,
                         umls_concept_filepath: str,
                         umls_concept_extraction_min_count: Optional[int] = None,
                         num_partitions: Optional[int] = None):
    umls_concept_sdf = annotation_sdf.select(F.explode(annotation_sdf._.umls_concepts).alias("umls_concept"))
    umls_concept_data = umls_concept_sdf.select(F.col("umls_concept")).schema.jsonValue()
    umls_concept_data_fields = [field["name"] for field in umls_concept_data["fields"][0]["type"]["fields"]]
    select_columns = [
        F.lower(F.col("umls_concept").text).alias("text_lower"),
        F.col("umls_concept").concepts.alias("concepts"),
        F.col("umls_concept").text.alias("text"),
    ]
    agg_columns = [
        pudf_get_most_common_text(F.collect_list("text")).alias("text"),
        F.to_json(F.first(F.col("concepts"))).alias("concepts"),
        F.count(F.col("text")).alias("count"),
    ]
    if "negation" in umls_concept_data_fields:
        select_columns.append(F.col("umls_concept").negation.alias("negation"))
        agg_columns.append(F.sum(F.col("negation").cast("int")).alias("negation_count"))

    umls_concept_sdf = umls_concept_sdf.select(*select_columns)
    umls_concept_sdf = umls_concept_sdf.groupby(["text_lower"]).agg(*agg_columns).orderBy(F.desc("count"))
    if umls_concept_extraction_min_count:
        umls_concept_sdf = umls_concept_sdf.filter(F.col("count") >= umls_concept_extraction_min_count)
    umls_concept_sdf = umls_concept_sdf.select("text", "count", "negation_count", "concepts")
    write_sdf_to_file(umls_concept_sdf, umls_concept_filepath, num_partitions)
