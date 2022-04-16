from typing import List, Optional
from utils.spark_util import extract_topn_common, write_sdf_to_file, pudf_get_most_common_text
from pyspark.sql.types import ArrayType, StringType, Row
from pyspark.sql import DataFrame, Column
from pyspark.ml.feature import NGram
import pyspark.sql.functions as F


def _udf_get_words(tokens: Column) -> Column:
    def get_words(tokens: List[Row]) -> List[str]:
        return [token.text.lower() for token in tokens]

    return F.udf(get_words, ArrayType(StringType()))(tokens)


def extract_vocab(annotation_sdf: DataFrame,
                  vocab_filepath: str,
                  filter_min_count: Optional[int] = None,
                  num_partitions: int = 1):
    tokens_sdf = annotation_sdf.select(F.explode(annotation_sdf.tokens).alias("token")).cache()
    tokens_sdf = tokens_sdf.select(F.lower(F.col("token").text).alias("word"),
                                   F.lower(F.col("token").lemma).alias("lemma"),
                                   F.col("token").pos.alias("pos"))
    count_sdf = tokens_sdf.groupby("word").agg(F.count("*").alias("count"))
    pos_sdf = tokens_sdf.groupby(["word", "pos"]).agg(F.count("*").alias("pos_count"))
    pos_sdf = extract_topn_common(pos_sdf, partition_by="word", key_by="pos", value_by="pos_count", top_n=3)
    pos_sdf = pos_sdf.withColumnRenamed("pos", "top_three_pos")
    lemma_sdf = tokens_sdf.groupby(["word", "lemma"]).agg(F.count("*").alias("lemma_count"))
    lemma_sdf = extract_topn_common(lemma_sdf, partition_by="word", key_by="lemma", value_by="lemma_count", top_n=3)
    lemma_sdf = lemma_sdf.withColumnRenamed("lemma", "top_three_lemma")
    vocab_sdf = count_sdf.join(pos_sdf, on="word", how="inner").join(lemma_sdf, on="word", how="inner")
    vocab_sdf = vocab_sdf.orderBy(F.asc("word"))
    if filter_min_count:
        vocab_sdf = vocab_sdf.filter(F.col("count") >= filter_min_count)
    write_sdf_to_file(vocab_sdf, vocab_filepath, num_partitions)


def extract_ngram(annotation_sdf: DataFrame,
                  ngram_filepath: str, n: int,
                  filter_min_count: Optional[int] = None,
                  num_partitions: int = 1):
    tokens_sdf = annotation_sdf.select(F.col("tokens"))
    words_sdf = tokens_sdf.withColumn("words", _udf_get_words(F.col("tokens")))
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    ngram_sdf = ngram.transform(words_sdf).select("ngrams")
    ngram_sdf = ngram_sdf.select(F.explode(F.col("ngrams")).alias("ngram"))
    ngram_sdf = ngram_sdf.groupby(["ngram"]).agg(F.count("*").alias("count"))
    ngram_sdf = ngram_sdf.orderBy(F.desc("count"))
    if filter_min_count:
        ngram_sdf = ngram_sdf.filter(F.col("count") >= filter_min_count)
    write_sdf_to_file(ngram_sdf, ngram_filepath, num_partitions)


def extract_phrase(annotation_sdf: DataFrame,
                   phrase_filepath: str,
                   filter_min_count: Optional[int] = None,
                   num_partitions: int = 1):
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
    if filter_min_count:
        phrase_sdf = phrase_sdf.filter(F.col("count") >= filter_min_count)
    write_sdf_to_file(phrase_sdf, phrase_filepath, num_partitions)


def extract_entity(annotation_sdf: DataFrame,
                   entity_filepath: str,
                   filter_min_count: Optional[int] = None,
                   num_partitions: int = 1):
    entity_sdf = annotation_sdf.select(F.explode(annotation_sdf.entities).alias("entity"))
    entity_sdf = entity_sdf.select(F.lower(F.col("entity").text).alias("text_lower"),
                                   F.col("entity").entity.alias("entity"),
                                   F.col("entity").text.alias("text"))
    entity_sdf = entity_sdf.groupby(["text_lower"]) \
        .agg(pudf_get_most_common_text(F.collect_list("entity")).alias("entity"),
             pudf_get_most_common_text(F.collect_list("text")).alias("text"),
             F.count(F.col("text")).alias("count")) \
        .orderBy(F.desc("count"))
    if filter_min_count:
        entity_sdf = entity_sdf.filter(F.col("count") >= filter_min_count)
    entity_sdf = entity_sdf.select("text", "entity", "count")
    write_sdf_to_file(entity_sdf, entity_filepath, num_partitions)


def extract_umls_concept(annotation_sdf: DataFrame,
                         umls_concept_filepath: str,
                         filter_min_count: Optional[int] = None,
                         num_partitions: int = 1):
    umls_concept_sdf = annotation_sdf.select(F.explode(annotation_sdf._.umls_concepts).alias("umls_concept"))
    umls_concept_sdf = umls_concept_sdf.select(F.lower(F.col("umls_concept").text).alias("text_lower"),
                                               F.col("umls_concept").concepts.alias("concepts"),
                                               F.col("umls_concept").text.alias("text"))
    umls_concept_sdf = umls_concept_sdf.groupby(["text_lower"]) \
        .agg(pudf_get_most_common_text(F.collect_list("text")).alias("text"),
             F.to_json(F.first(F.col("concepts"))).alias("concepts"),
             F.count(F.col("text")).alias("count")) \
        .orderBy(F.desc("count"))
    if filter_min_count:
        umls_concept_sdf = umls_concept_sdf.filter(F.col("count") >= filter_min_count)
    umls_concept_sdf = umls_concept_sdf.select("text", "count", "concepts")
    write_sdf_to_file(umls_concept_sdf, umls_concept_filepath, num_partitions)


if __name__ == "__main__":
    from utils.spark_util import get_spark_session
    from utils.resource_util import get_data_filepath, get_repo_dir
    from annotation.annotation_utils.annotation_util import load_annotation, read_annotation_config
    import os

    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)

    domain_dir = get_data_filepath(annotation_config["domain"])
    extraction_folder = annotation_config["extraction_folder"]

    # config_updates = {"spark.archives": "/Users/haiyang/github/datascience.tar.gz"}
    spark = get_spark_session("test", config_updates={}, master_config="local[4]", log_level="WARN")

    # load annotation
    annotation_dir = os.path.join(domain_dir, annotation_config["annotation_folder"])
    annotation_sdf = load_annotation(spark, annotation_dir, annotation_config["drop_non_english"])

    # # extract vocab
    # vocab_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["vocab_filename"])
    # extract_vocab(annotation_sdf, vocab_filepath)

    # # extract bigram
    # bigram_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["bigram_filename"])
    # extract_ngram(annotation_sdf, bigram_filepath, n=2, filter_min_count=3)

    # # extract trigram
    # trigram_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["trigram_filename"])
    # extract_ngram(annotation_sdf, trigram_filepath, n=3, filter_min_count=3)

    # # extract phrase
    # phrase_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["phrase_filename"])
    # extract_phrase(annotation_sdf, phrase_filepath, filter_min_count=3, num_partitions=3)

    # # extract entity
    # entity_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["entity_filename"])
    # extract_entity(annotation_sdf, entity_filepath, filter_min_count=1)

    # extract umls_concept
    umls_concept_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["umls_concept_filename"])
    extract_umls_concept(annotation_sdf, umls_concept_filepath, filter_min_count=1)
