from typing import List, Optional
from utils.spark_util import extract_topn_common, write_dataframe_to_file, pudf_get_most_common_text
from pyspark.sql.types import ArrayType, StringType, Row
from pyspark.sql import DataFrame, Column
from pyspark.ml.feature import NGram
import pyspark.sql.functions as F


def _udf_get_words(tokens: Column) -> Column:
    def get_words(tokens: List[Row]) -> List[str]:
        return [token.text.lower() for token in tokens]

    return F.udf(get_words, ArrayType(StringType()))(tokens)


def extract_vocab(annotation_df: DataFrame,
                  vocab_filepath: str,
                  filter_min_count: Optional[int] = None,
                  num_partitions: int = 1):
    tokens_df = annotation_df.select(F.explode(annotation_df.tokens).alias("token")).cache()
    tokens_df = tokens_df.select(F.lower(F.col("token").text).alias("word"),
                                 F.lower(F.col("token").lemma).alias("lemma"),
                                 F.col("token").pos.alias("pos"))
    count_df = tokens_df.groupby("word").agg(F.count("*").alias("count"))
    pos_df = tokens_df.groupby(["word", "pos"]).agg(F.count("*").alias("pos_count"))
    pos_df = extract_topn_common(pos_df, partition_by="word", key_by="pos", value_by="pos_count", top_n=3)
    lemma_df = tokens_df.groupby(["word", "lemma"]).agg(F.count("*").alias("lemma_count"))
    lemma_df = extract_topn_common(lemma_df, partition_by="word", key_by="lemma", value_by="lemma_count", top_n=3)
    vocab_df = count_df.join(pos_df, on="word", how="inner").join(lemma_df, on="word", how="inner")
    vocab_df = vocab_df.orderBy(F.asc("word"))
    if filter_min_count:
        vocab_df = vocab_df.filter(F.col("count") >= filter_min_count)
    write_dataframe_to_file(vocab_df, vocab_filepath, num_partitions)


def extract_ngram(annotation_df: DataFrame,
                  ngram_filepath: str, n: int,
                  filter_min_count: Optional[int] = None,
                  num_partitions: int = 1):
    tokens_df = annotation_df.select(F.col("tokens"))
    words_df = tokens_df.withColumn("words", _udf_get_words(F.col("tokens")))
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    ngram_df = ngram.transform(words_df).select("ngrams")
    ngram_df = ngram_df.select(F.explode(F.col("ngrams")).alias("ngram"))
    ngram_df = ngram_df.groupby(["ngram"]).agg(F.count("*").alias("count"))
    ngram_df = ngram_df.orderBy(F.desc("count"))
    if filter_min_count:
        ngram_df = ngram_df.filter(F.col("count") >= filter_min_count)
    write_dataframe_to_file(ngram_df, ngram_filepath, num_partitions)


def extract_phrase(annotation_df: DataFrame,
                   phrase_filepath: str,
                   filter_min_count: Optional[int] = None,
                   num_partitions: int = 1):
    phrase_df = annotation_df.select(F.explode(annotation_df._.phrases).alias("phrase"))
    phrase_df = phrase_df.select(F.lower(F.col("phrase").text).alias("phrase"),
                                 F.col("phrase").count.alias("count"),
                                 F.col("phrase").rank.alias("rank"),
                                 F.col("phrase").phrase_texts_with_ws.alias("phrase_texts_with_ws"),
                                 F.col("phrase").phrase_lemmas.alias("phrase_lemmas"),
                                 F.col("phrase").phrase_deps.alias("phrase_deps"))
    phrase_df = phrase_df.groupby(["phrase"]) \
        .agg(F.sum("count").alias("count"),
             F.mean("rank").alias("rank"),
             pudf_get_most_common_text(F.collect_list("phrase_texts_with_ws")).alias("phrase_texts_with_ws"),
             pudf_get_most_common_text(F.collect_list("phrase_lemmas")).alias("phrase_lemmas"),
             pudf_get_most_common_text(F.collect_list("phrase_deps")).alias("phrase_deps")) \
        .orderBy(F.desc("count"))
    if filter_min_count:
        phrase_df = phrase_df.filter(F.col("count") >= filter_min_count)
    write_dataframe_to_file(phrase_df, phrase_filepath, num_partitions)


def extract_entity(annotation_df: DataFrame,
                   entity_filepath: str,
                   filter_min_count: Optional[int] = None,
                   num_partitions: int = 1):
    entity_df = annotation_df.select(F.explode(annotation_df.entities).alias("entity"))
    entity_df = entity_df.select(F.lower(F.col("entity").entity).alias("text_lower"),
                                 F.col("entity").entity.alias("entity"),
                                 F.col("entity").text.alias("text"))
    entity_df = entity_df.groupby(["text_lower"]) \
        .agg(pudf_get_most_common_text(F.collect_list("entity")).alias("entity"),
             pudf_get_most_common_text(F.collect_list("text")).alias("text"),
             F.count(F.col("text")).alias("count")) \
        .orderBy(F.desc("count"))
    if filter_min_count:
        entity_df = entity_df.filter(F.col("count") >= filter_min_count)
    entity_df = entity_df.select("entity", "text", "count")
    write_dataframe_to_file(entity_df, entity_filepath, num_partitions)


if __name__ == "__main__":
    from utils.spark_util import get_spark_session
    from annotation.annotation_utils.annotation_util import load_annotation
    import os

    spark = get_spark_session("test", master_config="local[4]", log_level="WARN")
    test_dir = "/Users/haiyang/Desktop/annotation"

    annotation_dir = os.path.join(test_dir, "medium_test_annotation")
    annotation_df = load_annotation(spark, annotation_dir, drop_non_english=True)

    vocab_filepath = os.path.join(test_dir, "extraction", "vocab.csv")
    extract_vocab(annotation_df, vocab_filepath)

    ngram_filepath = os.path.join(test_dir, "extraction", "bigram.csv")
    extract_ngram(annotation_df, ngram_filepath, n=2, filter_min_count=3)

    ngram_filepath = os.path.join(test_dir, "extraction", "trigram.csv")
    extract_ngram(annotation_df, ngram_filepath, n=3, filter_min_count=3)

    phrase_filepath = os.path.join(test_dir, "extraction", "phrase.csv")
    extract_phrase(annotation_df, phrase_filepath, filter_min_count=3, num_partitions=3)

    entity_filepath = os.path.join(test_dir, "extraction", "entity.csv")
    extract_entity(annotation_df, entity_filepath, filter_min_count=3)
