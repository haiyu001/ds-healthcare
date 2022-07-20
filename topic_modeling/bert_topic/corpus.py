from typing import Optional, Dict
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, IntegerType
from annotation.annotation_utils.corpus_util import pudf_get_corpus_line
from utils.spark_util import write_sdf_to_file


def udf_extract_corpus_line(doc_text: Column,
                            tokens: Column,
                            sentences: Column,
                            word_match: Dict[str, str],
                            to_lower: bool = True):
    def extract_corpus_line(doc_text, tokens, sentences):
        sentence_list = []
        for sentence_id, sentence in enumerate(sentences):
            sentence_start_id, sentence_end_id = sentence["start_id"], sentence["end_id"]
            sentence_lemma_token_matches = []
            for token in tokens[sentence_start_id: sentence_end_id]:
                token_text = token.text
                if to_lower:
                    token_text = token_text.lower()
                if token_text in word_match:
                    token_match = word_match[token_text]
                    sentence_lemma_token_matches.append(token_match)
            sentence_dict = {
                "sentence_id": sentence_id,
                "sentence_text": doc_text[tokens[sentence_start_id].start_char: tokens[sentence_end_id - 1].end_char],
                "sentence_lemma": " ".join(sentence_lemma_token_matches)
            }
            sentence_list.append(sentence_dict)
        return sentence_list

    return F.udf(extract_corpus_line,
                 ArrayType(StructType([StructField("sentence_id", IntegerType()),
                                       StructField("sentence_text", StringType()),
                                       StructField("sentence_lemma", StringType())])))(doc_text, tokens, sentences)


def build_bert_topic_corpus_by_annotation(annotation_sdf: DataFrame,
                                          corpus_filepath: str,
                                          word_match: Dict[str, str],
                                          ngram_match_dict: Dict[str, str],
                                          lang: str,
                                          spacy_package: str,
                                          to_lower: bool = True,
                                          num_partitions: Optional[int] = None,
                                          metadata_fields_to_keep: Optional[str] = None):
    annotation_sdf = annotation_sdf.select(F.col("_").metadata.alias("metadata"),
                                           F.col("text").alias("doc_text"),
                                           F.col("tokens"),
                                           F.col("sentences"))
    bert_topic_corpus_sdf = annotation_sdf.select(F.col("metadata"),
                                                  udf_extract_corpus_line(F.col("doc_text"),
                                                                          F.col("tokens"),
                                                                          F.col("sentences"),
                                                                          word_match,
                                                                          to_lower).alias("sentences"))
    bert_topic_corpus_sdf = bert_topic_corpus_sdf.select(F.col("metadata"),
                                                         F.explode(F.col("sentences")).alias("sentence"))
    corpus_cols = [F.col(f"metadata.{field}").alias(field) for field in metadata_fields_to_keep.split(",")] + \
                  [F.col("sentence.sentence_id").alias("sentence_id"),
                   F.col("sentence.sentence_text").alias("sentence_text"),
                   F.col("sentence.sentence_lemma").alias("sentence_lemma")]
    bert_topic_corpus_sdf = bert_topic_corpus_sdf.select(*corpus_cols)
    bert_topic_corpus_sdf = bert_topic_corpus_sdf.withColumn(
        "sentence_lemma",
        pudf_get_corpus_line(F.col("sentence_lemma"), lang, spacy_package, ngram_match_dict))
    write_sdf_to_file(bert_topic_corpus_sdf, corpus_filepath, num_partitions)
