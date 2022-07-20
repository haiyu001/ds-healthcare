from typing import Dict, Any
from annotation.annotation_utils.annotation_util import load_annotation
from topic_modeling.bert_topic.corpus import build_bert_topic_corpus_by_annotation
from topic_modeling.topic_modeling_utils.corpus_util import get_corpus_word_match, get_corpus_noun_phrase_match_dict
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger, make_dir, load_json_file
from utils.resource_util import get_data_filepath
from utils.spark_util import get_spark_session, add_repo_pyfile, get_spark_master_config
from pyspark.sql import SparkSession, DataFrame
import argparse
import logging
import os


def build_tokenizer_creation_input(filter_unigram_filepath: str,
                                   corpus_word_match_filepath: str,
                                   filter_phrase_filepath: str,
                                   corpus_noun_phrase_match_filepath: str,
                                   bert_topic_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build tokenizer creation input data\n{'*' * 150}\n")

    get_corpus_word_match(filter_unigram_filepath,
                          corpus_word_match_filepath,
                          bert_topic_config["corpus_vocab_size"],
                          bert_topic_config["corpus_word_type_candidates"])

    get_corpus_noun_phrase_match_dict(filter_phrase_filepath,
                                      corpus_noun_phrase_match_filepath,
                                      bert_topic_config["corpus_phrase_filter_min_count"],
                                      bert_topic_config["corpus_match_lowercase"])


def build_bert_topic_corpus(annotation_sdf: DataFrame,
                            corpus_filepath: str,
                            word_match_filepath: str,
                            noun_phrase_match_dict_filepath: str,
                            bert_topic_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build bert topic corpus\n{'*' * 150}\n")

    word_match = load_json_file(word_match_filepath)
    noun_phrase_match_dict = load_json_file(noun_phrase_match_dict_filepath)

    build_bert_topic_corpus_by_annotation(annotation_sdf,
                                          corpus_filepath,
                                          word_match,
                                          noun_phrase_match_dict,
                                          bert_topic_config["lang"],
                                          bert_topic_config["spacy_package"],
                                          bert_topic_config["corpus_match_lowercase"],
                                          bert_topic_config["num_partitions"],
                                          bert_topic_config["metadata_fields_to_keep"])


def main(spark: SparkSession, bert_topic_config_filepath: str):
    bert_topic_config = read_config_to_dict(bert_topic_config_filepath)
    domain_dir = get_data_filepath(bert_topic_config["domain"])
    annotation_dir = os.path.join(domain_dir, bert_topic_config["annotation_folder"])
    topic_modeling_dir = make_dir(os.path.join(domain_dir, bert_topic_config["topic_modeling_folder"]))
    corpus_dir = make_dir(os.path.join(topic_modeling_dir, bert_topic_config["corpus_folder"]))
    corpus_filepath = os.path.join(corpus_dir, bert_topic_config["corpus_filename"])
    extraction_dir = os.path.join(domain_dir, bert_topic_config["extraction_folder"])
    filter_unigram_filepath = os.path.join(extraction_dir, bert_topic_config["filter_unigram_filename"])
    filter_phrase_filepath = os.path.join(extraction_dir, bert_topic_config["filter_phrase_filename"])
    corpus_word_match_filepath = os.path.join(corpus_dir, bert_topic_config["corpus_word_match_filename"])
    corpus_noun_phrase_match_filepath = os.path.join(corpus_dir, bert_topic_config["corpus_noun_phrase_match_filename"])

    annotation_sdf = load_annotation(spark, annotation_dir, bert_topic_config["drop_non_english"])

    build_tokenizer_creation_input(filter_unigram_filepath,
                                   corpus_word_match_filepath,
                                   filter_phrase_filepath,
                                   corpus_noun_phrase_match_filepath,
                                   bert_topic_config)

    build_bert_topic_corpus(annotation_sdf,
                            corpus_filepath,
                            corpus_word_match_filepath,
                            corpus_noun_phrase_match_filepath,
                            bert_topic_config)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_topic_conf", default="conf/bert_topic_template.cfg", required=False)

    bert_topic_config_filepath = parser.parse_args().bert_topic_conf

    spark = get_spark_session("bert topic corpus", {},
                              get_spark_master_config(bert_topic_config_filepath), log_level="WARN")
    add_repo_pyfile(spark)

    main(spark, bert_topic_config_filepath)
