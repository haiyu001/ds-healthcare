from typing import Dict, Any, Tuple, Optional
from annotation.components.annotator import load_annotation
from topic_modeling.lda.corpus import get_corpus_word_to_lemma, get_corpus_noun_phrase_match, \
    build_lda_corpus_by_annotation, save_mallet_corpus
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger, make_dir
from utils.resource_util import get_data_filepath
from utils.spark_util import get_spark_session, add_repo_pyfile, get_spark_master_config
from pyspark.sql import SparkSession, DataFrame
import argparse
import logging
import os


def build_corpus_creation_input(filter_unigram_filepath: str,
                                corpus_word_to_lemma_filepath: str,
                                filter_phrase_filepath: str,
                                corpus_noun_phrase_match_filepath: str,
                                lda_config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    logging.info(f"\n{'*' * 150}\n* build corpus creation input data\n{'*' * 150}\n")
    word_to_lemma = get_corpus_word_to_lemma(filter_unigram_filepath,
                                             corpus_word_to_lemma_filepath,
                                             lda_config["corpus_vocab_size"],
                                             lda_config["corpus_word_pos_candidates"])
    noun_phrase_match_dict = get_corpus_noun_phrase_match(filter_phrase_filepath,
                                                          corpus_noun_phrase_match_filepath,
                                                          lda_config["corpus_phrase_filter_min_count"])
    return word_to_lemma, noun_phrase_match_dict


def build_mallet_corpus(annotation_sdf: DataFrame,
                        word_to_lemma: Dict[str, str],
                        noun_phrase_match_dict: Optional[Dict[str, str]],
                        corpus_filepath: str,
                        mallet_docs_filepath: str,
                        mallet_id2word_filepath: str,
                        mallet_corpus_filepath: str,
                        mallet_corpus_csc_filepath: str,
                        mallet_vocab_filepath: str,
                        lda_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build mallet corpus\n{'*' * 150}\n")
    build_lda_corpus_by_annotation(annotation_sdf,
                                   lda_config["lang"],
                                   lda_config["spacy_package"],
                                   corpus_filepath,
                                   word_to_lemma,
                                   noun_phrase_match_dict,
                                   lda_config["corpus_match_lowercase"],
                                   lda_config["num_partitions"],
                                   lda_config["metadata_fields_to_keep"])

    save_mallet_corpus(corpus_filepath,
                       mallet_docs_filepath,
                       mallet_id2word_filepath,
                       mallet_corpus_filepath,
                       mallet_corpus_csc_filepath,
                       mallet_vocab_filepath)


def main(spark: SparkSession, lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    topic_modeling_dir = make_dir(os.path.join(domain_dir, lda_config["topic_modeling_folder"]))
    corpus_dir = make_dir(os.path.join(topic_modeling_dir, lda_config["corpus_folder"]))
    annotation_dir = os.path.join(domain_dir, lda_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, lda_config["extraction_folder"])
    filter_unigram_filepath = os.path.join(extraction_dir, lda_config["filter_unigram_filename"])
    filter_phrase_filepath = os.path.join(extraction_dir, lda_config["filter_phrase_filename"])
    corpus_word_to_lemma_filepath = os.path.join(corpus_dir, lda_config["corpus_word_to_lemma_filename"])
    corpus_noun_phrase_match_filepath = os.path.join(corpus_dir, lda_config["corpus_noun_phrase_match_filename"])
    corpus_filepath = os.path.join(corpus_dir, lda_config["corpus_filename"])
    mallet_docs_filepath = os.path.join(corpus_dir, lda_config["mallet_docs_filename"])
    mallet_id2word_filepath = os.path.join(corpus_dir, lda_config["mallet_id2word_filename"])
    mallet_corpus_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_filename"])
    mallet_corpus_csc_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_csc_filename"])
    mallet_vocab_filepath = os.path.join(corpus_dir, lda_config["mallet_vocab_filename"])

    word_to_lemma, noun_phrase_match_dict = build_corpus_creation_input(filter_unigram_filepath,
                                                                        corpus_word_to_lemma_filepath,
                                                                        filter_phrase_filepath,
                                                                        corpus_noun_phrase_match_filepath,
                                                                        lda_config)

    annotation_sdf = load_annotation(spark, annotation_dir, lda_config["drop_non_english"])

    build_lda_corpus_by_annotation(annotation_sdf,
                                   lda_config["lang"],
                                   lda_config["spacy_package"],
                                   corpus_filepath,
                                   word_to_lemma,
                                   noun_phrase_match_dict,
                                   lda_config["corpus_match_lowercase"],
                                   lda_config["num_partitions"],
                                   lda_config["metadata_fields_to_keep"])

    build_mallet_corpus(annotation_sdf,
                        word_to_lemma,
                        noun_phrase_match_dict,
                        corpus_filepath,
                        mallet_docs_filepath,
                        mallet_id2word_filepath,
                        mallet_corpus_filepath,
                        mallet_corpus_csc_filepath,
                        mallet_vocab_filepath,
                        lda_config)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lda_conf", default="conf/lda_template.cfg", required=False)

    lda_config_filepath = parser.parse_args().lda_conf

    spark = get_spark_session("annotation", {}, get_spark_master_config(lda_config_filepath), log_level="WARN")
    add_repo_pyfile(spark)

    main(spark, lda_config_filepath)
