from typing import Dict, Any
from annotation.components.annotator import load_annotation
from topic_modeling.lda.corpus import get_corpus_word_match, get_corpus_noun_phrase_match_dict, \
    build_lda_corpus_by_annotation, save_mallet_corpus, get_absa_word_match, get_absa_noun_phrase_match_dict
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger, make_dir, load_json_file
from utils.resource_util import get_data_filepath
from utils.spark_util import get_spark_session, add_repo_pyfile, get_spark_master_config
from pyspark.sql import SparkSession, DataFrame
import argparse
import logging
import os


def build_corpus_creation_input(filter_unigram_filepath: str,
                                corpus_word_match_filepath: str,
                                filter_phrase_filepath: str,
                                corpus_noun_phrase_match_filepath: str,
                                aspect_filepath: str,
                                opinion_filepath: str,
                                lda_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build corpus creation input data\n{'*' * 150}\n")
    corpus_source = lda_config["corpus_source"]
    if corpus_source == "corpus":
        get_corpus_word_match(filter_unigram_filepath,
                              corpus_word_match_filepath,
                              lda_config["corpus_vocab_size"],
                              lda_config["corpus_word_type_candidates"])
        get_corpus_noun_phrase_match_dict(filter_phrase_filepath,
                                          corpus_noun_phrase_match_filepath,
                                          lda_config["corpus_phrase_filter_min_count"],
                                          lda_config["corpus_match_lowercase"])
    elif corpus_source == "absa":
        get_absa_word_match(aspect_filepath,
                            opinion_filepath,
                            corpus_word_match_filepath,
                            lda_config["corpus_word_type_candidates"])
        get_absa_noun_phrase_match_dict(aspect_filepath,
                                        corpus_noun_phrase_match_filepath,
                                        lda_config["corpus_match_lowercase"])
    else:
        raise ValueError(f"Unsupported corpus_source of {corpus_source}")


def build_mallet_corpus(annotation_sdf: DataFrame,
                        word_match_filepath: str,
                        noun_phrase_match_dict_filepath: str,
                        corpus_filepath: str,
                        mallet_docs_filepath: str,
                        mallet_id2word_filepath: str,
                        mallet_corpus_filepath: str,
                        mallet_corpus_csc_filepath: str,
                        mallet_vocab_filepath: str,
                        lda_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build mallet corpus\n{'*' * 150}\n")

    word_match = load_json_file(word_match_filepath)
    noun_phrase_match_dict = load_json_file(noun_phrase_match_dict_filepath)

    build_lda_corpus_by_annotation(annotation_sdf,
                                   lda_config["lang"],
                                   lda_config["spacy_package"],
                                   corpus_filepath,
                                   word_match,
                                   noun_phrase_match_dict,
                                   lda_config["corpus_match_lowercase"],
                                   lda_config["num_partitions"],
                                   lda_config["metadata_fields_to_keep"])

    save_mallet_corpus(lda_config["corpus_doc_id_col"],
                       corpus_filepath,
                       mallet_docs_filepath,
                       mallet_id2word_filepath,
                       mallet_corpus_filepath,
                       mallet_corpus_csc_filepath,
                       mallet_vocab_filepath)


def main(spark: SparkSession, lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    absa_dir = os.path.join(domain_dir, lda_config["absa_folder"])
    topic_modeling_dir = make_dir(os.path.join(domain_dir, lda_config["topic_modeling_folder"]))
    corpus_dir = make_dir(os.path.join(topic_modeling_dir, lda_config["corpus_folder"]))
    annotation_dir = os.path.join(domain_dir, lda_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, lda_config["extraction_folder"])
    filter_unigram_filepath = os.path.join(extraction_dir, lda_config["filter_unigram_filename"])
    filter_phrase_filepath = os.path.join(extraction_dir, lda_config["filter_phrase_filename"])
    corpus_word_match_filepath = os.path.join(corpus_dir, lda_config["corpus_word_match_filename"])
    corpus_noun_phrase_match_filepath = os.path.join(corpus_dir, lda_config["corpus_noun_phrase_match_filename"])
    corpus_filepath = os.path.join(corpus_dir, lda_config["corpus_filename"])
    mallet_docs_filepath = os.path.join(corpus_dir, lda_config["mallet_docs_filename"])
    mallet_id2word_filepath = os.path.join(corpus_dir, lda_config["mallet_id2word_filename"])
    mallet_corpus_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_filename"])
    mallet_corpus_csc_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_csc_filename"])
    mallet_vocab_filepath = os.path.join(corpus_dir, lda_config["mallet_vocab_filename"])
    aspect_filepath = os.path.join(absa_dir, lda_config["aspect_filename"])
    opinion_filepath = os.path.join(absa_dir, lda_config["opinion_filename"])

    annotation_sdf = load_annotation(spark, annotation_dir, lda_config["drop_non_english"])

    build_corpus_creation_input(filter_unigram_filepath,
                                corpus_word_match_filepath,
                                filter_phrase_filepath,
                                corpus_noun_phrase_match_filepath,
                                aspect_filepath,
                                opinion_filepath,
                                lda_config)

    build_mallet_corpus(annotation_sdf,
                        corpus_word_match_filepath,
                        corpus_noun_phrase_match_filepath,
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
