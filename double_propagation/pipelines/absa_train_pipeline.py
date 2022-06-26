from typing import Dict, Any
from annotation.annotation_utils.annotator_spark_util import load_annotation
from double_propagation.absa.extractor import extract_candidates
from double_propagation.absa.grouping import build_grouping_wv_corpus, get_aspect_grouping_vecs, \
    get_opinion_grouping_vecs
from word_vector.wv_model import build_word2vec
from double_propagation.absa.ranking import load_word_to_dom_lemma_and_pos, save_aspect_ranking, save_opinion_ranking
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger, make_dir
from utils.resource_util import get_data_filepath
from utils.spark_util import get_spark_session, get_spark_master_config, add_repo_pyfile
from pyspark.sql import SparkSession
from pyspark.pandas import DataFrame
import argparse
import logging
import os


def extract_aspect_and_opinion(annotation_sdf: DataFrame,
                               aspect_candidates_filepath: str,
                               opinion_candidates_filepath: str,
                               absa_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* extract aspect and opinion candidates\n{'*' * 150}\n")
    extract_candidates(annotation_sdf,
                       aspect_candidates_filepath,
                       opinion_candidates_filepath,
                       aspect_threshold=absa_config["aspect_threshold"],
                       opinion_threshold=absa_config["aspect_threshold"],
                       sentence_filter_min_count=absa_config["sentence_filter_min_count"],
                       polarity_filter_min_ratio=absa_config["polarity_filter_min_ratio"],
                       aspect_opinion_filter_min_count=absa_config["aspect_opinion_filter_min_count"],
                       aspect_opinion_num_samples=absa_config["aspect_opinion_num_samples"],
                       max_iterations=absa_config["max_iterations"])


def rank_aspect_and_opinion(aspect_candidates_filepath: str,
                            opinion_candidates_filepath: str,
                            aspect_ranking_vecs_filepath: str,
                            opinion_ranking_vecs_filepath: str,
                            aspect_ranking_filepath: str,
                            opinion_ranking_filepath: str,
                            unigram_filepath: str,
                            phrase_filepath: str,
                            absa_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build aspect and opinion ranking \n{'*' * 150}\n")
    word_to_dom_lemma, word_to_dom_pos = load_word_to_dom_lemma_and_pos(unigram_filepath)

    save_aspect_ranking(aspect_candidates_filepath,
                        aspect_ranking_vecs_filepath,
                        aspect_ranking_filepath,
                        phrase_filepath,
                        word_to_dom_lemma,
                        word_to_dom_pos,
                        absa_config["aspect_filter_min_count"],
                        absa_config["aspect_opinion_num_samples"],
                        absa_config["noun_phrase_min_count"],
                        absa_config["noun_phrase_max_words_count"])

    save_opinion_ranking(opinion_candidates_filepath,
                         opinion_ranking_vecs_filepath,
                         opinion_ranking_filepath,
                         word_to_dom_lemma,
                         word_to_dom_pos,
                         absa_config["opinion_filter_min_count"])


def build_grouping_word2vec(annotation_sdf: DataFrame,
                            aspect_ranking_filepath: str,
                            absa_grouping_wv_corpus_filepath: str,
                            absa_grouping_wv_model_filepath: str,
                            absa_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build aspect and opinion grouping word vector\n{'*' * 150}\n")
    build_grouping_wv_corpus(annotation_sdf,
                             aspect_ranking_filepath,
                             absa_grouping_wv_corpus_filepath,
                             absa_config["lang"],
                             absa_config["spacy_package"],
                             absa_config["wv_corpus_match_lowercase"])

    build_word2vec(absa_config["wv_size"],
                   use_char_ngram=False,
                   wv_corpus_filepath=absa_grouping_wv_corpus_filepath,
                   wv_model_filepath=absa_grouping_wv_model_filepath)


def get_aspect_opinion_grouping_vecs(absa_grouping_wv_model_filepath: str,
                                     aspect_ranking_filepath: str,
                                     aspect_grouping_vecs_filepath: str,
                                     opinion_ranking_filepath: str,
                                     opinion_grouping_vecs_filepath: str):
    logging.info(f"\n{'*' * 150}\n* extract aspect and opinion grouping vecs \n{'*' * 150}\n")
    get_aspect_grouping_vecs(aspect_ranking_filepath,
                             absa_grouping_wv_model_filepath,
                             aspect_grouping_vecs_filepath)

    get_opinion_grouping_vecs(opinion_ranking_filepath,
                              absa_grouping_wv_model_filepath,
                              opinion_grouping_vecs_filepath)


def main(spark: SparkSession, absa_config_filepath: str):
    absa_config = read_config_to_dict(absa_config_filepath)
    domain_dir = get_data_filepath(absa_config["domain"])
    annotation_dir = os.path.join(domain_dir, absa_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, absa_config["extraction_folder"])
    absa_dir = os.path.join(domain_dir, absa_config["absa_folder"])
    absa_aspect_dir = make_dir(os.path.join(absa_dir, "aspect"))
    absa_opinion_dir = make_dir(os.path.join(absa_dir, "opinion"))
    absa_grouping_wv_dir = os.path.join(absa_dir, absa_config["grouping_wv_folder"])
    absa_grouping_wv_corpus_filepath = os.path.join(absa_grouping_wv_dir, absa_config["grouping_wv_corpus_filename"])
    absa_grouping_wv_model_filepath = os.path.join(absa_grouping_wv_dir, absa_config["grouping_wv_model_filename"])
    unigram_filepath = os.path.join(extraction_dir, absa_config["unigram_filename"])
    phrase_filepath = os.path.join(extraction_dir, absa_config["phrase_filename"])
    aspect_candidates_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_candidates_filename"])
    aspect_ranking_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_ranking_filename"])
    aspect_ranking_vecs_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_ranking_vecs_filename"])
    aspect_grouping_vecs_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_grouping_vecs_filename"])
    opinion_candidates_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_candidates_filename"])
    opinion_ranking_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_ranking_filename"])
    opinion_ranking_vecs_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_ranking_vecs_filename"])
    opinion_grouping_vecs_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_grouping_vecs_filename"])

    annotation_sdf = load_annotation(spark, annotation_dir, absa_config["drop_non_english"])

    # extract_aspect_and_opinion(annotation_sdf,
    #                            aspect_candidates_filepath,
    #                            opinion_candidates_filepath,
    #                            absa_config)

    rank_aspect_and_opinion(aspect_candidates_filepath,
                            opinion_candidates_filepath,
                            aspect_ranking_vecs_filepath,
                            opinion_ranking_vecs_filepath,
                            aspect_ranking_filepath,
                            opinion_ranking_filepath,
                            unigram_filepath,
                            phrase_filepath,
                            absa_config)

    build_grouping_word2vec(annotation_sdf,
                            aspect_ranking_filepath,
                            absa_grouping_wv_corpus_filepath,
                            absa_grouping_wv_model_filepath,
                            absa_config)

    get_aspect_opinion_grouping_vecs(absa_grouping_wv_model_filepath,
                                     aspect_ranking_filepath,
                                     aspect_grouping_vecs_filepath,
                                     opinion_ranking_filepath,
                                     opinion_grouping_vecs_filepath)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--absa_conf", default="conf/absa_template.cfg", required=False)

    absa_config_filepath = parser.parse_args().absa_conf

    spark = get_spark_session("annotation", {}, get_spark_master_config(absa_config_filepath), log_level="WARN")
    add_repo_pyfile(spark)

    main(spark, absa_config_filepath)
