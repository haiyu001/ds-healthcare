from typing import Dict, Any
from annotation.annotation_utils.annotator_util import get_canonicalization_nlp_model_config, get_nlp_model_config
from annotation.components.annotator import get_nlp_model, get_nlp_model_config_str, pudf_annotate, load_annotation
from annotation.components.canonicalizer import get_canonicalization
from annotation.components.canonicalizer_bigram import get_bigram_canonicalization_candidates, \
    get_bigram_canonicalization, build_canonicalization_wv_corpus
from annotation.components.canonicalizer_spell import get_spell_canonicalization_candidates, get_spell_canonicalization
from annotation.components.extractor import extract_unigram, extract_ngram, extract_phrase, extract_entity, \
    extract_umls_concept
from annotation.components.extraction_filter import filter_unigram, filter_phrase
from utils.config_util import read_config_to_dict
from word_vector.wv_model import build_word2vec
from utils.resource_util import get_data_filepath
from utils.spark_util import get_spark_session, add_repo_pyfile, write_sdf_to_dir, get_spark_master_config
from utils.general_util import setup_logger
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import argparse
import logging
import os


def load_input(spark: SparkSession,
               input_filepath: str,
               input_dir: str,
               annotation_config: Dict[str, str]) -> DataFrame:
    if input_filepath:
        logging.info(f"\n{'*' * 150}\n* load input from {input_filepath}\n{'*' * 150}\n")
        input_sdf = spark.read.text(input_filepath)
    elif input_dir:
        logging.info(f"\n{'*' * 150}\n* load input from {input_dir}\n{'*' * 150}\n")
        input_sdf = spark.read.text(os.path.join(input_dir, "*.json"))
    else:
        raise ValueError("set input_filepath or input_dir for annotation")
    num_partitions = annotation_config["num_partitions"]
    input_sdf = input_sdf.repartition(num_partitions).cache()
    return input_sdf


def build_annotation(input_sdf: DataFrame,
                     save_folder_dir: str,
                     save_folder_name: str,
                     nlp_model_config: Dict[str, Dict[str, Any]]):
    logging.info(
        f"\n{'*' * 150}\n* build annotation on {input_sdf.count()} records with following config\n{'*' * 150}\n")
    nlp_model = get_nlp_model(**nlp_model_config)
    logging.info(
        f"* nlp model config (use_gpu = {nlp_model_config['use_gpu']}):\n{get_nlp_model_config_str(nlp_model)}")
    del nlp_model
    annotation_sdf = input_sdf.select(pudf_annotate(F.col("value"), nlp_model_config))
    write_sdf_to_dir(annotation_sdf, save_folder_dir, save_folder_name, file_format="txt")


def build_extraction_and_canonicalization_candidates(canonicalization_annotation_sdf: DataFrame,
                                                     canonicalization_unigram_filepath: str,
                                                     canonicalization_bigram_filepath: str,
                                                     canonicalization_trigram_filepath: str,
                                                     bigram_canonicalization_candidates_filepath: str,
                                                     spell_canonicalization_candidates_filepath: str,
                                                     annotation_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* extract canonicalization unigram, bigram and trigram and "
                 f"build bigram & spell canonicalization candidates\n{'*' * 150}\n")
    unigram_sdf = extract_unigram(canonicalization_annotation_sdf, canonicalization_unigram_filepath)
    bigram_sdf = extract_ngram(canonicalization_annotation_sdf, canonicalization_bigram_filepath,
                               n=2, ngram_extraction_min_count=annotation_config["ngram_filter_min_count"])
    extract_ngram(canonicalization_annotation_sdf, canonicalization_trigram_filepath,
                  n=3, ngram_extraction_min_count=annotation_config["ngram_filter_min_count"])
    get_bigram_canonicalization_candidates(unigram_sdf,
                                           bigram_sdf,
                                           bigram_canonicalization_candidates_filepath)
    get_spell_canonicalization_candidates(unigram_sdf,
                                          canonicalization_annotation_sdf,
                                          spell_canonicalization_candidates_filepath,
                                          annotation_config["spell_canonicalization_suggestion_filter_min_count"])


def build_canonicalization_word2vec(canonicalization_annotation_sdf: DataFrame,
                                    bigram_canonicalization_candidates_filepath: str,
                                    wv_corpus_filepath: str,
                                    wv_model_filepath: str,
                                    canonicalization_nlp_model_config: str,
                                    annotation_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build canonicalization word vector\n{'*' * 150}\n")
    build_canonicalization_wv_corpus(canonicalization_annotation_sdf,
                                     bigram_canonicalization_candidates_filepath,
                                     wv_corpus_filepath,
                                     canonicalization_nlp_model_config["lang"],
                                     canonicalization_nlp_model_config["spacy_package"],
                                     annotation_config["wv_corpus_match_lowercase"],
                                     annotation_config["num_partitions"],)

    build_word2vec(vector_size=annotation_config["wv_size"],
                   use_char_ngram=True,
                   wv_corpus_filepath=wv_corpus_filepath,
                   wv_model_filepath=wv_model_filepath)


def build_canonicalization(spark: SparkSession,
                           canonicalization_unigram_filepath: str,
                           canonicalization_bigram_filepath: str,
                           canonicalization_trigram_filepath: str,
                           bigram_canonicalization_candidates_filepath: str,
                           spell_canonicalization_candidates_filepath: str,
                           bigram_canonicalization_filepath: str,
                           spell_canonicalization_filepath: str,
                           canonicalization_filepath: str,
                           wv_model_filepath: str,
                           annotation_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build bigram, spell, prefix, hyphen and ampersand canonicalization\n{'*' * 150}\n")
    spell_canonicalization_candidates_sdf = spark.read.csv(
        spell_canonicalization_candidates_filepath, header=True, quote='"', escape='"', inferSchema=True)
    wv_model_filepath = os.path.join(wv_model_filepath.rsplit(".", 1)[0], "fasttext")
    get_bigram_canonicalization(bigram_canonicalization_candidates_filepath,
                                wv_model_filepath,
                                bigram_canonicalization_filepath,
                                annotation_config["wv_bigram_canonicalization_filter_min_similarity"])
    get_spell_canonicalization(spell_canonicalization_candidates_sdf,
                               canonicalization_unigram_filepath,
                               wv_model_filepath,
                               spell_canonicalization_filepath,
                               annotation_config["spell_canonicalization_suggestion_filter_min_count"],
                               annotation_config["spell_canonicalization_edit_distance_filter_max_count"],
                               annotation_config["spell_canonicalization_misspelling_filter_max_percent"],
                               annotation_config["spell_canonicalization_word_pos_filter_min_percent"],
                               annotation_config["wv_spell_canonicalization_filter_min_similarity"])
    get_canonicalization(bigram_canonicalization_filepath,
                         spell_canonicalization_filepath,
                         canonicalization_unigram_filepath,
                         canonicalization_bigram_filepath,
                         canonicalization_trigram_filepath,
                         canonicalization_filepath,
                         annotation_config["conjunction_trigram_canonicalization_filter_min_count"])


def build_extraction(annotation_sdf: DataFrame,
                     unigram_filepath: str,
                     bigram_filepath: str,
                     trigram_filepath: str,
                     phrase_filepath: str,
                     entity_filepath: str,
                     umls_concept_filepath: str,
                     annotation_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* extract unigram, bigram, trigram, phrase, entity and umls_concept\n{'*' * 150}\n")
    extract_unigram(annotation_sdf, unigram_filepath)
    extract_ngram(annotation_sdf, bigram_filepath, 2, annotation_config["ngram_extraction_min_count"])
    extract_ngram(annotation_sdf, trigram_filepath, 3, annotation_config["ngram_extraction_min_count"])
    extract_phrase(annotation_sdf, phrase_filepath, annotation_config["phrase_extraction_min_count"])
    extract_entity(annotation_sdf, entity_filepath, annotation_config["entity_extraction_min_count"])
    extract_umls_concept(annotation_sdf, umls_concept_filepath, annotation_config["umls_concept_extraction_min_count"])


def filter_extraction(unigram_filepath: str,
                      phrase_filepath: str,
                      filter_unigram_filepath: str,
                      filter_phrase_filepath: str,
                      annotation_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* filter unigram, phrase\n{'*' * 150}\n")
    filter_phrase(phrase_filepath, filter_phrase_filepath, annotation_config["noun_phrase_words_max_count"])
    filter_unigram(unigram_filepath, filter_unigram_filepath,
                   annotation_config["unigram_filter_min_count"], annotation_config["stop_words_filter_min_count"])


def main(spark: SparkSession, nlp_model_config_filepath: str, annotation_config_filepath: str, ):
    # load annotation config
    annotation_config = read_config_to_dict(annotation_config_filepath)
    domain_dir = get_data_filepath(annotation_config["domain"])
    annotation_dir = os.path.join(domain_dir, annotation_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, annotation_config["extraction_folder"])
    canonicalization_dir = os.path.join(domain_dir, annotation_config["canonicalization_folder"])
    canonicalization_annotation_dir = os.path.join(
        canonicalization_dir, annotation_config["canonicalization_annotation_folder"])
    canonicalization_extraction_dir = os.path.join(
        canonicalization_dir, annotation_config["canonicalization_extraction_folder"])
    canonicalization_wv_dir = os.path.join(
        canonicalization_dir, annotation_config["canonicalization_wv_folder"])
    canonicalization_unigram_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["canonicalization_unigram_filename"])
    canonicalization_bigram_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["canonicalization_bigram_filename"])
    canonicalization_trigram_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["canonicalization_trigram_filename"])
    bigram_canonicalization_candidates_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["bigram_canonicalization_candidates_filename"])
    spell_canonicalization_candidates_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["spell_canonicalization_candidates_filename"])
    wv_corpus_filepath = os.path.join(
        canonicalization_wv_dir, annotation_config["canonicalization_wv_corpus_filename"])
    wv_model_filepath = os.path.join(
        canonicalization_wv_dir, annotation_config["canonicalization_wv_model_filename"])
    bigram_canonicalization_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["bigram_canonicalization_filename"])
    spell_canonicalization_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["spell_canonicalization_filename"])
    canonicalization_filepath = os.path.join(
        canonicalization_dir, annotation_config["canonicalization_filename"])
    unigram_filepath = os.path.join(extraction_dir, annotation_config["unigram_filename"])
    bigram_filepath = os.path.join(extraction_dir, annotation_config["bigram_filename"])
    trigram_filepath = os.path.join(extraction_dir, annotation_config["trigram_filename"])
    phrase_filepath = os.path.join(extraction_dir, annotation_config["phrase_filename"])
    entity_filepath = os.path.join(extraction_dir, annotation_config["entity_filename"])
    umls_concept_filepath = os.path.join(extraction_dir, annotation_config["umls_concept_filename"])
    filter_unigram_filepath = os.path.join(extraction_dir, annotation_config["filter_unigram_filename"])
    filter_phrase_filepath = os.path.join(extraction_dir, annotation_config["filter_phrase_filename"])
    input_filepath = annotation_config["input_filepath"]
    input_dir = annotation_config["input_dir"]

    # load nlp model config
    canonicalization_nlp_model_config = get_canonicalization_nlp_model_config(nlp_model_config_filepath)
    nlp_model_config = get_nlp_model_config(nlp_model_config_filepath, canonicalization_filepath)

    # # load input data
    # input_sdf = load_input(spark,
    #                        input_filepath,
    #                        input_dir,
    #                        annotation_config)
    #
    # # build canonicalization annotation
    # build_annotation(input_sdf,
    #                  canonicalization_dir,
    #                  annotation_config["canonicalization_annotation_folder"],
    #                  canonicalization_nlp_model_config)
    #
    # # load canonicalization annotation
    # canonicalization_annotation_sdf = load_annotation(spark,
    #                                                   canonicalization_annotation_dir,
    #                                                   annotation_config["drop_non_english"])
    #
    # # extract canonicalization unigram, bigram and trigram and build bigram & spell canonicalization candidates
    # build_extraction_and_canonicalization_candidates(canonicalization_annotation_sdf,
    #                                                  canonicalization_unigram_filepath,
    #                                                  canonicalization_bigram_filepath,
    #                                                  canonicalization_trigram_filepath,
    #                                                  bigram_canonicalization_candidates_filepath,
    #                                                  spell_canonicalization_candidates_filepath,
    #                                                  annotation_config)
    #
    # # build canonicalization word vector
    # build_canonicalization_word2vec(canonicalization_annotation_sdf,
    #                                 bigram_canonicalization_candidates_filepath,
    #                                 wv_corpus_filepath,
    #                                 wv_model_filepath,
    #                                 canonicalization_nlp_model_config,
    #                                 annotation_config)
    #
    # # build bigram, spell, prefix, hyphen and ampersand canonicalization
    # build_canonicalization(spark,
    #                        canonicalization_unigram_filepath,
    #                        canonicalization_bigram_filepath,
    #                        canonicalization_trigram_filepath,
    #                        bigram_canonicalization_candidates_filepath,
    #                        spell_canonicalization_candidates_filepath,
    #                        bigram_canonicalization_filepath,
    #                        spell_canonicalization_filepath,
    #                        canonicalization_filepath,
    #                        wv_model_filepath,
    #                        annotation_config)
    #
    # # build annotation with normalizer
    # build_annotation(input_sdf,
    #                  domain_dir,
    #                  annotation_config["annotation_folder"],
    #                  nlp_model_config)
    #
    # # load annotation
    # annotation_sdf = load_annotation(spark,
    #                                  annotation_dir,
    #                                  annotation_config["drop_non_english"])
    #
    # # extract unigram, bigram, trigram, phrase, entity and umls_concept
    # build_extraction(annotation_sdf,
    #                  unigram_filepath,
    #                  bigram_filepath,
    #                  trigram_filepath,
    #                  phrase_filepath,
    #                  entity_filepath,
    #                  umls_concept_filepath,
    #                  annotation_config)

    # filter phrase
    filter_extraction(unigram_filepath,
                      phrase_filepath,
                      filter_unigram_filepath,
                      filter_phrase_filepath,
                      annotation_config)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model_conf", default="conf/nlp_model_template.cfg", required=False)
    parser.add_argument("--annotation_conf", default="conf/annotation_template.cfg", required=False)

    nlp_model_config_filepath = parser.parse_args().nlp_model_conf
    annotation_config_filepath = parser.parse_args().annotation_conf

    spark = get_spark_session("annotation", {}, get_spark_master_config(annotation_config_filepath), log_level="WARN")
    add_repo_pyfile(spark)

    main(spark, nlp_model_config_filepath, annotation_config_filepath)
