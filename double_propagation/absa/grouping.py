from typing import Dict
from word_vector.wv_corpus import extact_wv_corpus_from_annotation
from word_vector.wv_model import build_word2vec
from pyspark.sql import DataFrame
import pandas as pd
import json
import logging


def get_aspect_match_dict(aspect_expand_rank_filepath: str) -> Dict[str, str]:
    aspect_expand_rank_pdf = pd.read_csv(aspect_expand_rank_filepath, encoding="utf-8",
                                         na_values="", keep_default_na=False)
    aspect_expand_rank_pdf["members"] = aspect_expand_rank_pdf["members"].str.lower().apply(json.loads)
    aspect_match_dict = {}
    for _, row in aspect_expand_rank_pdf.iterrows():
        text = "_".join(row["text"].lower().split())
        aspect_match_dict.update({member: text for member in row["members"] if member != text})
    noun_phrases = aspect_expand_rank_pdf["noun_phrases"].dropna().str.lower().apply(json.loads).tolist()
    noun_phrases = sum(noun_phrases, [])
    aspect_match_dict.update({noun_phrase: "_".join(noun_phrase.split()) for noun_phrase in noun_phrases})
    return aspect_match_dict


def build_aspect_grouping_wv_corpus(annotation_sdf: DataFrame,
                                    aspect_expand_rank_filepath: str,
                                    wv_corpus_filepath: str,
                                    lang: str,
                                    spacy_package: str,
                                    match_lowercase: bool):
    logging.info(f"\n{'=' * 100}\nbuild word vector corpus\n{'=' * 100}\n")
    ngram_match_dict = get_aspect_match_dict(aspect_expand_rank_filepath)
    extact_wv_corpus_from_annotation(annotation_sdf=annotation_sdf,
                                     lang=lang,
                                     spacy_package=spacy_package,
                                     wv_corpus_filepath=wv_corpus_filepath,
                                     ngram_match_dict=ngram_match_dict,
                                     match_lowercase=match_lowercase,
                                     num_partitions=4)


if __name__ == "__main__":
    from utils.general_util import setup_logger, save_pdf, make_dir, dump_json_file
    from annotation.components.annotator import load_annotation
    from utils.config_util import read_config_to_dict
    from utils.resource_util import get_repo_dir, get_data_filepath, get_model_filepath
    from utils.spark_util import get_spark_session, union_sdfs, pudf_get_most_common_text
    import os

    setup_logger()

    absa_config_filepath = os.path.join(get_repo_dir(), "double_propagation", "pipelines", "conf/absa_template.cfg")
    absa_config = read_config_to_dict(absa_config_filepath)

    domain_dir = get_data_filepath(absa_config["domain"])
    absa_dir = os.path.join(domain_dir, absa_config["absa_folder"])
    annotation_dir = os.path.join(domain_dir, absa_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, absa_config["extraction_folder"])
    absa_aspect_dir = os.path.join(absa_dir, "aspect")
    aspect_rank_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_rank_filename"])
    absa_wv_dir = os.path.join(absa_dir, absa_config["absa_wv_folder"])
    absa_wv_corpus_filepath = os.path.join(absa_wv_dir, absa_config["absa_wv_corpus_filename"])
    absa_wv_model_filepath = os.path.join(absa_wv_dir, absa_config["absa_wv_model_filename"])

    spark_cores = 4
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="INFO")

    annotation_sdf = load_annotation(spark,
                                     annotation_dir,
                                     absa_config["drop_non_english"])

    build_aspect_grouping_wv_corpus(annotation_sdf,
                                    aspect_rank_filepath,
                                    absa_wv_corpus_filepath,
                                    absa_config["lang"],
                                    absa_config["spacy_package"],
                                    absa_config["wv_corpus_match_lowercase"])

    build_word2vec(absa_config["wv_size"],
                   use_char_ngram=False,
                   wv_corpus_filepath=absa_wv_corpus_filepath,
                   wv_model_filepath=absa_wv_model_filepath)


