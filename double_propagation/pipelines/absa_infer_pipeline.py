from typing import Dict, Any
from annotation.components.annotator import load_annotation
from double_propagation.absa.triplet_inference import extract_triplet, extract_triplet_stats
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger
from utils.resource_util import get_data_filepath
from utils.spark_util import get_spark_session, get_spark_master_config, add_repo_pyfile
from pyspark.sql import SparkSession
from pyspark.pandas import DataFrame
import argparse
import logging
import os


def build_absa_inference(annotation_sdf: DataFrame,
                         aspect_filepath: str,
                         opinion_filepath: str,
                         save_folder_dir: str,
                         save_folder_name: str,
                         absa_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* run absa inference\n{'*' * 150}\n")
    extract_triplet(annotation_sdf,
                    aspect_filepath,
                    opinion_filepath,
                    save_folder_dir,
                    save_folder_name,
                    absa_config["lang"],
                    absa_config["spacy_package"],
                    absa_config["social"],
                    absa_config["intensifier_negation_max_distance"],
                    absa_config["cap_scalar"],
                    absa_config["neg_scalar"],
                    absa_config["metadata_fields_to_keep"],
                    absa_config["infer_aspect_without_opinion"])


def build_absa_stats(spark: SparkSession,
                     absa_inference_dir: str,
                     opinion_filepath: str,
                     aspect_stats_filepath: str,
                     opinion_stats_filepath: str):
    logging.info(f"\n{'*' * 150}\n* run absa stats\n{'*' * 150}\n")
    extract_triplet_stats(spark,
                          absa_inference_dir,
                          opinion_filepath,
                          aspect_stats_filepath,
                          opinion_stats_filepath, )


def main(spark: SparkSession, absa_config_filepath: str):
    absa_config = read_config_to_dict(absa_config_filepath)
    domain_dir = get_data_filepath(absa_config["domain"])
    annotation_dir = os.path.join(domain_dir, absa_config["annotation_folder"])
    absa_dir = os.path.join(domain_dir, absa_config["absa_folder"])
    inference_dir = os.path.join(domain_dir, absa_config["inference_folder"])
    aspect_filepath = os.path.join(absa_dir, absa_config["aspect_filename"])
    opinion_filepath = os.path.join(absa_dir, absa_config["opinion_filename"])
    absa_inference_dir = os.path.join(inference_dir, absa_config["absa_inference_folder"])
    aspect_stats_filepath = os.path.join(inference_dir, absa_config["aspect_stats_filename"])
    opinion_stats_filepath = os.path.join(inference_dir, absa_config["opinion_stats_filename"])

    annotation_sdf = load_annotation(spark, annotation_dir, absa_config["drop_non_english"])

    build_absa_inference(annotation_sdf,
                         aspect_filepath,
                         opinion_filepath,
                         inference_dir,
                         absa_config["absa_inference_folder"],
                         absa_config)

    build_absa_stats(spark,
                     absa_inference_dir,
                     opinion_filepath,
                     aspect_stats_filepath,
                     opinion_stats_filepath)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--absa_conf", default="conf/absa_template.cfg", required=False)

    absa_config_filepath = parser.parse_args().absa_conf

    spark = get_spark_session("annotation", {}, get_spark_master_config(absa_config_filepath), log_level="WARN")
    add_repo_pyfile(spark)

    main(spark, absa_config_filepath)
