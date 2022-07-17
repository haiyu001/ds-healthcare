from typing import Dict, Any
from topic_modeling.topic_modeling_utils.lda_pipeline_util import get_finetune_model_filepath, get_doc_topics_infer_filepath
from topic_modeling.lda.inference import extract_doc_topics, predict_topics, extract_topics_stats
from utils.spark_util import get_spark_session, add_repo_pyfile, get_spark_master_config
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger, load_json_file
from utils.resource_util import get_data_filepath
from pyspark.sql import SparkSession
import argparse
import logging
import os


def build_doc_topics(mallet_corpus_filepath: str,
                     finetune_model_filepath: str):
    logging.info(f"\n{'*' * 150}\n* build doc topics for corpus {mallet_corpus_filepath}\n{'*' * 150}\n")
    extract_doc_topics(mallet_corpus_filepath, finetune_model_filepath)


def build_lda_inference(spark: SparkSession,
                        finetune_model_filepath: str,
                        vis_topic_id_to_org_topics_filepath: str,
                        save_folder_dir: str,
                        save_folder_name: str,
                        lda_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build LDA inference\n{'*' * 150}\n")

    doc_topics_infer_filepath = get_doc_topics_infer_filepath(finetune_model_filepath)

    vis_topic_id_to_org_topics = load_json_file(vis_topic_id_to_org_topics_filepath)

    predict_topics(spark,
                   doc_topics_infer_filepath,
                   vis_topic_id_to_org_topics,
                   lda_config["inference_threshold"],
                   save_folder_dir,
                   save_folder_name)


def build_lda_stats(spark: SparkSession,
                    lda_inference_dir: str,
                    lda_stats_filepath: str):
    logging.info(f"\n{'*' * 150}\n* build LDA stats\n{'*' * 150}\n")
    extract_topics_stats(spark,
                         lda_inference_dir,
                         lda_stats_filepath)


def main(spark: SparkSession, lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    topic_modeling_dir = os.path.join(domain_dir, lda_config["topic_modeling_folder"])
    corpus_dir = os.path.join(topic_modeling_dir, lda_config["corpus_folder"])
    finetune_model_dir = os.path.join(topic_modeling_dir, lda_config["finetune_model_folder"])
    inference_dir = os.path.join(domain_dir, lda_config["inference_folder"])
    mallet_corpus_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_filename"])
    vis_topic_id_to_org_topics_filepath = \
        os.path.join(finetune_model_dir, lda_config["vis_topic_id_to_org_topics_filename"])
    lda_inference_dir = os.path.join(inference_dir, lda_config["lda_inference_folder"])
    lda_stats_filepath = os.path.join(inference_dir, lda_config["lda_stats_filename"])

    finetune_model_filepath = get_finetune_model_filepath(finetune_model_dir,
                                                          lda_config["iterations"],
                                                          lda_config["optimize_interval"],
                                                          lda_config["topic_alpha"],
                                                          lda_config["num_topics"])

    build_doc_topics(mallet_corpus_filepath,
                     finetune_model_filepath)

    build_lda_inference(spark,
                        finetune_model_filepath,
                        vis_topic_id_to_org_topics_filepath,
                        inference_dir,
                        lda_config["lda_inference_folder"],
                        lda_config)

    build_lda_stats(spark,
                    lda_inference_dir,
                    lda_stats_filepath)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lda_conf", default="conf/lda_template.cfg", required=False)

    lda_config_filepath = parser.parse_args().lda_conf

    spark = get_spark_session("lda inference", {}, get_spark_master_config(lda_config_filepath), log_level="WARN")
    add_repo_pyfile(spark)

    main(spark, lda_config_filepath)
