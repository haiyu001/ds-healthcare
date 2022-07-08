from typing import Dict, Any
from topic_modeling.lda.inference import extract_topics
from topic_modeling.lda_utils.pipeline_util import get_finetune_model_filepath
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger
from utils.resource_util import get_data_filepath
import argparse
import logging
import os


def build_inference(mallet_corpus_filepath: str,
                    finetune_model_dir: str,
                    lda_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build inference for corpus {mallet_corpus_filepath}\n{'*' * 150}\n")
    iterations = lda_config["iterations"]
    optimize_interval = lda_config["optimize_interval"]
    topic_alpha = lda_config["topic_alpha"]
    num_topics = lda_config["num_topics"]

    finetune_model_filepath = get_finetune_model_filepath(finetune_model_dir,
                                                          iterations,
                                                          optimize_interval,
                                                          topic_alpha,
                                                          num_topics)

    extract_topics(mallet_corpus_filepath, finetune_model_filepath)





def main(lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    topic_modeling_dir = os.path.join(domain_dir, lda_config["topic_modeling_folder"])
    corpus_dir = os.path.join(topic_modeling_dir, lda_config["corpus_folder"])
    finetune_model_dir = os.path.join(topic_modeling_dir, lda_config["finetune_model_folder"])
    topic_grouping_dendrogram_filepath = \
        os.path.join(finetune_model_dir, lda_config["topic_grouping_dendrogram_filename"])
    mallet_corpus_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_filename"])

    build_inference(mallet_corpus_filepath,
                    finetune_model_dir,
                    lda_config)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lda_conf", default="conf/lda_template.cfg", required=False)

    lda_config_filepath = parser.parse_args().lda_conf

    main(lda_config_filepath)