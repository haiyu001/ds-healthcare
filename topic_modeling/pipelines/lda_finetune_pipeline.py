from typing import Dict, Any
from topic_modeling.lda.finetuning import merge_topics, load_topic_merge_data
from topic_modeling.lda.visualization import save_lda_vis
from topic_modeling.lda_utils.train_util import get_model_folder_name, get_model_filename
from utils.config_util import read_config_to_dict
from utils.general_util import make_dir, setup_logger, split_filepath
from utils.resource_util import get_data_filepath
import argparse
import logging
import scipy
import os


def build_merge_topics(topic_merge_dendrogram_filepath: str,
                       mallet_corpus_csc_filepath: str,
                       models_dir: str,
                       finetune_model_dir: str,
                       lda_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build merge topics\n{'*' * 150}\n")
    iterations = lda_config["iterations"]
    optimize_interval = lda_config["optimize_interval"]
    topic_alpha = lda_config["topic_alpha"]
    num_topics = lda_config["num_topics"]

    model_folder_name = get_model_folder_name(iterations, optimize_interval, topic_alpha, num_topics)
    model_dir = os.path.join(models_dir, model_folder_name)
    mallet_model_filename = get_model_filename(iterations, optimize_interval, topic_alpha, num_topics)
    mallet_model_filepath = os.path.join(model_dir, mallet_model_filename)

    topic_merge_filepath = merge_topics(mallet_model_filepath,
                                        lda_config["lda_vis_topics_filename_suffix"],
                                        finetune_model_dir,
                                        topic_merge_dendrogram_filepath,
                                        lda_config["topic_merge_threshold"])

    merge_id_to_topics, merge_id_to_x_coor, merge_id_to_y_coor = load_topic_merge_data(topic_merge_filepath)

    mallet_corpus_csc = scipy.sparse.load_npz(mallet_corpus_csc_filepath)

    file_dir, file_name, _ = split_filepath(topic_merge_filepath)
    topic_merge_lda_vis_html_filepath = \
        os.path.join(file_dir, f"{file_name}_{lda_config['lda_vis_html_filename_suffix']}")
    topic_merge_lda_vis_lambdas_filepath = \
        os.path.join(file_dir, f"{file_name}_{lda_config['lda_vis_lambdas_filename_suffix']}")
    topic_merge_lda_vis_topics_filepath = \
        os.path.join(file_dir, f"{file_name}_{lda_config['lda_vis_topics_filename_suffix']}")

    save_lda_vis(mallet_corpus_csc,
                 mallet_model_filepath,
                 topic_merge_lda_vis_html_filepath,
                 topic_merge_lda_vis_lambdas_filepath,
                 topic_merge_lda_vis_topics_filepath,
                 merge_id_to_topics=merge_id_to_topics,
                 merge_id_to_x_coor=merge_id_to_x_coor,
                 merge_id_to_y_coor=merge_id_to_y_coor)


def main(lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    topic_modeling_dir = os.path.join(domain_dir, lda_config["topic_modeling_folder"])
    models_dir = make_dir(os.path.join(topic_modeling_dir, lda_config["models_folder"]))
    finetune_model_dir = make_dir(os.path.join(topic_modeling_dir, lda_config["finetune_model_folder"]))
    topic_merge_dendrogram_filepath = os.path.join(finetune_model_dir, lda_config["topic_merge_dendrogram_filename"])
    corpus_dir = os.path.join(topic_modeling_dir, lda_config["corpus_folder"])
    mallet_corpus_csc_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_csc_filename"])

    build_merge_topics(topic_merge_dendrogram_filepath,
                       mallet_corpus_csc_filepath,
                       models_dir,
                       finetune_model_dir,
                       lda_config)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lda_conf", default="conf/lda_template.cfg", required=False)

    lda_config_filepath = parser.parse_args().lda_conf

    main(lda_config_filepath)
