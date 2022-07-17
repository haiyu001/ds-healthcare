from typing import Dict, Any
from topic_modeling.lda.finetuning import topic_merging, load_topic_merging_data, topic_grouping
from topic_modeling.lda.mallet_wrapper import LdaMallet
from topic_modeling.lda.visualization import save_lda_vis
from topic_modeling.topic_modeling_utils.lda_pipeline_util import get_model_folder_name, get_model_filename, \
    get_prefix_by_mallet_model_filepath, get_finetune_model_filepath
from utils.config_util import read_config_to_dict
from utils.general_util import make_dir, setup_logger, split_filepath
from utils.resource_util import get_data_filepath
import shutil
import argparse
import logging
import scipy
import os


def setup_finetune_model_dir(candidate_models_dir: str,
                             finetune_model_dir: str,
                             lda_config: Dict[str, Any]) -> str:
    if os.path.exists(finetune_model_dir):
        shutil.rmtree(finetune_model_dir)
    make_dir(finetune_model_dir)

    iterations = lda_config["iterations"]
    optimize_interval = lda_config["optimize_interval"]
    topic_alpha = lda_config["topic_alpha"]
    num_topics = lda_config["num_topics"]

    logging.info(f"\n{'*' * 150}\n* setup finetune model directory for parameters: "
                 f"iterations({iterations}) optimize_interval({optimize_interval}) "
                 f"topic_alpha({topic_alpha}) num_topics({num_topics})\n{'*' * 150}\n")

    model_folder_name = get_model_folder_name(iterations, optimize_interval, topic_alpha, num_topics)
    src_model_dir = os.path.join(candidate_models_dir, model_folder_name)
    dest_model_dir = os.path.join(finetune_model_dir, model_folder_name)
    shutil.copytree(src_model_dir, dest_model_dir)

    finetune_model_filepath = get_finetune_model_filepath(finetune_model_dir,
                                                          iterations,
                                                          optimize_interval,
                                                          topic_alpha,
                                                          num_topics)

    finetune_model_prefix = get_prefix_by_mallet_model_filepath(finetune_model_filepath)
    LdaMallet.update_prefix(finetune_model_filepath, finetune_model_prefix)
    return finetune_model_filepath


def build_topic_merging(topic_merging_dendrogram_filepath: str,
                        finetune_model_filepath: str,
                        finetune_model_dir: str,
                        lda_config: Dict[str, Any]) -> str:
    logging.info(f"\n{'*' * 150}\n* build topics merging\n{'*' * 150}\n")

    topic_merging_filepath = topic_merging(finetune_model_filepath,
                                           lda_config["lda_vis_topics_filename_suffix"],
                                           finetune_model_dir,
                                           topic_merging_dendrogram_filepath,
                                           lda_config["topic_merging_threshold"])
    return topic_merging_filepath


def build_topic_merging_vis(topic_merging_filepath: str,
                            finetune_model_filepath: str,
                            mallet_corpus_csc_filepath: str,
                            lda_config: [str, Any]) -> str:
    logging.info(f"\n{'*' * 150}\n* build topics merging visualization\n{'*' * 150}\n")
    merge_id_to_topics, merge_id_to_x_coor, merge_id_to_y_coor = load_topic_merging_data(topic_merging_filepath)

    mallet_corpus_csc = scipy.sparse.load_npz(mallet_corpus_csc_filepath)

    file_dir, file_name, _ = split_filepath(topic_merging_filepath)
    topic_merging_lda_vis_html_filepath = \
        os.path.join(file_dir, f"{file_name}_{lda_config['lda_vis_html_filename_suffix']}")
    topic_merging_lda_vis_lambdas_filepath = \
        os.path.join(file_dir, f"{file_name}_{lda_config['lda_vis_lambdas_filename_suffix']}")
    topic_merging_lda_vis_topics_filepath = \
        os.path.join(file_dir, f"{file_name}_{lda_config['lda_vis_topics_filename_suffix']}")

    save_lda_vis(mallet_corpus_csc,
                 finetune_model_filepath,
                 topic_merging_lda_vis_html_filepath,
                 topic_merging_lda_vis_lambdas_filepath,
                 topic_merging_lda_vis_topics_filepath,
                 merge_id_to_topics=merge_id_to_topics,
                 merge_id_to_x_coor=merge_id_to_x_coor,
                 merge_id_to_y_coor=merge_id_to_y_coor)

    return topic_merging_lda_vis_topics_filepath


def build_topic_grouping(topic_merging_lda_vis_topics_filepath: str,
                         topic_grouping_dendrogram_filepath: str,
                         vis_topic_id_to_org_topics_filepath: str,
                         topic_merging_filepath: str,
                         lda_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build topics grouping\n{'*' * 150}\n")
    topic_grouping_filepath = topic_merging_filepath.replace("merging", "grouping")

    topic_grouping(topic_merging_lda_vis_topics_filepath,
                   topic_grouping_filepath,
                   topic_grouping_dendrogram_filepath,
                   vis_topic_id_to_org_topics_filepath,
                   lda_config["topic_grouping_threshold"])


def main(lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    topic_modeling_dir = os.path.join(domain_dir, lda_config["topic_modeling_folder"])
    corpus_dir = os.path.join(topic_modeling_dir, lda_config["corpus_folder"])
    candidate_models_dir = make_dir(os.path.join(topic_modeling_dir, lda_config["candidate_models_folder"]))
    finetune_model_dir = os.path.join(topic_modeling_dir, lda_config["finetune_model_folder"])
    mallet_corpus_csc_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_csc_filename"])
    topic_merging_dendrogram_filepath = \
        os.path.join(finetune_model_dir, lda_config["topic_merging_dendrogram_filename"])
    topic_grouping_dendrogram_filepath = \
        os.path.join(finetune_model_dir, lda_config["topic_grouping_dendrogram_filename"])
    vis_topic_id_to_org_topics_filepath = \
        os.path.join(finetune_model_dir, lda_config["vis_topic_id_to_org_topics_filename"])

    finetune_model_filepath = setup_finetune_model_dir(candidate_models_dir,
                                                       finetune_model_dir,
                                                       lda_config)

    topic_merging_filepath = build_topic_merging(topic_merging_dendrogram_filepath,
                                                 finetune_model_filepath,
                                                 finetune_model_dir,
                                                 lda_config)

    topic_merging_lda_vis_topics_filepath = build_topic_merging_vis(topic_merging_filepath,
                                                                    finetune_model_filepath,
                                                                    mallet_corpus_csc_filepath,
                                                                    lda_config)

    build_topic_grouping(topic_merging_lda_vis_topics_filepath,
                         topic_grouping_dendrogram_filepath,
                         vis_topic_id_to_org_topics_filepath,
                         topic_merging_filepath,
                         lda_config)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lda_conf", default="conf/lda_template.cfg", required=False)

    lda_config_filepath = parser.parse_args().lda_conf

    main(lda_config_filepath)
