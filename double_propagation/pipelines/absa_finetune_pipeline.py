from typing import Dict, Any
from double_propagation.absa.grouping import save_aspect_grouping, save_opinion_grouping, save_aspect, save_opinion
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger, make_dir
from utils.resource_util import get_data_filepath
import argparse
import logging
import os


def group_aspect_and_opinion(aspect_ranking_filepath: str,
                             aspect_grouping_filepath: str,
                             aspect_grouping_vecs_filepath: str,
                             aspect_grouping_dendrogram_filepath: str,
                             opinion_grouping_filepath: str,
                             opinion_grouping_vecs_filepath: str,
                             opinion_grouping_dendrogram_filepath: str,
                             absa_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build aspect and opinion grouping\n{'*' * 150}\n")
    save_aspect_grouping(aspect_ranking_filepath,
                         aspect_grouping_vecs_filepath,
                         aspect_grouping_dendrogram_filepath,
                         aspect_grouping_filepath,
                         absa_config["aspect_grouping_btm_threshold"],
                         absa_config["aspect_grouping_mid_threshold"],
                         absa_config["aspect_grouping_top_threshold"])

    save_opinion_grouping(aspect_ranking_filepath,
                          opinion_grouping_vecs_filepath,
                          opinion_grouping_dendrogram_filepath,
                          opinion_grouping_filepath,
                          absa_config["opinion_grouping_threshold"])


def finalize_aspect_and_opinion(aspect_grouping_filepath: str,
                                opinion_ranking_filepath: str,
                                opinion_grouping_filepath: str,
                                aspect_hierarchy_filepath: str,
                                opinion_hierarchy_filepath: str,
                                aspect_filepath: str,
                                opinion_filepath: str,
                                absa_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* finalize aspect and opinion\n{'*' * 150}\n")
    save_aspect(aspect_grouping_filepath,
                aspect_filepath,
                aspect_hierarchy_filepath)

    save_opinion(opinion_ranking_filepath,
                 opinion_grouping_filepath,
                 opinion_filepath,
                 opinion_hierarchy_filepath,
                 absa_config["opinion_filter_min_score"],
                 absa_config["drop_unknown_polarity_opinion"])


def main(absa_config_filepath: str):
    absa_config = read_config_to_dict(absa_config_filepath)
    domain_dir = get_data_filepath(absa_config["domain"])
    absa_dir = os.path.join(domain_dir, absa_config["absa_folder"])
    absa_aspect_dir = make_dir(os.path.join(absa_dir, "aspect"))
    absa_opinion_dir = make_dir(os.path.join(absa_dir, "opinion"))
    aspect_ranking_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_ranking_filename"])
    aspect_grouping_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_grouping_filename"])
    aspect_grouping_vecs_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_grouping_vecs_filename"])
    aspect_grouping_dendrogram_filepath = \
        os.path.join(absa_aspect_dir, absa_config["aspect_grouping_dendrogram_filename"])
    opinion_ranking_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_ranking_filename"])
    opinion_grouping_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_grouping_filename"])
    opinion_grouping_vecs_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_grouping_vecs_filename"])
    opinion_grouping_dendrogram_filepath = \
        os.path.join(absa_opinion_dir, absa_config["opinion_grouping_dendrogram_filename"])
    aspect_hierarchy_filepath = os.path.join(absa_dir, absa_config["aspect_hierarchy_filename"])
    opinion_hierarchy_filepath = os.path.join(absa_dir, absa_config["opinion_hierarchy_filename"])
    aspect_filepath = os.path.join(absa_dir, absa_config["aspect_filename"])
    opinion_filepath = os.path.join(absa_dir, absa_config["opinion_filename"])

    group_aspect_and_opinion(aspect_ranking_filepath,
                             aspect_grouping_filepath,
                             aspect_grouping_vecs_filepath,
                             aspect_grouping_dendrogram_filepath,
                             opinion_grouping_filepath,
                             opinion_grouping_vecs_filepath,
                             opinion_grouping_dendrogram_filepath,
                             absa_config)

    finalize_aspect_and_opinion(aspect_grouping_filepath,
                                opinion_ranking_filepath,
                                opinion_grouping_filepath,
                                aspect_hierarchy_filepath,
                                opinion_hierarchy_filepath,
                                aspect_filepath,
                                opinion_filepath,
                                absa_config)


if __name__ == "__main__":

    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--absa_conf", default="conf/absa_template.cfg", required=False)

    absa_config_filepath = parser.parse_args().absa_conf

    main(absa_config_filepath)
