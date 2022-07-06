from typing import Tuple, Dict, List
from scipy.cluster.hierarchy import fcluster
from machine_learning.hierarchical_clustering import get_linkage_matrix
from utils.general_util import save_pdf
import pandas as pd
import logging
import json
import os


def load_topic_merging_data(topic_merging_filepath: str) \
        -> Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, float]]:
    topic_merging_pdf = pd.read_csv(topic_merging_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    topic_merging_pdf["topic_merging"] = topic_merging_pdf["topic_merging"].apply(lambda x: json.loads(x))
    merge_id_to_topics = dict(zip(topic_merging_pdf["merge_id"], topic_merging_pdf["topic_merging"]))
    merge_id_to_x_coor = dict(zip(topic_merging_pdf["merge_id"], topic_merging_pdf["X"]))
    merge_id_to_y_coor = dict(zip(topic_merging_pdf["merge_id"], topic_merging_pdf["Y"]))
    return merge_id_to_topics, merge_id_to_x_coor, merge_id_to_y_coor


def topic_merging(mallet_model_filepath: str,
                  lda_vis_topics_filename_suffix: str,
                  finetune_model_dir: str,
                  topic_merging_dendrogram_filepath: str,
                  topic_merging_threshold: float) -> str:
    lda_vis_topics_filepath = f"{mallet_model_filepath}_{lda_vis_topics_filename_suffix}"
    lda_vis_topics_pdf = pd.read_csv(lda_vis_topics_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    lda_vis_topics_pdf = lda_vis_topics_pdf.sort_values(by="mrg_topics")
    topic_coordinates = lda_vis_topics_pdf[["X", "Y"]].values

    Z = get_linkage_matrix(topic_coordinates,
                           dendrogram_title="topic merging",
                           dendrogram_filepath=topic_merging_dendrogram_filepath,
                           metric="euclidean",
                           linkage_method="ward")

    labels = fcluster(Z, t=topic_merging_threshold, criterion="distance")
    topic_merging_count = len(set(labels))

    logging.info(f"\n{'=' * 100}\n"
                 f"num topics before merging: {len(labels)}\n"
                 f"num topics after merging:  {topic_merging_count}\n"
                 f"{'=' * 100}\n")

    topic_merging_pdf = lda_vis_topics_pdf[["mrg_topics", "X", "Y"]].copy()
    topic_merging_pdf["tmp_mrg_id"] = list(labels)
    topic_merging_pdf = topic_merging_pdf.groupby("tmp_mrg_id").agg({"mrg_topics": lambda x: sorted(x.tolist()),
                                                                     "X": "mean",
                                                                     "Y": "mean"})
    topic_merging_pdf["tmp_sort_id"] = topic_merging_pdf["mrg_topics"].apply(lambda x: x[0])
    topic_merging_pdf = topic_merging_pdf.sort_values(by="tmp_sort_id").drop(columns=["tmp_sort_id"])
    topic_merging_pdf = topic_merging_pdf.rename(columns={"mrg_topics": "topic_merging"})
    topic_merging_pdf["merge_id"] = list(range(topic_merging_count))
    topic_merging_pdf["topic_merging"] = topic_merging_pdf["topic_merging"].apply(
        lambda x: json.dumps(x, ensure_ascii=False))
    topic_merging_pdf = topic_merging_pdf[["merge_id", "topic_merging", "X", "Y"]]

    mallet_model_filename = os.path.basename(mallet_model_filepath)
    topic_merging_filename = f"{mallet_model_filename}_{topic_merging_threshold}_{topic_merging_count}_merging.csv"
    topic_merging_filepath = os.path.join(finetune_model_dir, topic_merging_filename)
    save_pdf(topic_merging_pdf, topic_merging_filepath)
    return topic_merging_filepath


def topic_grouping(lda_vis_topics_filepath: str,
                   topic_grouping_filepath: str,
                   topic_grouping_dendrogram_filepath: str,
                   topic_grouping_threshold: float,
                   topic_top_n_terms: int = 30):
    lda_vis_topics_pdf = pd.read_csv(lda_vis_topics_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    lda_vis_topics_pdf["Terms"] = lda_vis_topics_pdf["Terms"].apply(json.loads)
    topic_coordinates = lda_vis_topics_pdf[["X", "Y"]].values

    Z = get_linkage_matrix(topic_coordinates,
                           dendrogram_title="topic grouping",
                           dendrogram_filepath=topic_grouping_dendrogram_filepath,
                           metric="euclidean",
                           linkage_method="ward")

    labels = fcluster(Z, t=topic_grouping_threshold, criterion="distance")
    topic_grouping_count = len(set(labels))

    logging.info(f"\n{'=' * 100}\n"
                 f"num topics before grouping: {len(labels)}\n"
                 f"num topics after grouping:  {topic_grouping_count}\n"
                 f"{'=' * 100}\n")

    lda_vis_topics_pdf["category"] = [f"Category_{i}" for i in labels]
    lda_vis_topics_pdf["topic_terms"] = lda_vis_topics_pdf["Terms"].apply(
        lambda x: " ".join(
            [t["Term"] for t in sorted(x, key=lambda i: i["lambda_0.6"], reverse=True)][:topic_top_n_terms]))
    lda_vis_topics_pdf = lda_vis_topics_pdf.rename(columns={"vis_topics": "topic_id"})
    topic_grouping_pdf = lda_vis_topics_pdf[["category", "topic_id", "topic_terms"]] \
        .sort_values(by=["category", "topic_id"])
    save_pdf(topic_grouping_pdf, topic_grouping_filepath)
