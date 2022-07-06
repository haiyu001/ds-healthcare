from typing import Tuple, Dict, List
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from utils.general_util import save_pdf
import pandas as pd
import logging
import json
import os


def merge_topics(mallet_model_filepath: str,
                 lda_vis_topics_filename_suffix: str,
                 finetune_model_dir: str,
                 topic_merge_dendrogram_filepath: str,
                 topic_merge_threshold: float) -> str:
    lda_vis_topics_filepath = f"{mallet_model_filepath}_{lda_vis_topics_filename_suffix}"
    lda_vis_topics_pdf = pd.read_csv(lda_vis_topics_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    lda_vis_topics_pdf = lda_vis_topics_pdf.sort_values(by="mrg_topics")
    topic_coordinates = lda_vis_topics_pdf[["X", "Y"]].values

    condensed_distance_matrix = pdist(topic_coordinates, metric="euclidean")
    Z = linkage(condensed_distance_matrix, method="ward", metric="euclidean")

    plt.figure(figsize=(25, 15))
    plt.title("topics")
    dendrogram(Z)
    plt.savefig(topic_merge_dendrogram_filepath)

    labels = fcluster(Z, t=topic_merge_threshold, criterion="distance")
    topic_merge_count = len(set(labels))

    logging.info(f"\n{'=' * 100}\n"
                 f"num topics before merge: {len(labels)}\n"
                 f"num topics after merge:  {topic_merge_count}\n"
                 f"{'=' * 100}\n")

    topic_merge_pdf = lda_vis_topics_pdf[["mrg_topics", "X", "Y"]].copy()
    topic_merge_pdf["tmp_mrg_id"] = list(labels)
    topic_merge_pdf = topic_merge_pdf.groupby("tmp_mrg_id").agg({"mrg_topics": lambda x: sorted(x.tolist()),
                                                                 "X": "mean",
                                                                 "Y": "mean"})
    topic_merge_pdf["tmp_sort_id"] = topic_merge_pdf["mrg_topics"].apply(lambda x: x[0])
    topic_merge_pdf = topic_merge_pdf.sort_values(by="tmp_sort_id").drop(columns=["tmp_sort_id"])
    topic_merge_pdf = topic_merge_pdf.rename(columns={"mrg_topics": "merge_topics"})
    topic_merge_pdf["merge_id"] = list(range(topic_merge_count))
    topic_merge_pdf["merge_topics"] = topic_merge_pdf["merge_topics"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    topic_merge_pdf = topic_merge_pdf[["merge_id", "merge_topics", "X", "Y"]]

    mallet_model_filename = os.path.basename(mallet_model_filepath)
    topic_merge_filename = f"{mallet_model_filename}_{topic_merge_threshold}_{topic_merge_count}_merge.csv"
    topic_merge_filepath = os.path.join(finetune_model_dir, topic_merge_filename)
    save_pdf(topic_merge_pdf, topic_merge_filepath)
    return topic_merge_filepath


def load_topic_merge_data(topic_merge_filepath: str) \
        -> Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, float]]:
    topic_merge_pdf = pd.read_csv(topic_merge_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    topic_merge_pdf["merge_topics"] = topic_merge_pdf["merge_topics"].apply(lambda x: json.loads(x))
    merge_id_to_topics = dict(zip(topic_merge_pdf["merge_id"], topic_merge_pdf["merge_topics"]))
    merge_id_to_x_coor = dict(zip(topic_merge_pdf["merge_id"], topic_merge_pdf["X"]))
    merge_id_to_y_coor = dict(zip(topic_merge_pdf["merge_id"], topic_merge_pdf["Y"]))
    return merge_id_to_topics, merge_id_to_x_coor, merge_id_to_y_coor



