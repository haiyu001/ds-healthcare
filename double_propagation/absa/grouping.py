from typing import Dict
from double_propagation.absa_utils.extractor_util import load_absa_seed_opinions
from double_propagation.absa_utils.grouping_util import get_hierarchies_in_csv
from word_vector.wv_corpus import extact_wv_corpus_from_annotation
from word_vector.wv_space import WordVec, load_txt_vecs_to_pdf
from word_vector.wv_model import build_word2vec
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from pyspark.sql import DataFrame
import pandas as pd
import json
import logging


def get_aspect_match_dict(aspect_ranking_filepath: str) -> Dict[str, str]:
    aspect_ranking_pdf = pd.read_csv(
        aspect_ranking_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspect_ranking_pdf["members"] = aspect_ranking_pdf["members"].str.lower().apply(json.loads)
    aspect_match_dict = {}
    for _, row in aspect_ranking_pdf.iterrows():
        aspect_match_dict.update({member: "_".join(member.split()) for member in row["members"] if " " in member})
    noun_phrases = sum(aspect_ranking_pdf["noun_phrases"].dropna().str.lower().apply(json.loads).tolist(), [])
    aspect_match_dict.update({noun_phrase: "_".join(noun_phrase.split()) for noun_phrase in noun_phrases})
    return aspect_match_dict


def build_grouping_wv_corpus(annotation_sdf: DataFrame,
                             aspect_ranking_filepath: str,
                             wv_corpus_filepath: str,
                             lang: str,
                             spacy_package: str,
                             match_lowercase: bool):
    logging.info(f"\n{'=' * 100}\nbuild word vector corpus\n{'=' * 100}\n")
    ngram_match_dict = get_aspect_match_dict(aspect_ranking_filepath)
    extact_wv_corpus_from_annotation(annotation_sdf=annotation_sdf,
                                     lang=lang,
                                     spacy_package=spacy_package,
                                     wv_corpus_filepath=wv_corpus_filepath,
                                     ngram_match_dict=ngram_match_dict,
                                     match_lowercase=match_lowercase,
                                     num_partitions=4)


def get_aspect_grouping_vecs(aspect_ranking_filepath: str,
                             grouping_txt_vecs_filepath: str,
                             aspect_grouping_vecs_filepath: str):
    grouping_vecs = WordVec(grouping_txt_vecs_filepath, use_oov_strategy=True)
    aspect_ranking_pdf = pd.read_csv(aspect_ranking_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspect_words = aspect_ranking_pdf["text"].str.lower().str.split().str.join("_")
    grouping_vecs.extract_txt_vecs(aspect_words, aspect_grouping_vecs_filepath, l2_norm=False)


def aspect_grouping(aspect_ranking_filepath: str,
                    aspect_grouping_vecs_filepath: str,
                    aspect_grouping_dendrogram_filepath: str,
                    aspect_grouping_filepath: str,
                    btm_threshold: float,
                    mid_threshold: float,
                    top_threshold: float):
    aspect_ranking_pdf = pd.read_csv(aspect_ranking_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspect_ranking_pdf["members"] = aspect_ranking_pdf["members"].apply(json.loads)
    aspect_ranking_pdf["noun_phrases"] = aspect_ranking_pdf["noun_phrases"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else [])
    aspect_ranking_pdf["members"] = aspect_ranking_pdf["members"] + aspect_ranking_pdf["noun_phrases"]
    aspect_ranking_pdf["vec_text"] = aspect_ranking_pdf["text"].str.lower().str.split().str.join("_")
    aspect_ranking_pdf = aspect_ranking_pdf.set_index("vec_text")

    aspect_ranking_vecs_pdf = load_txt_vecs_to_pdf(aspect_grouping_vecs_filepath, l2_norm=False)
    condensed_distance_matrix = pdist(aspect_ranking_vecs_pdf.values, metric="cosine")
    Z = linkage(condensed_distance_matrix, method="ward", metric="cosine")

    plt.figure(figsize=(25, 15))
    plt.title("aspect grouping")
    dendrogram(Z)
    plt.savefig(aspect_grouping_dendrogram_filepath)

    btm_labels = fcluster(Z, t=btm_threshold, criterion="distance")
    aspect_ranking_pdf["btm"] = pd.Series(btm_labels, aspect_ranking_pdf.index)
    mid_labels = fcluster(Z, t=mid_threshold, criterion="distance")
    aspect_ranking_pdf["mid"] = pd.Series(mid_labels, aspect_ranking_pdf.index)
    top_labels = fcluster(Z, t=top_threshold, criterion="distance")
    aspect_ranking_pdf["top"] = pd.Series(top_labels, aspect_ranking_pdf.index)

    aspect_grouping_pdf = aspect_ranking_pdf[["top", "mid", "btm", "text", "members"]].sort_values(["top", "mid", "btm"])
    aspect_vecs = WordVec(aspect_grouping_vecs_filepath, use_oov_strategy=True)

    # btm group naming
    btm_group_id_to_name = {}
    for group_id, group_pdf in aspect_grouping_pdf.groupby("btm"):
        centroid = aspect_vecs.get_centroid_word(group_pdf.index.tolist())
        name = " ".join(centroid.split("_")).title() + " Category"
        btm_group_id_to_name[group_id] = name
    aspect_grouping_pdf["btm"] = aspect_grouping_pdf["btm"].replace(btm_group_id_to_name)

    # mid group naming
    top_counts = dict(Counter(aspect_grouping_pdf["top"].tolist()))
    for group_id, group_pdf in aspect_grouping_pdf.groupby("mid"):
        mid_count = group_pdf.shape[0]
        top_count = top_counts[group_pdf["top"].tolist()[0]]
        btm_count = len(group_pdf["btm"].unique())
        if mid_count == top_count or btm_count == 1:
            aspect_grouping_pdf.loc[group_pdf.index, "mid"] = 0
    mid_group_ids = sorted([i for i in aspect_grouping_pdf["mid"].unique() if i != 0])
    mid_group_id_mapping = {0: None}
    mid_group_id_mapping.update({id: f"Mid Category {i + 1}" for i, id in enumerate(mid_group_ids)})
    aspect_grouping_pdf["mid"] = aspect_grouping_pdf["mid"].replace(mid_group_id_mapping)

    # top group naming
    if btm_threshold == mid_threshold and mid_threshold == top_threshold:
        aspect_grouping_pdf["top"] = None
    else:
        aspect_grouping_pdf["top"] = aspect_grouping_pdf["top"].apply(lambda x: f"Top Category {x}")

    aspect_grouping_pdf["members"] = aspect_grouping_pdf["members"].apply(json.dumps, ensure_ascii=False)
    aspect_grouping_pdf.columns = ["top_category", "mid_category", "btm_category", "aspect", "members"]
    save_pdf(aspect_grouping_pdf, aspect_grouping_filepath)


def get_opinion_grouping_vecs(opinion_ranking_filepath: str,
                              grouping_txt_vecs_filepath: str,
                              opinion_grouping_vecs_filepath: str):
    grouping_vecs = WordVec(grouping_txt_vecs_filepath, use_oov_strategy=True)
    opinion_ranking_pdf = pd.read_csv(opinion_ranking_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    extracted_opinions = opinion_ranking_pdf["text"].str.lower().tolist()
    seed_opinions_dict = load_absa_seed_opinions()
    seed_opinions = [i for i in seed_opinions_dict if i in grouping_vecs.vocab_set]
    opinion_words = sorted(list(set(extracted_opinions + seed_opinions)))
    grouping_vecs.extract_txt_vecs(opinion_words, opinion_grouping_vecs_filepath, l2_norm=False)


def opinion_grouping(opinion_grouping_vecs_filepath: str,
                     aspect_ranking_filepath: str,
                     opinion_grouping_dendrogram_filepath: str,
                     opinion_grouping_filepath: str,
                     grouping_threshold: float):

    aspect_rankning_pdf = pd.read_csv(aspect_ranking_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspects = set(aspect_rankning_pdf["text"].str.lower().tolist())
    seed_opinions = load_absa_seed_opinions().keys()

    opinion_ranking_vecs_pdf = load_txt_vecs_to_pdf(opinion_grouping_vecs_filepath)
    condensed_distance_matrix = pdist(opinion_ranking_vecs_pdf.values, metric="cosine")
    Z = linkage(condensed_distance_matrix, method="ward", metric="cosine")

    plt.figure(figsize=(25, 15))
    plt.title("opinion grouping")
    dendrogram(Z)
    plt.savefig(opinion_grouping_dendrogram_filepath)

    categories = fcluster(Z, t=grouping_threshold, criterion="distance")
    opinions = opinion_ranking_vecs_pdf.index.tolist()
    opinion_types = []
    for opinion in opinions:
        if opinion in aspects and opinion in seed_opinions:
            opinion_types.append("generic_aspect")
        elif opinion in aspects and opinion not in seed_opinions:
            opinion_types.append("specific_aspect")
        elif opinion not in aspects and opinion not in seed_opinions:
            opinion_types.append("specific_opinion")
        else:
            opinion_types.append("generic_opinion")
    opinion_grouping_pdf = pd.DataFrame({"category": categories,
                                         "opinion": opinions,
                                         "types": opinion_types}).sort_values(by="category")
    opinion_grouping_pdf["category"] = opinion_grouping_pdf["category"].apply(lambda x: f"Category {x}")
    save_pdf(opinion_grouping_pdf, opinion_grouping_filepath)


def get_aspect_hierarchy(aspect_grouping_filepath: str, aspect_hierarchy_filepath: str):
    aspect_grouping_pdf = pd.read_csv(aspect_grouping_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspect_grouping_pdf["members"] = aspect_grouping_pdf["members"].apply(json.loads)
    child_parent_dict = {}

    for i, row in aspect_grouping_pdf.iterrows():
        top_category, mid_category, btm_category = row["top_category"], row["mid_category"], row["btm_category"]
        btm_category_aspects_count = aspect_grouping_pdf[aspect_grouping_pdf["btm_category"] == btm_category].shape[0]
        aspect, members = row["aspect"], row["members"]
        if len(members) > 1:
            members_category = aspect.title()
            if btm_category_aspects_count == 1:
                child_parent_dict.update({member: btm_category for member in members})
            else:
                child_parent_dict.update({member: members_category for member in members})
                child_parent_dict.update({members_category: btm_category})
        else:
            child_parent_dict.update({aspect: btm_category})

        if not isinstance(top_category, str):
            top_category = None
        if not isinstance(mid_category, str):
            mid_category = None

        if not mid_category:
            child_parent_dict.update({btm_category: top_category})
        else:
            child_parent_dict.update({btm_category: mid_category})
            child_parent_dict.update({mid_category: top_category})

        if top_category:
            child_parent_dict.update({top_category: None})
    child_to_parent = OrderedDict(sorted(child_parent_dict.items()))
    get_hierarchies_in_csv(child_to_parent, aspect_hierarchy_filepath)


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
    absa_opinion_dir = os.path.join(absa_dir, "opinion")
    grouping_wv_dir = os.path.join(absa_dir, absa_config["grouping_wv_folder"])
    grouping_wv_corpus_filepath = os.path.join(grouping_wv_dir, absa_config["grouping_wv_corpus_filename"])
    grouping_wv_model_filepath = os.path.join(grouping_wv_dir, absa_config["grouping_wv_model_filename"])
    aspect_ranking_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_ranking_filename"])
    opinion_ranking_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_ranking_filename"])
    aspect_grouping_vecs_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_grouping_vecs_filename"])
    aspect_grouping_dendrogram_filepath = \
        os.path.join(absa_aspect_dir, absa_config["aspect_grouping_dendrogram_filename"])
    aspect_grouping_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_grouping_filename"])
    opinion_grouping_vecs_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_grouping_vecs_filename"])
    opinion_grouping_dendrogram_filepath = \
        os.path.join(absa_opinion_dir, absa_config["opinion_grouping_dendrogram_filename"])
    opinion_grouping_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_grouping_filename"])
    aspect_hierarchy_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_hierarchy_filename"])

    # spark_cores = 4
    # spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="INFO")
    #
    # annotation_sdf = load_annotation(spark,
    #                                  annotation_dir,
    #                                  absa_config["drop_non_english"])
    #
    # build_grouping_wv_corpus(annotation_sdf,
    #                          aspect_ranking_filepath,
    #                          grouping_wv_corpus_filepath,
    #                          absa_config["lang"],
    #                          absa_config["spacy_package"],
    #                          absa_config["wv_corpus_match_lowercase"])
    #
    # build_word2vec(absa_config["wv_size"],
    #                use_char_ngram=False,
    #                wv_corpus_filepath=grouping_wv_corpus_filepath,
    #                wv_model_filepath=grouping_wv_model_filepath)

    # get_aspect_grouping_vecs(aspect_ranking_filepath,
    #                          grouping_wv_model_filepath,
    #                          aspect_grouping_vecs_filepath)
    #
    # aspect_grouping(aspect_ranking_filepath,
    #                 aspect_grouping_vecs_filepath,
    #                 aspect_grouping_dendrogram_filepath,
    #                 aspect_grouping_filepath,
    #                 btm_threshold=0.3,
    #                 mid_threshold=0.8,
    #                 top_threshold=1.5)
    #
    # get_opinion_grouping_vecs(opinion_ranking_filepath,
    #                           grouping_wv_model_filepath,
    #                           opinion_grouping_vecs_filepath)
    #
    # opinion_grouping(opinion_grouping_vecs_filepath,
    #                  aspect_ranking_filepath,
    #                  opinion_grouping_dendrogram_filepath,
    #                  opinion_grouping_filepath,
    #                  grouping_threshold=0.3)

    get_aspect_hierarchy(aspect_grouping_filepath, aspect_hierarchy_filepath)
