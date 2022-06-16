from typing import Dict
from double_propagation.absa.enumerations import Polarity
from double_propagation.absa_utils.extractor_util import load_absa_seed_opinions
from double_propagation.absa_utils.grouping_util import get_hierarchies_in_csv
from utils.general_util import save_pdf, dump_json_file
from word_vector.wv_corpus import build_wv_corpus_by_annotation
from word_vector.wv_space import WordVec, load_txt_vecs_to_pdf
from word_vector.wv_model import build_word2vec
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from pyspark.sql import DataFrame
import collections
import pandas as pd
import json
import logging


def get_aspect_match_dict(aspect_ranking_filepath: str, match_lowercase: bool = True) -> Dict[str, str]:
    aspect_ranking_pdf = pd.read_csv(
        aspect_ranking_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspect_ranking_pdf["members"] = aspect_ranking_pdf["members"].str.lower().apply(json.loads) if match_lowercase \
        else aspect_ranking_pdf["members"].apply(json.loads)
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
    ngram_match_dict = get_aspect_match_dict(aspect_ranking_filepath, match_lowercase)
    build_wv_corpus_by_annotation(annotation_sdf=annotation_sdf,
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


def save_aspect_grouping(aspect_ranking_filepath: str,
                         aspect_grouping_vecs_filepath: str,
                         aspect_grouping_dendrogram_filepath: str,
                         aspect_grouping_filepath: str,
                         aspect_grouping_btm_threshold: float,
                         aspect_grouping_mid_threshold: float,
                         aspect_grouping_top_threshold: float):
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

    btm_labels = fcluster(Z, t=aspect_grouping_btm_threshold, criterion="distance")
    aspect_ranking_pdf["btm"] = pd.Series(btm_labels, aspect_ranking_pdf.index)
    mid_labels = fcluster(Z, t=aspect_grouping_mid_threshold, criterion="distance")
    aspect_ranking_pdf["mid"] = pd.Series(mid_labels, aspect_ranking_pdf.index)
    top_labels = fcluster(Z, t=aspect_grouping_top_threshold, criterion="distance")
    aspect_ranking_pdf["top"] = pd.Series(top_labels, aspect_ranking_pdf.index)

    aspect_grouping_pdf = aspect_ranking_pdf[["top", "mid", "btm", "text", "members"]].sort_values(
        ["top", "mid", "btm"])
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
    if aspect_grouping_btm_threshold == aspect_grouping_mid_threshold and aspect_grouping_mid_threshold == aspect_grouping_top_threshold:
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


def save_opinion_grouping(aspect_ranking_filepath: str,
                          opinion_grouping_vecs_filepath: str,
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
                                         "type": opinion_types}).sort_values(by="category")
    opinion_grouping_pdf["category"] = opinion_grouping_pdf["category"].apply(lambda x: f"Category {x}")
    save_pdf(opinion_grouping_pdf, opinion_grouping_filepath)


def save_aspect(aspect_grouping_filepath: str, aspect_filepath: str, aspect_hierarchy_filepath: str):
    aspect_grouping_pdf = pd.read_csv(aspect_grouping_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspect_grouping_pdf["members"] = aspect_grouping_pdf["members"].str.lower().apply(json.loads)
    child_parent_dict = {}

    for i, row in aspect_grouping_pdf.iterrows():
        top_category, mid_category, btm_category = row["top_category"], row["mid_category"], row["btm_category"]
        btm_category_aspects_count = aspect_grouping_pdf[aspect_grouping_pdf["btm_category"] == btm_category].shape[0]
        aspect, members = row["aspect"], row["members"]
        if len(members) > 1:
            members_category = aspect if aspect.isupper() else aspect.title()
            if btm_category_aspects_count == 1:
                child_parent_dict.update({member: btm_category for member in members})
            else:
                child_parent_dict.update({member: members_category for member in members})
                child_parent_dict.update({members_category: btm_category})
        else:
            child_parent_dict.update({aspect.lower(): btm_category})

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
    dump_json_file(child_to_parent, aspect_filepath)
    get_hierarchies_in_csv(child_to_parent, aspect_hierarchy_filepath)


def save_opinion(opinion_ranking_filepath: str,
                 opinion_grouping_filepath: str,
                 opinion_filepath: str,
                 opinion_filter_min_score: float,
                 drop_unknown_polarity_opinion: bool):
    opinion_grouping_pdf = pd.read_csv(opinion_grouping_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    opinion_ranking_pdf = pd.read_csv(opinion_ranking_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    opinion_grouping_pdf["opinion"] = opinion_grouping_pdf["opinion"].str.lower()
    opinion_ranking_pdf["opinion"] = opinion_ranking_pdf["text"].str.lower()
    if drop_unknown_polarity_opinion:
        opinion_ranking_pdf = opinion_ranking_pdf[opinion_ranking_pdf["polarity"] != Polarity.UNK.name]
    if opinion_filter_min_score:
        opinion_ranking_pdf = opinion_ranking_pdf[opinion_ranking_pdf["max_score"] >= opinion_filter_min_score]

    opinion_to_score = dict(zip(opinion_ranking_pdf["opinion"], opinion_ranking_pdf["max_score"]))
    opinion_to_polarity = dict(zip(opinion_ranking_pdf["opinion"], opinion_ranking_pdf["polarity"]))
    opinion_to_category = dict(zip(opinion_grouping_pdf["opinion"], opinion_grouping_pdf["category"]))
    opinion_to_type = dict(zip(opinion_grouping_pdf["opinion"], opinion_grouping_pdf["type"]))
    seed_opinions = load_absa_seed_opinions()
    opinion_dict = collections.defaultdict(dict)
    for opinion, opinion_type in opinion_to_type.items():
        if opinion_type.startswith("generic"):
            sentiment_score = -1.0 if seed_opinions[opinion] == "NEG" else 1.0
        elif opinion_type.startswith("specific") and opinion in opinion_to_score:
            sentiment_score = opinion_to_score[opinion] * (-1.0 if opinion_to_polarity[opinion] == "NEG" else 1.0)
        opinion_dict[opinion]["category"] = opinion_to_category[opinion]
        opinion_dict[opinion]["type"] = opinion_to_type[opinion]
        opinion_dict[opinion]["sentiment_score"] = sentiment_score
    opinion_to_sentiment_score = OrderedDict(sorted(opinion_dict.items()))
    dump_json_file(opinion_to_sentiment_score, opinion_filepath)
