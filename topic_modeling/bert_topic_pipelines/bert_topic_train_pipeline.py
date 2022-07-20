from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from topic_modeling.bert_topic.bertopic_wrapper import BERTopicWrapper
from topic_modeling.bert_topic.training import load_documents, get_doc_embeddings, get_reduced_embeddings, \
    train_and_save_bert_topic_model
from utils.config_util import read_config_to_dict
from utils.general_util import make_dir, setup_logger
from utils.resource_util import get_data_filepath
import numpy as np
import pandas as pd
import umap
import argparse
import logging
import hdbscan
import os


def build_bert_topic_doc_embeddings(corpus_filepath: str,
                                    doc_embeddings_filepath: str,
                                    reduced_embeddings_filepath: str,
                                    umap_model_filepath: str,
                                    bert_topic_config: Dict[str, Any]) -> Tuple[pd.DataFrame, umap.UMAP, np.ndarray]:
    logging.info(f"\n{'*' * 150}\n* build bert topic doc embeddings\n{'*' * 150}\n")
    # load docs
    documents_pdf = load_documents(corpus_filepath,
                                   bert_topic_config["corpus_doc_id_col"])
    # get doc embeddings
    embeddings = get_doc_embeddings(documents_pdf["Document"].tolist(),
                                    doc_embeddings_filepath,
                                    bert_topic_config["sentence_transformer_model_name"],
                                    bert_topic_config["torch_device"],
                                    show_progress_bar=True)
    # dimension reduction
    umap_model, reduced_embeddings = get_reduced_embeddings(embeddings,
                                                            reduced_embeddings_filepath,
                                                            umap_model_filepath,
                                                            bert_topic_config["n_neighbors"],
                                                            bert_topic_config["n_components"],
                                                            bert_topic_config["umap_metric"],
                                                            bert_topic_config["min_dist"],
                                                            bert_topic_config["low_memory"],
                                                            y=None)
    return documents_pdf, umap_model, reduced_embeddings


def build_bert_topic_model(documents_pdf: pd.DataFrame,
                           reduced_embeddings: np.ndarray,
                           umap_model: umap.UMAP,
                           hdbscan_model_filepath: str,
                           vectorizer_model_filepath: str,
                           bert_topic_model_filepath: str,
                           bert_topic_info_filepath: str,
                           bert_topic_vis_filepath: str,
                           representation_vocab_filepath: str,
                           bert_topic_config: Dict[str, Any]):
    logging.info(f"\n{'*' * 150}\n* build bert topic model\n{'*' * 150}\n")
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=bert_topic_config["min_cluster_size"],
                                    min_samples=bert_topic_config["min_samples"],
                                    metric=bert_topic_config["cluster_metric"],
                                    cluster_selection_method=bert_topic_config["cluster_selection_method"],
                                    prediction_data=bert_topic_config["prediction_data"])
    bert_topic_model = BERTopicWrapper(language=bert_topic_config["language"],
                                       top_n_words=bert_topic_config["top_n_words"],
                                       nr_topics=bert_topic_config["nr_topics"],
                                       calculate_probabilities=bert_topic_config["calculate_probabilities"],
                                       diversity=bert_topic_config["diversity"],
                                       umap_model=umap_model,
                                       hdbscan_model=hdbscan_model,
                                       vectorizer_model=CountVectorizer(),
                                       seed_topic_list=None,
                                       verbose=True)
    train_and_save_bert_topic_model(bert_topic_model,
                                    documents_pdf,
                                    reduced_embeddings,
                                    hdbscan_model_filepath,
                                    vectorizer_model_filepath,
                                    bert_topic_model_filepath,
                                    bert_topic_info_filepath,
                                    bert_topic_vis_filepath,
                                    representation_vocab_filepath)


def main(bert_topic_config_filepath: str):
    bert_topic_config = read_config_to_dict(bert_topic_config_filepath)
    domain_dir = get_data_filepath(bert_topic_config["domain"])
    topic_modeling_dir = make_dir(os.path.join(domain_dir, bert_topic_config["topic_modeling_folder"]))
    corpus_dir = make_dir(os.path.join(topic_modeling_dir, bert_topic_config["corpus_folder"]))
    bert_topic_model_dir = make_dir(os.path.join(topic_modeling_dir, bert_topic_config["bert_topic_model_folder"]))
    corpus_filepath = os.path.join(corpus_dir, bert_topic_config["corpus_filename"])
    doc_embeddings_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["doc_embeddings_filename"])
    reduced_embeddings_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["reduced_embeddings_filename"])
    umap_model_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["umap_model_filename"])
    hdbscan_model_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["hdbscan_model_filename"])
    vectorizer_model_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["vectorizer_model_filename"])
    bert_topic_model_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["bert_topic_model_filename"])
    bert_topic_info_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["bert_topic_info_filename"])
    bert_topic_vis_filepath = os.path.join(bert_topic_model_dir, bert_topic_config["bert_topic_vis_filename"])
    representation_vocab_filepath = os.path.join(bert_topic_model_dir,
                                                 bert_topic_config["representation_vocab_filename"])

    documents_pdf, umap_model, reduced_embeddings = build_bert_topic_doc_embeddings(corpus_filepath,
                                                                                    doc_embeddings_filepath,
                                                                                    reduced_embeddings_filepath,
                                                                                    umap_model_filepath,
                                                                                    bert_topic_config)

    build_bert_topic_model(documents_pdf,
                           reduced_embeddings,
                           umap_model,
                           hdbscan_model_filepath,
                           vectorizer_model_filepath,
                           bert_topic_model_filepath,
                           bert_topic_info_filepath,
                           bert_topic_vis_filepath,
                           representation_vocab_filepath,
                           bert_topic_config)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_topic_conf", default="conf/bert_topic_template.cfg", required=False)

    bert_topic_config_filepath = parser.parse_args().bert_topic_conf

    main(bert_topic_config_filepath)
