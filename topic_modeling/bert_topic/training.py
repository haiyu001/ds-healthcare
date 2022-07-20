from typing import List, Optional, Tuple
from topic_modeling.bert_topic.bertopic_wrapper import BERTopicWrapper
from sentence_transformers import SentenceTransformer
from utils.general_util import dump_json_file, save_pdf
import pandas as pd
import numpy as np
import joblib
import logging
import umap
import os


def load_documents(corpus_filepath: str, corpus_doc_id_col: str) -> pd.DataFrame:
    corpus_pdf = pd.read_json(corpus_filepath, orient="records", lines=True, encoding="utf-8")
    corpus_pdf = corpus_pdf.rename(columns={"sentence_text": "Document",
                                            "sentence_lemma": "Representation"})
    corpus_pdf["ID"] = corpus_pdf[corpus_doc_id_col] + "_" + corpus_pdf["sentence_id"].astype(str)
    corpus_pdf["Topic"] = None
    corpus_pdf = corpus_pdf[["ID", "Document", "Topic", "Representation"]]
    return corpus_pdf


def get_doc_embeddings(docs: List[str],
                       doc_embeddings_filepath: str,
                       sentence_transformer_model_name: str,
                       torch_device: Optional[str] = None,
                       show_progress_bar: bool = True) -> np.ndarray:
    sentence_model = SentenceTransformer(sentence_transformer_model_name)
    if not os.path.exists(doc_embeddings_filepath):
        embeddings = sentence_model.encode(docs, device=torch_device, show_progress_bar=show_progress_bar)
        np.save(doc_embeddings_filepath, embeddings)
        logging.info(f"\n{'=' * 100}\ndoc embeddings dimension: {embeddings.shape}\n{'=' * 100}\n")
    else:
        embeddings = np.load(doc_embeddings_filepath)
        logging.info(f"\n{'=' * 100}\nloaded doc embeddings dimension: {embeddings.shape}\n{'=' * 100}\n")
    return embeddings


def get_reduced_embeddings(embeddings: np.ndarray,
                           reduced_embeddings_filepath: str,
                           umap_model_filepath: str,
                           n_neighbors: int = 15,
                           n_components: int = 10,
                           umap_metric: str = "cosine",
                           min_dist: float = 0.1,
                           low_memory: bool = False,
                           y: Optional[List[int]] = None) -> Tuple[umap.UMAP, np.ndarray]:
    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                           n_components=n_components,
                           metric=umap_metric,
                           low_memory=low_memory,
                           min_dist=min_dist,
                           verbose=True)
    if not os.path.exists(reduced_embeddings_filepath):
        umap_embeddings = umap_model.fit_transform(embeddings, y)
        umap_embeddings = np.nan_to_num(umap_embeddings)
        np.save(reduced_embeddings_filepath, umap_embeddings)
        joblib.dump(umap_model, umap_model_filepath)
        logging.info(f"\n{'=' * 100}\nreduced embeddings dimension: {umap_embeddings.shape}\n{'=' * 100}\n")
    else:
        umap_embeddings = np.load(reduced_embeddings_filepath)
        umap_model = joblib.load(umap_model_filepath)
        logging.info(f"\n{'=' * 100}\nloading umap model from {umap_model_filepath}\n"
                     f"loaded reduced embeddings dimension: {umap_embeddings.shape}\n{'=' * 100}\n")
    return umap_model, umap_embeddings


def save_bert_topic_info_and_vis(bert_topic_model: BERTopicWrapper,
                                 bert_topic_info_filepath: str,
                                 bert_topic_vis_filepath: str):
    topic_info_pdf = bert_topic_model.get_topic_info()
    topic_terms_list = []
    for topic_id, term_prob_list in bert_topic_model.get_topics().items():
        topic_terms = []
        for term, prob in term_prob_list:
            topic_terms.append(term)
        topic_terms = " ".join(topic_terms)
        topic_terms_list.append(topic_terms)
    topic_info_pdf["Terms"] = topic_terms_list
    save_pdf(topic_info_pdf, bert_topic_info_filepath)
    bert_topic_model.visualize_topics().write_html(bert_topic_vis_filepath)


def train_and_save_bert_topic_model(bert_topic_model: BERTopicWrapper,
                                    documents: pd.DataFrame,
                                    reduced_embeddings: np.ndarray,
                                    hdbscan_model_filepath: str,
                                    vectorizer_model_filepath: str,
                                    bert_topic_model_filepath: str,
                                    bert_topic_info_filepath: str,
                                    bert_topic_vis_filepath: str,
                                    representation_vocab_filepath: str):
    bert_topic_model.fit_transform(documents, reduced_embeddings, y=None)
    joblib.dump(bert_topic_model.hdbscan_model, hdbscan_model_filepath)
    joblib.dump(bert_topic_model.vectorizer_model, vectorizer_model_filepath)
    bert_topic_model.save(bert_topic_model_filepath, save_embedding_model=False)
    save_bert_topic_info_and_vis(bert_topic_model, bert_topic_info_filepath, bert_topic_vis_filepath)
    dump_json_file(bert_topic_model.vectorizer_model.get_feature_names(), representation_vocab_filepath)
