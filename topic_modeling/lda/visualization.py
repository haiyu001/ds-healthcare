from typing import Optional, Dict, List, Any
from utils.general_util import save_pdf
from topic_modeling.lda.mallet_wrapper import LdaMallet, malletmodel2ldamodel
from pyLDAvis._prepare import PreparedData, prepare as lda_vis_prepare
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import matutils
from scipy.sparse import csc_matrix
import pandas as pd
import numpy as np
import pyLDAvis
import json


def get_lda_vis_opts(lda_model: LdaModel,
                     corpus_csc: csc_matrix,
                     id2word: Dictionary,
                     doc_topic_dists: np.ndarray,
                     beta: float = 0.01) -> Dict[str, Any]:
    argsort = np.asarray(list(id2word.token2id.values()), dtype=np.long)
    # topic_term_dists
    topic = lda_model.state.get_lambda()
    topic = topic / topic.sum(axis=1)[:, None]
    topic_term_dists = topic[:, argsort]
    # doc_lengths
    doc_lengths = corpus_csc.sum(axis=0).A.ravel()
    # term_frequency
    term_frequency = corpus_csc.sum(axis=1).A.ravel()[argsort]
    term_frequency[term_frequency == 0] = beta
    # vocab
    vocab = list(id2word.token2id.keys())
    lda_vis_data = {
        "doc_topic_dists": doc_topic_dists,
        "topic_term_dists": topic_term_dists,
        "doc_lengths": doc_lengths,
        "term_frequency": term_frequency,
        "vocab": vocab,
    }
    return lda_vis_data


def update_dists_by_merge_id(lda_vis_opts: Dict[str, Any],
                             merge_id_to_topics: Dict[int, List[int]]) -> Dict[str, Any]:
    doc_topic_dists_pdf = pd.DataFrame(lda_vis_opts["doc_topic_dists"])
    topic_term_dists_pdf = pd.DataFrame(lda_vis_opts["topic_term_dists"])
    merged_doc_topic_dists_pdf = pd.DataFrame()
    merged_topic_term_dists_list = []
    for i in merge_id_to_topics:
        merged_doc_topic_dists_pdf[i] = doc_topic_dists_pdf[merge_id_to_topics[i]].sum(axis=1)
    for i in merge_id_to_topics:
        merged_topic_term_dists_list.append(topic_term_dists_pdf.iloc[merge_id_to_topics[i], :].mean(axis=0))
    merged_topic_term_dists_pdf = pd.concat(merged_topic_term_dists_list, axis=1).T
    lda_vis_opts["doc_topic_dists"] = merged_doc_topic_dists_pdf.values
    lda_vis_opts["topic_term_dists"] = merged_topic_term_dists_pdf.values
    return lda_vis_opts


def get_lda_vis_data(lda_vis_opts: dict[str, Any],
                     num_terms_to_display: int = 30,
                     lambda_step: float = 0.01,
                     sort_topics: bool = True,
                     multi_dimensional_scaling: str = "tsne",
                     merge_id_to_x_coor: Optional[Dict[int, float]] = None,
                     merge_id_to_y_coor: Optional[Dict[int, float]] = None) -> PreparedData:
    lda_vis_data = lda_vis_prepare(**lda_vis_opts,
                                   R=num_terms_to_display,
                                   lambda_step=lambda_step,
                                   sort_topics=sort_topics,
                                   mds=multi_dimensional_scaling)
    if merge_id_to_x_coor and merge_id_to_y_coor:
        topic_coordinates_pdf = lda_vis_data.topic_coordinates.drop(columns=["x", "y"])
        topic_coordinates_pdf["x"] = [merge_id_to_x_coor[i] for i in topic_coordinates_pdf.index.tolist()]
        topic_coordinates_pdf["y"] = [merge_id_to_y_coor[i] for i in topic_coordinates_pdf.index.tolist()]
        lda_vis_data = PreparedData(topic_coordinates_pdf,
                                    lda_vis_data.topic_info,
                                    lda_vis_data.token_table,
                                    lda_vis_data.R,
                                    lda_vis_data.lambda_step,
                                    lda_vis_data.plot_opts,
                                    lda_vis_data.topic_order)
    return lda_vis_data


def save_lda_vis_stats(lda_vis_dict: Dict[str, Any],
                       group_id_to_topics: Dict[int, List[int]],
                       lda_vis_lambdas_filepath: str,
                       lda_vis_topics_filepath: str,
                       lda_vis_topics_lambda=0.6):
    lda_vis_pdf = pd.DataFrame(lda_vis_dict["mdsDat"])
    lda_vis_pdf["mrg_topics"] = [i - 1 for i in lda_vis_dict["topic.order"]]
    lda_vis_pdf["org_topics"] = lda_vis_pdf["mrg_topics"].apply(lambda x: group_id_to_topics[x])
    lda_vis_pdf = lda_vis_pdf[["topics", "mrg_topics", "org_topics", "x", "y", "Freq"]]
    lda_vis_pdf.columns = ["vis_topics", "mrg_topics", "org_topics", "X", "Y", "Freq"]

    lda_vis_terms_pdf = pd.DataFrame(lda_vis_dict["tinfo"])
    lda_vis_terms_pdf = lda_vis_terms_pdf[lda_vis_terms_pdf["Category"] != "Default"]
    lda_vis_terms_pdf["vis_topics"] = lda_vis_terms_pdf["Category"].apply(lambda x: int(x[5:]))
    lda_vis_terms_pdf["Prob"] = np.exp(lda_vis_terms_pdf["logprob"])
    lda_vis_terms_pdf = lda_vis_terms_pdf.merge(lda_vis_pdf[["vis_topics", "mrg_topics"]], on="vis_topics")
    lda_vis_terms_pdf = lda_vis_terms_pdf[
        ["mrg_topics", "vis_topics", "Term", "Freq", "Total", "Prob", "logprob", "loglift"]]
    lda_vis_terms_pdf = lda_vis_terms_pdf.sort_values(by=["mrg_topics", "logprob"], ascending=[True, False])
    for lambda_value in np.arange(0, 1 + 0.05, 0.05):
        lambda_value = round(lambda_value, 2)
        lda_vis_terms_pdf[f"lambda_{lambda_value}"] = lambda_value * lda_vis_terms_pdf["logprob"] + \
                                                      (1 - lambda_value) * lda_vis_terms_pdf["loglift"]
    save_pdf(lda_vis_terms_pdf, lda_vis_lambdas_filepath)

    mrg_topic_id_to_terms = dict()
    for i in lda_vis_terms_pdf["mrg_topics"].tolist():
        topic_terms_df = \
            lda_vis_terms_pdf[lda_vis_terms_pdf["mrg_topics"] == i][["Term", "Prob", f"lambda_{lda_vis_topics_lambda}"]]
        topic_terms = topic_terms_df.to_dict(orient="records")
        mrg_topic_id_to_terms[i] = json.dumps(topic_terms, ensure_ascii=False)
    lda_vis_pdf["Terms"] = lda_vis_pdf["mrg_topics"].apply(lambda x: mrg_topic_id_to_terms[x])
    save_pdf(lda_vis_pdf, lda_vis_topics_filepath)


def save_lda_vis(mallet_model_filepath: str,
                 lda_vis_html_filepath: str,
                 mallet_corpus_csc: csc_matrix,
                 num_terms_to_display: int = 30,
                 lambda_step: float = 0.01,
                 sort_topics: bool = True,
                 multi_dimensional_scaling: str = "tsne",
                 merge_id_to_topics: Optional[Dict[int, List[int]]] = None,
                 merge_id_to_x_coor: Optional[Dict[int, float]] = None,
                 merge_id_to_y_coor: Optional[Dict[int, float]] = None):
    mallet_model = LdaMallet.load(mallet_model_filepath)
    lda_model = malletmodel2ldamodel(mallet_model)

    doc_topics = list(mallet_model.load_document_topics())
    doc_topic_dists = matutils.corpus2dense(doc_topics, mallet_model.num_topics).T
    lda_vis_opts = get_lda_vis_opts(lda_model, mallet_corpus_csc, mallet_model.id2word, doc_topic_dists)

    if merge_id_to_topics is None:
        merge_id_to_topics = {i: [i] for i in list(range(mallet_model.num_topics))}
        merge_id_to_x_coor = None
        merge_id_to_y_coor = None
    lda_vis_opts = update_dists_by_merge_id(lda_vis_opts, merge_id_to_topics)

    lda_vis_data = get_lda_vis_data(lda_vis_opts,
                                    num_terms_to_display,
                                    lambda_step,
                                    sort_topics,
                                    multi_dimensional_scaling,
                                    merge_id_to_x_coor,
                                    merge_id_to_y_coor)
    pyLDAvis.save_html(lda_vis_data, lda_vis_html_filepath)

    lda_vis_lambdas_filepath = f"{lda_vis_html_filepath.rsplit('.', 1)[0]}_lambdas.csv"
    lda_vis_topics_filepath = f"{lda_vis_html_filepath.rsplit('.', 1)[0]}_topics.csv"
    save_lda_vis_stats(lda_vis_data.to_dict(),
                       merge_id_to_topics,
                       lda_vis_lambdas_filepath,
                       lda_vis_topics_filepath,
                       lda_vis_topics_lambda=0.6)
