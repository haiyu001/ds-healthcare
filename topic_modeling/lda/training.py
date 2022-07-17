from typing import Tuple, List
from topic_modeling.topic_modeling_utils.lda_pipeline_util import get_mallet_filepath, get_prefix_by_mallet_model_filepath
from topic_modeling.lda.mallet_wrapper import LdaMallet
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from utils.general_util import save_pdf
from gensim import corpora
import pandas as pd
import logging
import os


def setup_models_coherence_file(models_coherence_filepath):
    if not os.path.exists(models_coherence_filepath):
        models_coherence_pdf = pd.DataFrame({"iterations": [],
                                             "optimize_interval": [],
                                             "topic_alpha": [],
                                             "num_topics": [],
                                             "coherence": []})
    else:
        models_coherence_pdf = \
            pd.read_csv(models_coherence_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    save_pdf(models_coherence_pdf, models_coherence_filepath)


def update_models_coherence_file(mallet_model_filepath: str,
                                 mallet_docs: List[List[str]],
                                 mallet_id2word: Dictionary,
                                 workers: int,
                                 models_coherence_filepath: str):
    mallet_model = LdaMallet.load(mallet_model_filepath)
    coherence_model = CoherenceModel(model=mallet_model, texts=mallet_docs, dictionary=mallet_id2word,
                                     coherence="c_v", processes=workers)
    coherence_row_df = pd.DataFrame({"iterations": [mallet_model.iterations],
                                     "optimize_interval": [mallet_model.optimize_interval],
                                     "topic_alpha": [mallet_model.topic_alpha],
                                     "num_topics": [mallet_model.num_topics],
                                     "coherence": [coherence_model.get_coherence()]})
    setup_models_coherence_file(models_coherence_filepath)
    save_pdf(coherence_row_df, models_coherence_filepath, csv_header=False, csv_mode="a")


def train_mallet_lda_model(mallet_id2word: corpora.Dictionary,
                           mallet_corpus: List[List[Tuple[int, int]]],
                           workers: int,
                           iterations: int,
                           optimize_interval: int,
                           topic_alpha: float,
                           num_topics: int,
                           mallet_model_filepath: str):
    logging.info(f"\n{'=' * 100}\n"
                 f"iterations:  {iterations}\n"
                 f"optimize_interval: {optimize_interval}\n"
                 f"topic_alpha: {topic_alpha}\n"
                 f"num_topics: {num_topics}\n"
                 f"{'=' * 100}\n")

    prefix = get_prefix_by_mallet_model_filepath(mallet_model_filepath)

    mallet_model = LdaMallet(get_mallet_filepath(),
                             mallet_corpus,
                             mallet_id2word,
                             workers,
                             iterations,
                             optimize_interval,
                             num_topics,
                             topic_alpha,
                             prefix)

    mallet_model.save(mallet_model_filepath)
