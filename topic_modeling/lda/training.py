from typing import Tuple, List
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger
from utils.resource_util import get_data_filepath
from scipy.sparse import csc_matrix
from gensim import corpora
import argparse
import os


def build_mallet_lda_model(mallet_docs: List[List[str]],
                           mallet_id2word: corpora.Dictionary,
                           mallet_corpus: List[List[Tuple[int, int]]],
                           mallet_corpus_csc: csc_matrix,
                           num_topics: int,
                           alpha: float,
                           workers: int,
                           optimize_interval: int,
                           iterations: int):
    pass


def main(lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    topic_modeling_dir = os.path.join(domain_dir, lda_config["topic_modeling_folder"])
    corpus_dir = os.path.join(topic_modeling_dir, lda_config["corpus_folder"])
    annotation_dir = os.path.join(domain_dir, lda_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, lda_config["extraction_folder"])
    filter_unigram_filepath = os.path.join(extraction_dir, lda_config["filter_unigram_filename"])
    filter_phrase_filepath = os.path.join(extraction_dir, lda_config["filter_phrase_filename"])
    corpus_word_to_lemma_filepath = os.path.join(corpus_dir, lda_config["corpus_word_to_lemma_filename"])
    corpus_noun_phrase_match_filepath = os.path.join(corpus_dir, lda_config["corpus_noun_phrase_match_filename"])
    corpus_filepath = os.path.join(corpus_dir, lda_config["corpus_filename"])
    mallet_docs_filepath = os.path.join(corpus_dir, lda_config["mallet_docs_filename"])
    mallet_id2word_filepath = os.path.join(corpus_dir, lda_config["mallet_id2word_filename"])
    mallet_corpus_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_filename"])
    mallet_corpus_csc_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_csc_filename"])
    mallet_vocab_filepath = os.path.join(corpus_dir, lda_config["mallet_vocab_filename"])


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lda_conf", default="conf/lda_template.cfg", required=False)

    lda_config_filepath = parser.parse_args().lda_conf

    main(lda_config_filepath)
