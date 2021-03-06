from typing import Optional
from utils.general_util import make_dir
from gensim.models import FastText, Word2Vec
import multiprocessing
import logging
import os


def get_corpus_size(corpus_file_path: str) -> int:
    count = 0
    with open(corpus_file_path) as corpus:
        for _ in corpus:
            count += 1
    return count


def build_word2vec(vector_size: int,
                   use_char_ngram: bool,
                   wv_corpus_filepath: str,
                   wv_model_filepath: str,
                   min_count: int = 5,
                   workers: Optional[int] = None,
                   epochs: int = 8,
                   max_final_vocab: int = 1000000):
    if workers is None:
        corpus_size = get_corpus_size(wv_corpus_filepath)
        workers = max(1, min(int(corpus_size / 200000), 16, multiprocessing.cpu_count()))

    logging.info(f"\n{'=' * 100}\n"
                 f"{'fastText' if use_char_ngram else 'Word2Vec'}"
                 f"\tepochs: {epochs} | workers: {workers} | min_count: {min_count} | max_final_vocab: {max_final_vocab}"
                 f"\n{'=' * 100}\n")

    default_params = dict(
        vector_size=vector_size,
        epochs=epochs,
        min_count=min_count,
        max_final_vocab=max_final_vocab,
        window=5,
        alpha=0.02,
        min_alpha=0.0001,
        sg=1,  # skip-gram if sg = 1 else cbow
        hs=0,  # negative sampling if hs = 0 else hierarchical softmax
        negative=5,  # negative sampling
        sorted_vocab=1,  # sort the vocabulary by descending frequency before assigning word indices.
        max_vocab_size=None,  # no limit
        workers=workers,  # require install cython
        batch_words=10000,
        sample=0.0001,
    )

    if use_char_ngram:
        model = FastText(corpus_file=wv_corpus_filepath, min_n=3, max_n=6, word_ngrams=1, **default_params)
    else:
        model = Word2Vec(corpus_file=wv_corpus_filepath, **default_params)

    model_dir = make_dir(os.path.splitext(wv_model_filepath)[0])
    model_path = os.path.join(model_dir, "fasttext" if use_char_ngram else "word2vec")
    model.save(model_path, separately=[])
    model.wv.save_word2vec_format(wv_model_filepath, binary=False)
