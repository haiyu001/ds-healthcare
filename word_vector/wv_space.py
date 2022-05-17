from typing import Optional, List
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import marisa_trie
import pandas as pd
import numpy as np
import logging
import re


class WordVec(object):

    def __init__(self, txt_vecs_filepath: str, mwe_delimiter: str = "_", use_oov_strategy: bool = False):
        self.txt_vecs_filepath = txt_vecs_filepath
        self.mwe_delimiter = mwe_delimiter
        self.use_oov_strategy = use_oov_strategy
        self.vecs_pdf = load_txt_vecs_to_pdf(self.txt_vecs_filepath, l2_norm=True)
        self.vocab_trie = marisa_trie.Trie(self.get_vocab())

    def _get_prefix_words(self, prefix: str) -> List[str]:
        words = []
        while len(prefix) >= 2:
            words = self.vocab_trie.keys(prefix)
            if words:
                break
            prefix = prefix[:-1]
        return words

    def _get_oov_vec(self, word: str) -> pd.Series:
        words = self._get_prefix_words(word)
        if not words and self.mwe_delimiter in word:
            word = word.split(self.mwe_delimiter)[-1]
            words = self._get_prefix_words(word)
        if words:
            return self.vecs_pdf.loc[words].mean()
        else:
            raise ValueError(f'No any word in vocab starts with "{word}" prefixes.')

    def get_word_vec(self, word) -> pd.Series:
        if word in self.vecs_pdf.index:
            return self.vecs_pdf.loc[word]
        elif self.use_oov_strategy:
            return self._get_oov_vec(word)
        else:
            raise ValueError(f"{word} is out of vocabulary")

    def get_words_vecs(self, words: List[str]) -> pd.DataFrame:
        words_vecs = np.zeros(shape=(len(words), self.vecs_pdf.shape[1]))
        for i, word in enumerate(words):
            words_vecs[i] = self.get_word_vec(word)
        words_vecs_pdf = pd.DataFrame(words_vecs, index=words, dtype="float64")
        return words_vecs_pdf
    
    def get_vocab(self) -> List[str]:
        return self.vecs_pdf.index.tolist()

    def cosine_similarity(self, word1: str, word2: str) -> float:
        vec1 = self.get_word_vec(word1)
        vec2 = self.get_word_vec(word2)
        return vec1.dot(vec2).item()

    def similar_by_word(self, word: str, topn: Optional[int] = None) -> pd.Series:
        word_vec = self.get_word_vec(word)
        similarity_pdf = self.vecs_pdf.dot(word_vec).sort_values(ascending=False)
        if topn:
            return similarity_pdf[:topn]
        else:
            return similarity_pdf

    def get_words_similarity_matrix(self, words: List[str]) -> pd.DataFrame:
        words_vecs_pdf = self.get_words_vecs(words)
        similarity_pdf = words_vecs_pdf.dot(self.vecs_pdf.T)
        return similarity_pdf

    def get_centroid_word(self, words: List[str]) -> str:
        words_vecs_pdf = self.get_words_vecs(words)
        centroid_word_vec = words_vecs_pdf.mean()
        centroid_word_norm_vec = normalize(centroid_word_vec.fillna(0).values.reshape(1, -1))[0]
        similarity_pdf = words_vecs_pdf.dot(centroid_word_norm_vec)
        centroid_word = similarity_pdf.nlargest(1).index[0]
        return centroid_word

    def extract_txt_vecs(self, words: List[str], save_filepath: str):
        if len(words) != len(set(words)):
            raise ValueError("Some words are duplicated")
        if len(set(words) & set(self.get_vocab())) != len(words) and not self.use_oov_strategy:
            raise ValueError("Some words are out of vocabulary, set use_oov_strategy for handling oov extraction")
        num_rows, num_cols = len(words), self.vecs_pdf.shape[1]
        with open(save_filepath, "w", encoding="utf-8") as output:
            output.write(f"{num_rows} {num_cols}\n")
            for word in words:
                word_vec = self.get_word_vec(word)
                word_vec_str = " ".join([str(i) for i in word_vec])
                output.write(word + " " + word_vec_str + "\n")
        logging.info(f"the extracted word vecs dimension: ({num_rows}, {num_cols})")


class ConceptNetWordVec(WordVec):

    def get_word_vec(self, word) -> pd.Series:
        word = re.sub(r"\d\d", "#", word)
        return super().get_word_vec(word)


def load_txt_vecs_to_pdf(txt_vecs_filepath: str, l2_norm: bool = False) -> pd.DataFrame:
    word_vecs = KeyedVectors.load_word2vec_format(txt_vecs_filepath, binary=False)
    vocab = list(word_vecs.key_to_index.keys())
    vecs = word_vecs.vectors
    if l2_norm:
        vecs = normalize(vecs, norm="l2", axis=1)
    index = pd.Index(vocab, name="word")
    return pd.DataFrame(vecs, index=index, dtype="float64")
