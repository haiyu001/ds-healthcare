from typing import Optional, List
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import marisa_trie
import pandas as pd
import numpy as np
import re


class WordVec(object):

    def __init__(self,
                 txt_vecs_filepath: str,
                 from_conceptnet: bool = False,
                 use_oov_strategy: bool = False):
        self.txt_vecs_filepath = txt_vecs_filepath
        self.from_conceptnet = from_conceptnet
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
        if not words and "_" in word:
            word = word.split("_")[-1]
            words = self._get_prefix_words(word)
        if words:
            return self.vecs_pdf.loc[words].mean()
        else:
            raise ValueError(f'No any word in vocab starts with "{word}" prefixes.')

    def _get_word_vec(self, word) -> pd.Series:
        if self.from_conceptnet:
            word = re.sub(r"\d\d", "#", word)
        if word in self.vecs_pdf.index:
            return self.vecs_pdf.loc[word]
        elif self.use_oov_strategy:
            return self._get_oov_vec(word)
        else:
            raise ValueError(f"{word} is out of vocabulary")

    def _get_words_vecs(self, words: List[str]) -> pd.DataFrame:
        words_vecs = np.zeros(shape=(len(words), self.vecs_pdf.shape[1]))
        for i, word in enumerate(words):
            words_vecs[i] = self._get_word_vec(word)
        words_vecs_pdf = pd.DataFrame(words_vecs, index=words, dtype="float64")
        return words_vecs_pdf
    
    def get_vocab(self) -> List[str]:
        return self.vecs_pdf.index.tolist()

    def cosine_similarity(self, word1: str, word2: str) -> float:
        vec1 = self._get_word_vec(word1)
        vec2 = self._get_word_vec(word2)
        return vec1.dot(vec2).item()

    def similar_by_word(self, word: str, topn: Optional[int] = None) -> pd.Series:
        word_vec = self._get_word_vec(word)
        similarity_pdf = self.vecs_pdf.dot(word_vec).sort_values(ascending=False)
        if topn:
            return similarity_pdf[:topn]
        else:
            return similarity_pdf

    def get_words_similarity_matrix(self, words: List[str]) -> pd.DataFrame:
        words_vecs_pdf = self._get_words_vecs(words)
        similarity_pdf = words_vecs_pdf.dot(self.vecs_pdf.T)
        return similarity_pdf

    def get_centroid_word(self, words: List[str]) -> str:
        words_vecs_pdf = self._get_words_vecs(words)
        centroid_word_vec = words_vecs_pdf.mean()
        centroid_word_norm_vec = normalize(centroid_word_vec.fillna(0).values.reshape(1, -1))[0]
        similarity_pdf = words_vecs_pdf.dot(centroid_word_norm_vec)
        centroid_word = similarity_pdf.nlargest(1).index[0]
        return centroid_word

    def extract_txt_vecs(self, words: List[str], save_filepath: str):
        words, vocab = set(words), set(self.get_vocab())
        words_in_vocab = words & vocab
        if len(words_in_vocab) != len(words):
            raise ValueError("Some words are out of vocabulary")
        with open(save_filepath, "w", encoding="utf-8") as output:
            with open(self.txt_vecs_filepath, "r", encoding="utf-8") as input:
                _, num_col = next(input).strip("\n").strip().split()
                output.write(f"{len(words_in_vocab)} {num_col}\n")
                for line in input:
                    word = line.split()[0]
                    if word in words_in_vocab:
                        output.write(line)


def load_txt_vecs_to_pdf(txt_vecs_filepath: str, l2_norm: bool = True) -> pd.DataFrame:
    word_vecs = KeyedVectors.load_word2vec_format(txt_vecs_filepath, binary=False)
    vocab = list(word_vecs.key_to_index.keys())
    vecs = word_vecs.vectors
    if l2_norm:
        vecs = normalize(vecs, norm="l2", axis=1)
    return pd.DataFrame(vecs, index=vocab, dtype="float64")


if __name__ == "__main__":
    from utils.resource_util import get_model_filepath
    from utils.general_util import load_json_file

    conceptnet_txt_vecs_filepath = get_model_filepath("model", "conceptnet", "numberbatch-en-19.08.txt")
    opinion_txt_vecs_filepath = "/Users/haiyang/Desktop/opinion_vecs.txt"

    absa_seed_opinions_filepath = get_model_filepath("lexicon", "absa_seed_opinions.json")
    opinions = list(load_json_file(absa_seed_opinions_filepath).keys())

    # conceptnet_vecs = WordVec(conceptnet_txt_vecs_filepath, from_conceptnet=True)
    # conceptnet_vecs.extract_txt_vecs(opinions, save_filepath=opinion_txt_vecs_filepath)

    opinion_vecs = WordVec(opinion_txt_vecs_filepath, from_conceptnet=False, use_oov_strategy=True)
    print(opinion_vecs.similar_by_word("abandon", topn=10))
    print(opinion_vecs.get_words_similarity_matrix(["abandon", "forsake", "renounce"]))
    print(opinion_vecs.get_centroid_word(["abandon", "forsake", "renounce"]))








    

