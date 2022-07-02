import string
from typing import List, Tuple
from annotation.annotation_utils.corpus_util import is_valid_token
from annotation.tokenization.preprocessor import REPLACE_EMAIL, REPLACE_URL, REPLACE_HASHTAG, REPLACE_HANDLE
from utils.general_util import save_pdf
from utils.resource_util import load_stop_words
import collections
import json
import operator
import pandas as pd


def _get_noun_phrases_ids(pos_list: List[str], noun_phrase_max_words_count: int = 4) -> List[Tuple[int, int]]:
    res = []
    noun_propn_ids = [i for i, pos in enumerate(pos_list) if pos == "NOUN" or pos == "PROPN"]
    size = len(noun_propn_ids)
    if size >= 2:
        start = end = noun_propn_ids[0]
        i = 1
        while i <= size:
            if i != size and noun_propn_ids[i] == noun_propn_ids[i - 1] + 1:
                end = noun_propn_ids[i]
            else:
                if 2 <= end - start + 1 <= noun_phrase_max_words_count:
                    res.append((start, end + 1))
                start = end = noun_propn_ids[i] if i < size else -1
            i += 1
    return res


def filter_unigram(unigram_filepath: str,
                   filter_unigram_filepath: str,
                   unigram_filter_min_count: int = 5,
                   stop_words_filter_min_count: int = 15):
    unigram_pdf = pd.read_csv(unigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    unigram_pdf["pos_candidates"] = unigram_pdf["top_three_pos"].apply(json.loads)
    unigram_pdf["lemma_candidates"] = unigram_pdf["top_three_lemma"].apply(json.loads)
    unigram_pdf["lemma"] = unigram_pdf["lemma_candidates"].apply(
        lambda x: max(x.items(), key=operator.itemgetter(1))[0])
    unigram_pdf["pos"] = unigram_pdf["pos_candidates"].apply(
        lambda x: max(x.items(), key=operator.itemgetter(1))[0])

    stop_words = load_stop_words(stop_words_filter_min_count) + \
                 [REPLACE_EMAIL.lower(), REPLACE_URL.lower(), REPLACE_HASHTAG.lower(), REPLACE_HANDLE.lower()]

    unigram_pdf = unigram_pdf[unigram_pdf["count"] >= unigram_filter_min_count]
    unigram_pdf = unigram_pdf[unigram_pdf["pos"].isin(["NOUN", "PROPN", "ADJ", "ADV", "VERB"])]
    unigram_pdf["check"] = unigram_pdf["word"].str.strip(string.punctuation)
    unigram_pdf = unigram_pdf[unigram_pdf["check"].str.match(r"^[a-z0-9][a-z0-9&_-]+$")]
    unigram_pdf = unigram_pdf[unigram_pdf["check"].apply(lambda x: is_valid_token(x))]
    unigram_pdf = unigram_pdf[~(unigram_pdf["check"].isin(stop_words))]

    unigram_pdf = unigram_pdf[["word", "lemma", "count", "top_three_pos"]]
    unigram_pdf = unigram_pdf.sort_values(by="count", ascending=False)
    save_pdf(unigram_pdf, filter_unigram_filepath)


def filter_phrase(phrase_filepath: str,
                  filter_phrase_filepath: str,
                  noun_phrase_words_max_count: int = 4):
    phrase_pdf = pd.read_csv(phrase_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    phrase_pdf["phrase_words"] = phrase_pdf["phrase_words"].apply(json.loads)
    phrase_pdf["phrase_poses"] = phrase_pdf["phrase_poses"].apply(json.loads)
    phrase_pdf["phrase_lemmas"] = phrase_pdf["phrase_lemmas"].apply(json.loads)
    phrase_pdf["phrase_deps"] = phrase_pdf["phrase_deps"].apply(json.loads)
    noun_phrase_to_count = collections.defaultdict(int)
    noun_phrase_to_word_list = collections.defaultdict(list)
    noun_phrase_to_lemma_list = collections.defaultdict(list)
    noun_phrase_to_dep_list = collections.defaultdict(list)
    for _, row in phrase_pdf.iterrows():
        phrase_words, phrase_poses, phrase_lemmas, phrase_deps, phrase_count = \
            row["phrase_words"], row["phrase_poses"], row["phrase_lemmas"], row["phrase_deps"], row["count"]
        for start, end in _get_noun_phrases_ids(phrase_poses, noun_phrase_words_max_count):
            if end - start > 1:
                noun_phrase = tuple([i.strip().lower() for i in phrase_words[start: end]])
                noun_phrase_to_count[noun_phrase] += phrase_count
                noun_phrase_to_word_list[noun_phrase].append(tuple([i.strip() for i in phrase_words[start: end]]))
                noun_phrase_to_lemma_list[noun_phrase].append(tuple(phrase_lemmas[start: end]))
                noun_phrase_to_dep_list[noun_phrase].append(tuple(phrase_deps[start: end]))
    noun_phrase_to_words = {phrase: collections.Counter(word_list).most_common(1)[0][0]
                            for phrase, word_list in noun_phrase_to_word_list.items()}
    noun_phrase_to_word_lemmas = {phrase: collections.Counter(lemma_list).most_common(1)[0][0]
                                  for phrase, lemma_list in noun_phrase_to_lemma_list.items()}
    noun_phrase_to_word_deps = {phrase: collections.Counter(dep_list).most_common(1)[0][0]
                                for phrase, dep_list in noun_phrase_to_dep_list.items()}
    noun_phrase_record_list = []
    for noun_phrase in noun_phrase_to_count:
        words, word_lemmas, word_deps = noun_phrase_to_words[noun_phrase], noun_phrase_to_word_lemmas[noun_phrase], \
                                        noun_phrase_to_word_deps[noun_phrase]
        if word_deps[0] == "nummod" or \
                (word_deps[0] == "amod" and word_deps[1] != "punct") or \
                len(word_lemmas[0]) < 2 or \
                any("'" in lemma for lemma in word_lemmas):
            continue
        noun_phrase_record_list.append({
            "noun_phrase": " ".join(words),
            "lemma": " ".join(word_lemmas),
            "count": noun_phrase_to_count[noun_phrase],
        })
    noun_phrase_pdf = pd.DataFrame(noun_phrase_record_list).sort_values(by="count", ascending=False)
    save_pdf(noun_phrase_pdf, filter_phrase_filepath)
