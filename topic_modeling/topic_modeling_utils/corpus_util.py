from utils.general_util import dump_json_file
from utils.resource_util import load_stop_words
import pandas as pd
import string
import json


def get_corpus_word_match(filter_unigram_filepath: str,
                          corpus_word_match_filepath: str,
                          corpus_vocab_size: int = 10000,
                          corpus_word_type_candidates: str = "NOUN,PROPN,ADJ,ADV,VERB"):
    corpus_word_type_candidates = [i.strip() for i in corpus_word_type_candidates.split(",")]
    filter_unigram_pdf = pd.read_csv(filter_unigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    filter_unigram_pdf["top_three_pos"] = filter_unigram_pdf["top_three_pos"].apply(json.loads)
    filter_unigram_pdf = filter_unigram_pdf[filter_unigram_pdf["top_three_pos"].apply(
        lambda x: any(i in corpus_word_type_candidates for i in x))]
    filter_unigram_pdf["lemma"] = filter_unigram_pdf["lemma"].str.strip(string.punctuation)
    filter_unigram_pdf = filter_unigram_pdf.groupby("lemma").agg({"word": pd.Series.tolist, "count": sum}).reset_index()
    filter_unigram_pdf = filter_unigram_pdf.sort_values(by="count", ascending=False)
    filter_unigram_pdf = filter_unigram_pdf.head(corpus_vocab_size)

    stop_words = set(load_stop_words())
    corpus_word_match = dict()
    for _, row in filter_unigram_pdf.iterrows():
        lemma, words = row["lemma"], row["word"]
        if lemma not in stop_words:
            for word in words:
                corpus_word_match[word] = lemma
    dump_json_file(corpus_word_match, corpus_word_match_filepath)


def get_corpus_noun_phrase_match_dict(filter_phrase_filepath: str,
                                      corpus_noun_phrase_match_filepath: str,
                                      corpus_phrase_filter_min_count: int,
                                      match_lowercase: bool = True):
    filer_phrase_df = pd.read_csv(filter_phrase_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    filer_phrase_df = filer_phrase_df[filer_phrase_df["count"] >= corpus_phrase_filter_min_count]
    if match_lowercase:
        filer_phrase_df["lemma"] = filer_phrase_df["lemma"].str.lower()
    phrase_lemmas = filer_phrase_df["lemma"].tolist()
    noun_phrase_match_dict = {phrase_lemma: "_".join(phrase_lemma.split()) for phrase_lemma in phrase_lemmas}
    dump_json_file(noun_phrase_match_dict, corpus_noun_phrase_match_filepath)