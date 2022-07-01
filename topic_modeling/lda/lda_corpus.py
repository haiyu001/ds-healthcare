from typing import Dict
import pandas as pd
import json
from utils.config_util import read_config_to_dict
from utils.general_util import save_pdf, make_dir, dump_json_file
import os
from utils.resource_util import get_data_filepath


def get_corpus_word_to_lemma(filter_unigram_filepath: str,
                             corpus_word_to_lemma_filepath: str,
                             corpus_vocab_size: int = 10000,
                             corpus_word_pos_candidates: str = "NOUN,PROPN,ADJ,ADV,VERB") -> Dict[str, str]:
    corpus_word_pos_candidates = [i.strip() for i in corpus_word_pos_candidates.split(",")]
    filter_unigram_pdf = pd.read_csv(filter_unigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    filter_unigram_pdf["top_three_pos"] = filter_unigram_pdf["top_three_pos"].apply(json.loads)
    filter_unigram_pdf = filter_unigram_pdf[filter_unigram_pdf["top_three_pos"].apply(
        lambda x: any(i in corpus_word_pos_candidates for i in x))]
    filter_unigram_pdf = filter_unigram_pdf.groupby("lemma").agg({"word": pd.Series.tolist, "count": sum}).reset_index()
    filter_unigram_pdf = filter_unigram_pdf.sort_values(by="count", ascending=False)
    filter_unigram_pdf = filter_unigram_pdf.head(corpus_vocab_size)

    corpus_word_to_lemma = dict()
    for _, row in filter_unigram_pdf.iterrows():
        lemma, words = row["lemma"], row["word"]
        for word in words:
            corpus_word_to_lemma[word] = lemma
    dump_json_file(corpus_word_to_lemma, corpus_word_to_lemma_filepath)
    return corpus_word_to_lemma


if __name__ == "__main__":

    lda_config_filepath = "/Users/haiyang/github/ds-healthcare/topic_modeling/pipelines/conf/lda_template.cfg"
    lda_config = read_config_to_dict(lda_config_filepath)

    domain_dir = get_data_filepath(lda_config["domain"])
    extraction_dir = os.path.join(domain_dir, lda_config["extraction_folder"])
    topic_modeling_dir = make_dir(os.path.join(domain_dir, lda_config["topic_modeling_folder"]))
    corpus_dir = make_dir(os.path.join(topic_modeling_dir, lda_config["corpus_folder"]))
    corpus_word_to_lemma_filepath = os.path.join(corpus_dir, lda_config["corpus_word_to_lemma_filename"])
    filter_unigram_filepath = os.path.join(extraction_dir, lda_config["filter_unigram_filename"])

    get_corpus_word_to_lemma(filter_unigram_filepath,
                             corpus_word_to_lemma_filepath,
                             lda_config["corpus_vocab_size"],
                             lda_config["corpus_word_pos_candidates"])