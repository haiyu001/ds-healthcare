from typing import Dict, Optional, List, Tuple
from annotation.annotation_utils.corpus_util import pudf_get_corpus_line
from utils.general_util import dump_json_file, dump_pickle_file, load_pickle_file
from utils.spark_util import write_sdf_to_file
from pyspark.sql import DataFrame, Column
from pyspark.sql.types import StringType
from scipy.sparse import csc_matrix
from gensim import corpora, matutils
import pyspark.sql.functions as F
import pandas as pd
import scipy
import json
import string
import logging


def udf_get_doc_text_by_token_lemmas(tokens: Column, word_to_lemma: Dict[str, str], to_lower: bool = True) -> Column:
    def get_doc_lemmas(doc_tokens):
        doc_token_lemmas = []
        for token in doc_tokens:
            token_text = token.text
            if token_text in word_to_lemma:
                token_lemma = word_to_lemma[token_text].lower() if to_lower else word_to_lemma[token_text]
                doc_token_lemmas.append(token_lemma)
        return " ".join(doc_token_lemmas)

    return F.udf(get_doc_lemmas, StringType())(tokens)


def udf_get_corpus_line_with_metadata(metadata: Column,
                                      corpus_line: Column,
                                      metadata_fields_to_keep: Optional[str] = None):
    def get_corpus_line_with_metadata(metadata, corpus_line):
        corpus_line_dict = {"corpus_line": corpus_line}
        if metadata_fields_to_keep:
            corpus_line_metadata = {field: metadata[field] for field in metadata_fields_to_keep.split(",")}
            corpus_line_dict.update(corpus_line_metadata)
        return json.dumps(corpus_line_dict, ensure_ascii=False)

    return F.udf(get_corpus_line_with_metadata, StringType())(metadata, corpus_line)


def get_corpus_word_to_lemma(filter_unigram_filepath: str,
                             corpus_word_to_lemma_filepath: str,
                             corpus_vocab_size: int = 10000,
                             corpus_word_pos_candidates: str = "NOUN,PROPN,ADJ,ADV,VERB") -> Dict[str, str]:
    corpus_word_pos_candidates = [i.strip() for i in corpus_word_pos_candidates.split(",")]
    filter_unigram_pdf = pd.read_csv(filter_unigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    filter_unigram_pdf["top_three_pos"] = filter_unigram_pdf["top_three_pos"].apply(json.loads)
    filter_unigram_pdf = filter_unigram_pdf[filter_unigram_pdf["top_three_pos"].apply(
        lambda x: any(i in corpus_word_pos_candidates for i in x))]
    filter_unigram_pdf["lemma"] = filter_unigram_pdf["lemma"].str.strip(string.punctuation)
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


def get_corpus_noun_phrase_match_dict(filter_phrase_filepath: str,
                                      corpus_noun_phrase_match_filepath: str,
                                      corpus_phrase_filter_min_count: int,
                                      match_lowercase: bool = True) -> Dict[str, str]:
    filer_phrase_df = pd.read_csv(filter_phrase_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    filer_phrase_df = filer_phrase_df[filer_phrase_df["count"] >= corpus_phrase_filter_min_count]
    if match_lowercase:
        filer_phrase_df["lemma"] = filer_phrase_df["lemma"].str.lower()
    phrase_lemmas = filer_phrase_df["lemma"].tolist()
    noun_phrase_match_dict = {phrase_lemma: "_".join(phrase_lemma.split()) for phrase_lemma in phrase_lemmas}
    dump_json_file(noun_phrase_match_dict, corpus_noun_phrase_match_filepath)
    return noun_phrase_match_dict


def load_docs_from_corpus(corpus_filepath: str) -> List[List[str]]:
    docs = []
    with open(corpus_filepath, "r") as input:
        for line in input:
            line_data = json.loads(line)
            docs.append(line_data["corpus_line"].split())
    return docs


def build_lda_corpus_by_annotation(annotation_sdf: DataFrame,
                                   lang: str,
                                   spacy_package: str,
                                   corpus_filepath: str,
                                   word_to_lemma: Dict[str, str],
                                   ngram_match_dict: Optional[Dict[str, str]] = None,
                                   match_lowercase: bool = True,
                                   num_partitions: Optional[int] = None,
                                   metadata_fields_to_keep: Optional[str] = None):
    corpus_sdf = annotation_sdf.select(
        F.col("_").metadata.alias("metadata"),
        udf_get_doc_text_by_token_lemmas(F.col("tokens"), word_to_lemma, match_lowercase).alias("text"))
    corpus_sdf = corpus_sdf.withColumn("corpus_line",
                                       pudf_get_corpus_line(F.col("text"), lang, spacy_package, ngram_match_dict))
    corpus_sdf = corpus_sdf.select(udf_get_corpus_line_with_metadata(F.col("metadata"),
                                                                     F.col("corpus_line"),
                                                                     metadata_fields_to_keep))
    write_sdf_to_file(corpus_sdf, corpus_filepath, num_partitions)


def save_mallet_corpus(corpus_filepath: str,
                       mallet_docs_filepath: str,
                       mallet_id2word_filepath: str,
                       mallet_corpus_filepath: str,
                       mallet_corpus_csc_filepath: str,
                       mallet_vocab_filepath: str):
    mallet_docs = load_docs_from_corpus(corpus_filepath)
    mallet_id2word = corpora.Dictionary(mallet_docs)
    mallet_corpus = [mallet_id2word.doc2bow(doc) for doc in mallet_docs]  # list of list of (vocab_id, count)
    mallet_corpus_csc = matutils.corpus2csc(mallet_corpus)
    mallet_vocab = {mallet_id2word.get(i): i for i in mallet_id2word}

    dump_pickle_file(mallet_docs, mallet_docs_filepath)
    mallet_id2word.save(mallet_id2word_filepath)
    dump_pickle_file(mallet_corpus, mallet_corpus_filepath)
    scipy.sparse.save_npz(mallet_corpus_csc_filepath, mallet_corpus_csc)
    dump_json_file(mallet_vocab, mallet_vocab_filepath)
    logging.info(f"\n{'=' * 100}\n"
                 f"corpus number of docs:  {len(mallet_docs)}\n"
                 f"corpus vocabulary size: {len(mallet_vocab)}\n"
                 f"{'=' * 100}\n")


def load_mallet_corpus(mallet_docs_filepath: str,
                       mallet_id2word_filepath: str,
                       mallet_corpus_filepath: str,
                       mallet_corpus_csc_filepath: str) -> \
        Tuple[List[List[str]], corpora.Dictionary, List[List[Tuple[int, int]]], csc_matrix]:
    mallet_docs = load_pickle_file(mallet_docs_filepath)
    mallet_id2word = corpora.Dictionary.load(mallet_id2word_filepath)
    mallet_corpus = load_pickle_file(mallet_corpus_filepath)
    mallet_corpus_csc = scipy.sparse.load_npz(mallet_corpus_csc_filepath)
    return mallet_docs, mallet_id2word, mallet_corpus, mallet_corpus_csc
