from typing import Dict, Optional, List, Tuple, Any
from annotation.annotation_utils.corpus_util import pudf_get_corpus_line
from double_propagation.absa_utils.triplet_inference_util import load_aspect_hierarchy
from utils.general_util import dump_json_file, dump_pickle_file, load_pickle_file, load_json_file
from utils.spark_util import write_sdf_to_file
from pyspark.sql import DataFrame, Column
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from scipy.sparse import csc_matrix
from gensim import corpora, matutils
import scipy
import json
import logging


def udf_get_doc_text_by_word_match(tokens: Column, word_match: Dict[str, str], to_lower: bool = True) -> Column:
    def get_doc_text_by_word_match(doc_tokens):
        doc_token_matches = []
        for token in doc_tokens:
            token_text = token.text
            if to_lower:
                token_text = token_text.lower()
            if token_text in word_match:
                token_match = word_match[token_text]
                doc_token_matches.append(token_match)
        return " ".join(doc_token_matches)

    return F.udf(get_doc_text_by_word_match, StringType())(tokens)


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


def get_absa_word_match(aspect_filepath: str,
                        opinion_filepath: str,
                        corpus_word_match_filepath: str,
                        corpus_word_type_candidates: str = "aspect,opinion"):
    aspects = list(load_aspect_hierarchy(aspect_filepath).keys())
    aspect_word_match = {aspect.lower(): aspect for aspect in aspects if " " not in aspect}
    opinions = load_json_file(opinion_filepath)
    opinion_word_match = {opinion.lower(): opinion for opinion in opinions}
    absa_word_match = {}
    corpus_word_type_candidates = [i.strip() for i in corpus_word_type_candidates.split(",")]
    for word_type in corpus_word_type_candidates:
        if word_type == "aspect":
            absa_word_match.update(aspect_word_match)
        elif word_type == "opinion":
            absa_word_match.update(opinion_word_match)
        else:
            raise ValueError(f"Unsupported word_type of {word_type}")
    dump_json_file(absa_word_match, corpus_word_match_filepath)


def get_absa_noun_phrase_match_dict(aspect_filepath: str,
                                    corpus_noun_phrase_match_filepath: str,
                                    match_lowercase: bool = True):
    aspects = list(load_aspect_hierarchy(aspect_filepath).keys())
    if match_lowercase:
        aspects = {i.lower() for i in aspects}
    noun_phrase_match_dict = {aspect: "_".join(aspect.split()) for aspect in aspects if " " in aspect}
    dump_json_file(noun_phrase_match_dict, corpus_noun_phrase_match_filepath)


def load_docs_from_corpus(corpus_filepath: str, corpus_doc_id_col: str) -> List[Tuple[str, List[str]]]:
    docs = []
    with open(corpus_filepath, "r") as input:
        for line in input:
            line_data = json.loads(line)
            docs.append((line_data[corpus_doc_id_col], line_data["corpus_line"].split()))
    return docs


def build_lda_corpus_by_annotation(annotation_sdf: DataFrame,
                                   lang: str,
                                   spacy_package: str,
                                   corpus_filepath: str,
                                   word_match: Dict[str, str],
                                   ngram_match_dict: Optional[Dict[str, str]] = None,
                                   match_lowercase: bool = True,
                                   num_partitions: Optional[int] = None,
                                   metadata_fields_to_keep: Optional[str] = None):
    corpus_sdf = annotation_sdf.select(
        F.col("_").metadata.alias("metadata"),
        udf_get_doc_text_by_word_match(F.col("tokens"), word_match, match_lowercase).alias("text"))
    corpus_sdf = corpus_sdf.withColumn("corpus_line",
                                       pudf_get_corpus_line(F.col("text"), lang, spacy_package, ngram_match_dict))
    corpus_sdf = corpus_sdf.select(udf_get_corpus_line_with_metadata(F.col("metadata"),
                                                                     F.col("corpus_line"),
                                                                     metadata_fields_to_keep))
    write_sdf_to_file(corpus_sdf, corpus_filepath, num_partitions)


def save_mallet_corpus(corpus_doc_id_col: str,
                       corpus_filepath: str,
                       mallet_docs_filepath: Optional[str] = None,
                       mallet_id2word_filepath: Optional[str] = None,
                       mallet_corpus_filepath: Optional[str] = None,
                       mallet_corpus_csc_filepath: Optional[str] = None,
                       mallet_vocab_filepath: Optional[str] = None):
    mallet_docs_with_id = load_docs_from_corpus(corpus_filepath, corpus_doc_id_col)
    mallet_docs = [doc for doc_id, doc in mallet_docs_with_id]
    mallet_id2word = corpora.Dictionary(mallet_docs)
    # mallet_corpus: List[Tuple["doc_id", List[Tuple["term_vocab_id", "term_count"]]]]
    mallet_corpus = [(doc_id, mallet_id2word.doc2bow(doc)) for doc_id, doc in mallet_docs_with_id]
    mallet_corpus_csc = matutils.corpus2csc([bow for doc_id, bow in mallet_corpus])
    mallet_vocab = {mallet_id2word.get(i): i for i in mallet_id2word}

    logging.info(f"\n{'=' * 100}\n"
                 f"corpus number of docs:  {len(mallet_docs)}\n"
                 f"corpus vocabulary size: {len(mallet_vocab)}\n"
                 f"{'=' * 100}\n")

    if mallet_docs_filepath:
        dump_pickle_file(mallet_docs, mallet_docs_filepath)
    if mallet_docs_filepath and mallet_id2word_filepath:
        mallet_id2word.save(mallet_id2word_filepath)
    if mallet_corpus_filepath:
        dump_pickle_file(mallet_corpus, mallet_corpus_filepath)
    if mallet_corpus_csc_filepath:
        scipy.sparse.save_npz(mallet_corpus_csc_filepath, mallet_corpus_csc)
    if mallet_vocab_filepath:
        dump_json_file(mallet_vocab, mallet_vocab_filepath)


def load_mallet_corpus(mallet_docs_filepath: Optional[str] = None,
                       mallet_id2word_filepath: Optional[str] = None,
                       mallet_corpus_filepath: Optional[str] = None,
                       mallet_corpus_csc_filepath: Optional[str] = None) -> Tuple[Any, ...]:
    res = []
    if mallet_docs_filepath:
        mallet_docs: List[List[str]] = load_pickle_file(mallet_docs_filepath)
        res.append(mallet_docs)
    if mallet_id2word_filepath:
        mallet_id2word: corpora.Dictionary = corpora.Dictionary.load(mallet_id2word_filepath)
        res.append(mallet_id2word)
    if mallet_corpus_filepath:
        mallet_corpus: List[Tuple[str, List[Tuple[int, int]]]] = load_pickle_file(mallet_corpus_filepath)
        res.append(mallet_corpus)
    if mallet_corpus_csc_filepath:
        mallet_corpus_csc: csc_matrix = scipy.sparse.load_npz(mallet_corpus_csc_filepath)
        res.append(mallet_corpus_csc)
    return tuple(res)
