from typing import List, Optional, Tuple
from annotation.tokenization.preprocessor import REPLACE_EMAIL, REPLACE_URL, REPLACE_HASHTAG, REPLACE_HANDLE
from utils.general_util import save_pdf
from utils.resource_util import load_stop_words
from utils.spark_util import extract_topn_common, write_sdf_to_file, pudf_get_most_common_text
from pyspark.sql.types import ArrayType, StringType, Row, BooleanType
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
from pyspark.ml.feature import NGram
from string import punctuation
import pandas as pd
import collections
import operator
import json


def pudf_is_valid_ngram(ngrams: Column) -> Column:
    def is_valid_ngram(ngrams: pd.Series) -> pd.Series:
        ngrams_words = ngrams.apply(lambda x: x.split())
        is_valid = ngrams_words.apply(lambda x: x[0] not in punctuation and x[-1] not in punctuation)
        return is_valid

    return F.pandas_udf(is_valid_ngram, BooleanType())(ngrams)


def udf_get_words(tokens: Column) -> Column:
    def get_words(tokens: List[Row]) -> List[str]:
        return [token.text.lower() for token in tokens]

    return F.udf(get_words, ArrayType(StringType()))(tokens)


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


def extract_unigram(annotation_sdf: DataFrame,
                    unigram_filepath: str,
                    num_partitions: Optional[int] = None) -> DataFrame:
    tokens_sdf = annotation_sdf.select(F.explode(annotation_sdf.tokens).alias("token")).cache()
    tokens_sdf = tokens_sdf.select(F.lower(F.col("token").text).alias("word"),
                                   F.lower(F.col("token").lemma).alias("lemma"),
                                   F.col("token").pos.alias("pos"))
    count_sdf = tokens_sdf.groupby("word").agg(F.count("*").alias("count"))
    pos_sdf = tokens_sdf.groupby(["word", "pos"]).agg(F.count("*").alias("pos_count"))
    pos_sdf = extract_topn_common(pos_sdf, partition_by="word", key_by="pos", value_by="pos_count", topn=3)
    pos_sdf = pos_sdf.withColumnRenamed("pos", "top_three_pos")
    lemma_sdf = tokens_sdf.groupby(["word", "lemma"]).agg(F.count("*").alias("lemma_count"))
    lemma_sdf = extract_topn_common(lemma_sdf, partition_by="word", key_by="lemma", value_by="lemma_count", topn=3)
    lemma_sdf = lemma_sdf.withColumnRenamed("lemma", "top_three_lemma")
    unigram_sdf = count_sdf.join(pos_sdf, on="word", how="inner").join(lemma_sdf, on="word", how="inner")
    unigram_sdf = unigram_sdf.orderBy(F.asc("word"))
    write_sdf_to_file(unigram_sdf, unigram_filepath, num_partitions)
    return unigram_sdf


def extract_ngram(annotation_sdf: DataFrame,
                  ngram_filepath: str, n: int,
                  ngram_extraction_min_count: Optional[int] = None,
                  num_partitions: Optional[int] = None) -> DataFrame:
    tokens_sdf = annotation_sdf.select(F.col("tokens"))
    words_sdf = tokens_sdf.withColumn("words", udf_get_words(F.col("tokens")))
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    ngram_sdf = ngram.transform(words_sdf).select("ngrams")
    ngram_sdf = ngram_sdf.select(F.explode(F.col("ngrams")).alias("ngram"))
    ngram_sdf = ngram_sdf.groupby(["ngram"]).agg(F.count("*").alias("count"))
    ngram_sdf = ngram_sdf.orderBy(F.desc("count"))
    ngram_sdf = ngram_sdf.filter(pudf_is_valid_ngram(F.col("ngram")))
    if ngram_extraction_min_count:
        ngram_sdf = ngram_sdf.filter(F.col("count") >= ngram_extraction_min_count)
    write_sdf_to_file(ngram_sdf, ngram_filepath, num_partitions)
    return ngram_sdf


def extract_phrase(annotation_sdf: DataFrame,
                   phrase_filepath: str,
                   phrase_extraction_min_count: Optional[int] = None,
                   num_partitions: Optional[int] = None):
    phrase_sdf = annotation_sdf.select(F.explode(annotation_sdf._.phrases).alias("phrase"))
    phrase_sdf = phrase_sdf.select(F.lower(F.col("phrase").text).alias("phrase_lower"),
                                   F.col("phrase").text.alias("text"),
                                   F.to_json(F.col("phrase").phrase_words).alias("phrase_words"),
                                   F.to_json(F.col("phrase").phrase_poses).alias("phrase_poses"),
                                   F.to_json(F.col("phrase").phrase_lemmas).alias("phrase_lemmas"),
                                   F.to_json(F.col("phrase").phrase_deps).alias("phrase_deps"))
    phrase_sdf = phrase_sdf.groupby(["phrase_lower"]).agg(
        pudf_get_most_common_text(F.collect_list("text")).alias("text"),
        F.count(F.col("text")).alias("count"),
        pudf_get_most_common_text(F.collect_list("phrase_words")).alias("phrase_words"),
        pudf_get_most_common_text(F.collect_list("phrase_poses")).alias("phrase_poses"),
        pudf_get_most_common_text(F.collect_list("phrase_lemmas")).alias("phrase_lemmas"),
        pudf_get_most_common_text(F.collect_list("phrase_deps")).alias("phrase_deps")) \
        .orderBy(F.desc("count"))
    if phrase_extraction_min_count:
        phrase_sdf = phrase_sdf.filter(F.col("count") >= phrase_extraction_min_count)
    phrase_sdf = phrase_sdf.select("text", "count", "phrase_words", "phrase_poses", "phrase_lemmas", "phrase_deps")
    write_sdf_to_file(phrase_sdf, phrase_filepath, num_partitions)


def extract_entity(annotation_sdf: DataFrame,
                   entity_filepath: str,
                   entity_extraction_min_count: Optional[int] = None,
                   num_partitions: Optional[int] = None):
    entity_sdf = annotation_sdf.select(F.explode(annotation_sdf.entities).alias("entity"))
    entity_data = entity_sdf.select(F.col("entity")).schema.jsonValue()
    entity_data_fields = [field["name"] for field in entity_data["fields"][0]["type"]["fields"]]
    select_columns = [
        F.lower(F.col("entity").text).alias("text_lower"),
        F.col("entity").entity.alias("entity"),
        F.col("entity").text.alias("text"),
    ]
    agg_columns = [
        pudf_get_most_common_text(F.collect_list("text")).alias("text"),
        pudf_get_most_common_text(F.collect_list("entity")).alias("entity"),
        F.count(F.col("text")).alias("count"),
    ]
    if "negation" in entity_data_fields:
        select_columns.append(F.col("entity").negation.alias("negation"))
        agg_columns.append(F.sum(F.col("negation").cast("int")).alias("negation_count"))

    entity_sdf = entity_sdf.select(*select_columns)
    entity_sdf = entity_sdf.groupby(["text_lower"]).agg(*agg_columns).orderBy(F.asc("entity"), F.desc("count"))
    if entity_extraction_min_count:
        entity_sdf = entity_sdf.filter(F.col("count") >= entity_extraction_min_count)
    entity_sdf = entity_sdf.select("text", "entity", "count", "negation_count")
    write_sdf_to_file(entity_sdf, entity_filepath, num_partitions)


def extract_umls_concept(annotation_sdf: DataFrame,
                         umls_concept_filepath: str,
                         umls_concept_extraction_min_count: Optional[int] = None,
                         num_partitions: Optional[int] = None):
    umls_concept_sdf = annotation_sdf.select(F.explode(annotation_sdf._.umls_concepts).alias("umls_concept"))
    umls_concept_data = umls_concept_sdf.select(F.col("umls_concept")).schema.jsonValue()
    umls_concept_data_fields = [field["name"] for field in umls_concept_data["fields"][0]["type"]["fields"]]
    select_columns = [
        F.lower(F.col("umls_concept").text).alias("text_lower"),
        F.col("umls_concept").concepts.alias("concepts"),
        F.col("umls_concept").text.alias("text"),
    ]
    agg_columns = [
        pudf_get_most_common_text(F.collect_list("text")).alias("text"),
        F.to_json(F.first(F.col("concepts"))).alias("concepts"),
        F.count(F.col("text")).alias("count"),
    ]
    if "negation" in umls_concept_data_fields:
        select_columns.append(F.col("umls_concept").negation.alias("negation"))
        agg_columns.append(F.sum(F.col("negation").cast("int")).alias("negation_count"))

    umls_concept_sdf = umls_concept_sdf.select(*select_columns)
    umls_concept_sdf = umls_concept_sdf.groupby(["text_lower"]).agg(*agg_columns).orderBy(F.desc("count"))
    if umls_concept_extraction_min_count:
        umls_concept_sdf = umls_concept_sdf.filter(F.col("count") >= umls_concept_extraction_min_count)
    umls_concept_sdf = umls_concept_sdf.select("text", "count", "negation_count", "concepts")
    write_sdf_to_file(umls_concept_sdf, umls_concept_filepath, num_partitions)


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

    stop_words = load_stop_words(stop_words_filter_min_count)
    stop_words = stop_words + \
                 [REPLACE_EMAIL.lower(), REPLACE_URL.lower(), REPLACE_HASHTAG.lower(), REPLACE_HANDLE.lower()]
    unigram_pdf = unigram_pdf[unigram_pdf["count"] >= unigram_filter_min_count]
    unigram_pdf = unigram_pdf[unigram_pdf["pos"].isin(["NOUN", "PROPN", "ADJ", "ADV", "VERB"])]
    unigram_pdf = unigram_pdf[~(unigram_pdf["word"].str.startswith("http"))]
    unigram_pdf = unigram_pdf[~(unigram_pdf["word"].isin(stop_words))]
    unigram_pdf = unigram_pdf[unigram_pdf["word"].str.match(r"^[a-z][a-z&_-]+$")]

    unigram_pdf = unigram_pdf.sort_values(by="count", ascending=False)
    unigram_pdf = unigram_pdf[["word", "lemma", "count", "top_three_pos"]]
    save_pdf(unigram_pdf, filter_unigram_filepath)


