from typing import Set, Dict, Optional, Iterator, List, Tuple
from utils.general_util import save_pdf
from gensim.models.fasttext import FastText
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
import editdistance
import pandas as pd
import json


def pudf_get_valid_suggestions(suggestions: Column, unigram: Set[str]) -> Column:
    def get_valid_suggestions(suggestions: pd.Series) -> pd.Series:
        valid_suggestions = suggestions.apply(lambda x: [i for i in x if i.lower() in unigram])
        return valid_suggestions

    return F.pandas_udf(get_valid_suggestions, ArrayType(StringType()))(suggestions)


def pudf_get_misspelling_topn_similar_words(misspelling_iter: Column,
                                            wv_model_filepath: str,
                                            wv_filter_min_similarity: float = 0.8) -> Column:
    def test(misspelling_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        wv_model = FastText.load(wv_model_filepath).wv
        for misspelling in misspelling_iter:
            similar_words = misspelling.apply(wv_model.similar_by_word, topn=10)
            similar_words_dict = similar_words.apply(_filter_by_wv_min_similarity,
                                                     wv_filter_min_similarity=wv_filter_min_similarity)
            similar_words_json_str = similar_words_dict.apply(json.dumps, ensure_ascii=False)
            yield similar_words_json_str

    return F.pandas_udf(test, StringType())(misspelling_iter)


def _filter_by_wv_min_similarity(word_similarity: List[Tuple[str, float]],
                                 wv_filter_min_similarity: float) -> Dict[str, float]:
    return {word: similarity for word, similarity in word_similarity if similarity >= wv_filter_min_similarity}


def _set_suggesions_similarity_by_top_similar_words(suggestions: List[str],
                                                    top_similar_words: Dict[str, float]) -> Optional[Dict[str, float]]:
    suggestions_similarity = {}
    for suggestion in suggestions:
        suggestion_lower = suggestion.lower()
        if suggestion_lower in top_similar_words:
            suggestions_similarity[suggestion] = top_similar_words.get(suggestion_lower)
    return suggestions_similarity if suggestions_similarity else None


def _get_valid_word_pos(word_pos_freq_dict: Dict[str, int],
                        word_pos_filter_min_percent: float) -> Tuple[List[str], int]:
    word_pos_count = sum(word_pos_freq_dict.values())
    threshold = word_pos_count * word_pos_filter_min_percent
    valid_word_pos = [k for k, v in word_pos_freq_dict.items() if v >= threshold]
    return valid_word_pos, word_pos_count


def _load_word_to_pos(unigram_filepath: str) -> Dict[str, Dict[str, int]]:
    unigram_pdf = pd.read_csv(unigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    unigram_pdf["pos"] = unigram_pdf["top_three_pos"].apply(json.loads)
    word_to_pos = dict(zip(unigram_pdf["word"], unigram_pdf["pos"]))
    return word_to_pos


def is_valid_correction_candidate(misspelling: str, correction_lower: str) -> bool:
    if correction_lower.endswith(misspelling) or misspelling.endswith(correction_lower) or \
            misspelling[1:] == correction_lower[1:] or misspelling == (correction_lower + "s") or \
            (misspelling in correction_lower and not correction_lower.startswith(misspelling)):
        return False
    return True


def _get_correction(misspelling: str,
                    suggestions_similarity: Dict[str, float],
                    word_to_pos: Dict[str, Dict[str, int]],
                    spell_canonicalization_suggestion_filter_min_count: int,
                    spell_canonicalization_edit_distance_filter_max_count: int,
                    spell_canonicalization_misspelling_filter_max_percent: float,
                    spell_canonicalization_word_pos_filter_min_percent: float) -> \
        Tuple[Optional[str], Optional[int], Optional[float]]:
    edit_distance_threshold = 1 if len(misspelling) <= 4 else spell_canonicalization_edit_distance_filter_max_count
    misspelling_pos, misspelling_count = _get_valid_word_pos(word_to_pos[misspelling],
                                                             spell_canonicalization_word_pos_filter_min_percent)
    correction_candidates = []
    if suggestions_similarity:
        for suggestion, similarity in suggestions_similarity.items():
            suggestion_lower = suggestion.lower()
            suggestion_pos, suggestion_count = _get_valid_word_pos(word_to_pos[suggestion_lower],
                                                                   spell_canonicalization_word_pos_filter_min_percent)
            common_pos = set(misspelling_pos).intersection(set(suggestion_pos))
            edit_distance = editdistance.eval(suggestion_lower, misspelling)
            if common_pos and edit_distance <= edit_distance_threshold and \
                    misspelling_count <= suggestion_count * spell_canonicalization_misspelling_filter_max_percent and \
                    is_valid_correction_candidate(misspelling, suggestion_lower):
                correction_candidates.append((suggestion, suggestion_count, similarity, edit_distance))

    correction_candidates = sorted(correction_candidates, key=lambda x: (x[-1], -x[-2]))
    corrections = {t[0] for t in correction_candidates}
    for correction, correction_count, similarity, _ in correction_candidates:
        if correction.islower() or \
                ("PROPN" in common_pos and misspelling_count >= spell_canonicalization_suggestion_filter_min_count):
            if correction.isupper() and correction.title() in corrections:
                correction = correction.title()
            return correction, correction_count, similarity
    return None, None, None


def get_spell_canonicalization_candidates(unigram_sdf: DataFrame,
                                          annotation_sdf: DataFrame,
                                          spell_canonicalization_candidates_filepath: str,
                                          spell_canonicalization_suggestion_filter_min_count: int = 5,
                                          num_partitions: int = 1):
    unigram_sdf = unigram_sdf.filter(F.col("count") >= spell_canonicalization_suggestion_filter_min_count)
    unigram = set([x.word for x in unigram_sdf.select("word").distinct().collect()])
    misspelling_sdf = annotation_sdf.select(F.explode(annotation_sdf._.misspellings).alias("misspelling"))
    misspelling_sdf = misspelling_sdf.select(F.lower(F.col("misspelling").text).alias("misspelling"),
                                             F.size(F.col("misspelling").ids).alias("count"),
                                             F.col("misspelling").suggestions.alias("suggestions"))
    misspelling_sdf = misspelling_sdf.groupby(["misspelling"]).agg(
        F.sum("count").alias("misspelling_count"),
        F.array_distinct(F.flatten(F.collect_set("suggestions"))).alias("suggestions"))
    spell_canonicalization_candidates_sdf = misspelling_sdf.withColumn(
        "suggestions", pudf_get_valid_suggestions(F.col("suggestions"), unigram))
    spell_canonicalization_candidates_sdf = spell_canonicalization_candidates_sdf.filter(
        F.size("suggestions") > 0).orderBy(F.desc("misspelling_count"))
    spell_canonicalization_candidates_sdf = spell_canonicalization_candidates_sdf.withColumn(
        "suggestions", F.to_json(F.col("suggestions")))
    write_sdf_to_file(spell_canonicalization_candidates_sdf, spell_canonicalization_candidates_filepath, num_partitions)


def get_spell_canonicalization(spell_canonicalization_candidates_sdf: DataFrame,
                               unigram_filepath: str,
                               wv_model_filepath: str,
                               spell_canonicalization_filepath: str,
                               spell_canonicalization_suggestion_filter_min_count: int = 5,
                               spell_canonicalization_edit_distance_filter_max_count: int = 2,
                               spell_canonicalization_misspelling_filter_max_percent: float = 0.25,
                               spell_canonicalization_word_pos_filter_min_percent: float = 0.25,
                               wv_spell_canonicalization_filter_min_similarity: float = 0.8):
    word_to_pos = _load_word_to_pos(unigram_filepath)
    spell_canonicalization_candidates_sdf = spell_canonicalization_candidates_sdf.withColumn(
        "top_similar_words", pudf_get_misspelling_topn_similar_words(F.col("misspelling"), wv_model_filepath,
                                                                     wv_spell_canonicalization_filter_min_similarity))
    pdf = spell_canonicalization_candidates_sdf.toPandas()
    pdf["suggestions"] = pdf["suggestions"].apply(json.loads)
    pdf["top_similar_words"] = pdf["top_similar_words"].apply(json.loads)
    pdf["suggestions_similarity"] = pdf.apply(
        lambda x: _set_suggesions_similarity_by_top_similar_words(x["suggestions"], x["top_similar_words"]), axis=1)
    pdf[["correction", "correction_count", "similarity"]] = pdf.apply(lambda x: _get_correction(
        x["misspelling"], x["suggestions_similarity"], word_to_pos,
        spell_canonicalization_suggestion_filter_min_count,
        spell_canonicalization_edit_distance_filter_max_count,
        spell_canonicalization_misspelling_filter_max_percent,
        spell_canonicalization_word_pos_filter_min_percent), axis=1, result_type="expand")
    pdf = pdf[["misspelling", "correction", "misspelling_count", "correction_count", "similarity"]]
    pdf = pdf.dropna(subset=["correction"]).sort_values(by="misspelling_count", ascending=False)
    save_pdf(pdf, spell_canonicalization_filepath)


if __name__ == "__main__":
    from annotation.annotation_utils.annotator_util import read_annotation_config
    from annotation.components.annotator import load_annotation
    from utils.resource_util import get_data_filepath, get_repo_dir
    from utils.spark_util import get_spark_session, write_sdf_to_file
    import os

    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)

    domain_dir = get_data_filepath(annotation_config["domain"])
    canonicalization_dir = os.path.join(domain_dir, annotation_config["canonicalization_folder"])
    canonicalization_wv_folder = annotation_config["canonicalization_wv_folder"]

    canonicalization_annotation_dir = os.path.join(
        canonicalization_dir, annotation_config["canonicalization_annotation_folder"])
    canonicalization_extraction_dir = os.path.join(
        canonicalization_dir, annotation_config["canonicalization_extraction_folder"])
    unigram_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["canonicalization_unigram_filename"])
    spell_canonicalization_candidates_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["spell_canonicalization_candidates_filename"])
    spell_canonicalization_filepath = os.path.join(
        canonicalization_extraction_dir, annotation_config["spell_canonicalization_filename"])

    spark_cores = 6
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="WARN")

    # ================================ spell canonicalization candidates =====================================

    unigram_sdf = spark.read.csv(unigram_filepath, header=True, quote='"', escape='"', inferSchema=True)
    canonicalization_annotation_sdf = load_annotation(
        spark, canonicalization_annotation_dir, annotation_config["drop_non_english"])
    get_spell_canonicalization_candidates(unigram_sdf,
                                          canonicalization_annotation_sdf,
                                          spell_canonicalization_candidates_filepath,
                                          annotation_config["spell_canonicalization_suggestion_filter_min_count"])

    # ===================================== spell canonicalization ==========================================

    spell_canonicalization_candidates_sdf = spark.read.csv(
        spell_canonicalization_candidates_filepath, header=True, quote='"', escape='"', inferSchema=True)
    wv_model_filepath = os.path.join(canonicalization_dir, canonicalization_wv_folder, "model", "fasttext")
    get_spell_canonicalization(spell_canonicalization_candidates_sdf,
                               unigram_filepath,
                               wv_model_filepath,
                               spell_canonicalization_filepath,
                               annotation_config["spell_canonicalization_suggestion_filter_min_count"],
                               annotation_config["spell_canonicalization_edit_distance_filter_max_count"],
                               annotation_config["spell_canonicalization_misspelling_filter_max_percent"],
                               annotation_config["spell_canonicalization_word_pos_filter_min_percent"],
                               annotation_config["wv_spell_canonicalization_filter_min_similarity"])
