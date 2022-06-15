from collections import Counter
from typing import Dict, List, Tuple
from pyspark.sql import Column
from pyspark.sql.types import StringType, ArrayType
import pyspark.sql.functions as F
from utils.general_util import load_json_file
from spacy import Language
from spacy.matcher import Matcher
import pandas as pd
import json


def load_aspect_hierarchy(aspect_filepath: str) -> Dict[str, str]:
    child_to_parent = load_json_file(aspect_filepath)
    categories = set(child_to_parent.values())
    aspect_to_hierarchy = dict()
    for aspect, parent in child_to_parent.items():
        hierarchy = aspect
        if aspect not in categories:
            while parent is not None:
                hierarchy = f"{parent}::{hierarchy}"
                parent = child_to_parent[parent]
            aspect_to_hierarchy[aspect] = hierarchy
    return aspect_to_hierarchy


def get_aspect_matcher(nlp: Language, aspects: List[str]) -> Matcher:
    aspect_matcher = Matcher(nlp.vocab)
    patterns = [[{"LOWER": word} for word in aspect.split()] for aspect in aspects]
    aspect_matcher.add("aspects", patterns)
    return aspect_matcher


def get_mark_sign(sentiment_score: int) -> Tuple[str, str]:
    if sentiment_score > 0.0:
        return "{{", "}}"
    elif sentiment_score < 0.0:
        return "<<", ">>"
    else:
        return "((", "))"


def pudf_score_to_polarity(score: Column) -> Column:
    def score_to_polarity(score: pd.Series) -> pd.Series:
        polarity = score.apply(lambda x: "NEU" if x == 0 else "POS" if x > 0 else "NEG")
        return polarity

    return F.pandas_udf(score_to_polarity, StringType())(score)


def udf_collect_opinions(opinion_dicts: Column) -> Column:
    def collect_opinions(opinion_dicts):
        opinions = [opinion_dict["opinion"] for opinion_dict in opinion_dicts]
        return opinions

    return F.udf(collect_opinions, ArrayType(StringType()))(opinion_dicts)


def udf_get_top_common_values(values_col, topn=3):
    def get_top_common_values(values_col):
        if len(values_col) == 0:
            return None
        else:
            return json.dumps(dict(Counter(values_col).most_common(topn)), ensure_ascii=False)
    return F.udf(get_top_common_values, StringType())(values_col)
