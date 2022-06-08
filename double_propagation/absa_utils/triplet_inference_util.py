from typing import Dict, List, Tuple
from utils.general_util import load_json_file
from spacy import Language
from spacy.matcher import Matcher


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
