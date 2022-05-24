from typing import Set, Dict, Tuple
from double_propagation.absa.enumerations import POS, Polarity
from utils.general_util import load_json_file
from utils.resource_util import get_model_filepath
import pandas as pd
import operator
import json


VALID_OPINION_REX = r"^[a-z][a-z&_]+$"

VALID_ASPECT_REX = r"^[a-z0-9][a-z0-9&_ ]+$"


def load_absa_stop_words() -> Set[str]:
    absa_stop_words_filepath = get_model_filepath("lexicon", "absa_stop_words.json")
    absa_stop_words = load_json_file(absa_stop_words_filepath)
    return absa_stop_words.keys()


def load_absa_seed_opinions() -> Dict[str, int]:
    absa_seed_opinions_filepath = get_model_filepath("lexicon", "absa_seed_opinions.json")
    absa_seed_opinions = load_json_file(absa_seed_opinions_filepath)
    return absa_seed_opinions


def get_sentence_sentiment(sentiment: int) -> str:
    if sentiment == 0:
        return Polarity.NEG.name
    elif sentiment == 2:
        return Polarity.POS.name
    else:
        return Polarity.UNK.name


def norm_pos(tag, pos):
    if tag is None:
        return POS.OTHER.name
    if pos == "NOUN":
        return POS.NN.name
    if pos == "PROPN":
        return POS.NNP.name
    if pos == "ADJ":
        return POS.ADJ.name
    if tag in ["NN", "NNS"] and pos == "PRON":
        return POS.PRON.name
    if tag.startswith("RB"):
        return POS.ADV.name
    if tag == "CC":
        return POS.CC.name
    if tag == "DT" or tag == "PDT":
        return POS.DET.name
    if tag == "EX":
        return POS.EX.name
    if tag == "UH":
        return POS.INTJ.name
    if tag == "LS":
        return POS.LS.name
    if tag == "MD":
        return POS.MD.name
    if tag == "CD":
        return POS.NUM.name
    if tag == "POS":
        return POS.POS.name
    if tag == "IN" or tag == "TO":
        return POS.PREP.name
    if tag.startswith("PRP"):
        return POS.PRON.name
    if tag == "RP":
        return POS.RP.name
    if tag == "SYM":
        return POS.SYM.name
    if tag.startswith("VB"):
        return POS.VB.name
    if tag == "WRB":
        return POS.WH_ADV.name
    if tag == "WDT":
        return POS.WH_DET.name
    if tag.startswith("WP"):
        return POS.WH_PRON.name
    return POS.OTHER.name
