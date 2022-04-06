from typing import Dict, Union, List, Any
from utils.config_util import read_config, config_type_casting
from utils.resource_util import get_model_filepath
from stanza.resources.common import process_pipeline_parameters, maintain_processor_list
from spacy import Language
import spacy
import stanza
import json
import os

DEFAULT_SPACY_PACKAGE = "en_core_web_md-3.2.0"


def get_stanza_model_dir() -> str:
    model_dir = get_model_filepath("stanza")
    return model_dir


def get_spacy_model_path(lang: str, package: str) -> str:
    model_path = get_model_filepath("spacy", lang, package, package.rsplit("-", 1)[0], package)
    return model_path


def download_stanza_model(lang: str, package: str = "default", processors: Union[str, Dict[str, str]] = {}):
    stanza.download(lang, get_model_filepath("stanza"), package, processors)


def load_blank_nlp(lang: str, package: str, exclude: List[str] =
                   ["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]) -> Language:
    spacy_model_path = get_spacy_model_path(lang, package)
    blank_nlp = spacy.load(spacy_model_path, exclude=exclude)
    return blank_nlp


def get_stanza_load_list(lang: str = "en",
                         package: str = "default",
                         processors: Union[str, Dict[str, str]] = {}) -> List[List[str]]:
    stanza_dir = get_stanza_model_dir()
    resources_filepath = os.path.join(stanza_dir, "resources.json")
    with open(resources_filepath) as infile:
        resources = json.load(infile)
    lang, _, package, processors = process_pipeline_parameters(lang, stanza_dir, package, processors)
    stanza_load_list = maintain_processor_list(resources, lang, package, processors)
    return stanza_load_list


def read_annotation_config(config_filepath: str) -> Dict[str, Any]:
    config = read_config(config_filepath)
    nlp_model_config = dict(
        use_gpu=config["Annotator"].getboolean('use_gpu'),
        lang=config["Annotator"]["lang"],
        spacy_package=config["Annotator"]["spacy_package"],
        meta_tokenizer_config=config_type_casting(config.items("MetaTokenizer")),
        preprocessor_config=config_type_casting(config.items("Preprocessor")),
        stanza_base_tokenizer_package=config["BaseTokenizer"]["stanza_base_tokenizer_package"],
        normalizer_config=config_type_casting(config.items("Normalizer")),
        stanza_pipeline_config=config_type_casting(config.items("StanzaPipeline")),
        spacy_pipeline_config=config_type_casting(config.items("SpacyPipeline")),
        custom_pipes_config=[(k, {}) for k, v in config_type_casting(config.items("CustomPipes")).items() if v])
    return nlp_model_config


